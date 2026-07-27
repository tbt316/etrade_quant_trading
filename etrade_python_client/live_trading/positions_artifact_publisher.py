"""Writer-only atomic publisher for the static positions artifact."""

from __future__ import annotations

import fcntl
import hashlib
import os
import secrets
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from live_trading.positions_artifact import (
    ArtifactMetadata,
    MAX_ARTIFACT_FUTURE_SKEW_SECONDS,
    MAX_POSITIONS_ARTIFACT_BYTES,
    PositionsArtifactError,
    PositionsArtifactSigningKey,
    PositionsSnapshot,
    inspect_positions_html,
    render_positions_html,
)
from live_trading.runtime_config import (
    RuntimeConfig,
    positions_artifact_runtime_binding,
)


MIN_PRODUCER_FRESHNESS_WINDOW_SECONDS = 60


class PositionsArtifactPublishError(RuntimeError):
    """A publication failed before a durable commit was acknowledged."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class PositionsArtifactCommitUnknown(PositionsArtifactPublishError):
    """The replacement may be visible but its durable state is ambiguous."""

    def __init__(self, sha256: str):
        super().__init__("positions_artifact_commit_unknown")
        self.sha256 = sha256


@dataclass(frozen=True, slots=True)
class PublishReceipt:
    state: str
    sha256: str
    size_bytes: int
    source_as_of: str
    source_generation: str

    def __post_init__(self) -> None:
        if self.state not in {"committed_durable", "already_current"}:
            raise ValueError("publish receipt state is invalid")


class PositionsArtifactPublisher:
    """Serialize and durably replace one configured artifact."""

    __slots__ = (
        "path",
        "max_source_age_seconds",
        "broker_environment",
        "runtime_binding",
        "_signing_key",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        max_source_age_seconds: int,
        signing_key: PositionsArtifactSigningKey,
        broker_environment: str,
        runtime_binding: str,
    ) -> None:
        candidate = Path(path)
        if (
            not candidate.is_absolute()
            or not candidate.name
            or candidate.name in {".", ".."}
        ):
            raise ValueError("path must identify an absolute artifact file")
        if (
            type(max_source_age_seconds) is not int
            or not 1 <= max_source_age_seconds <= 86_400
        ):
            raise ValueError("max_source_age_seconds is invalid")
        if type(signing_key) is not PositionsArtifactSigningKey:
            raise TypeError(
                "signing_key must be an exact PositionsArtifactSigningKey"
            )
        if broker_environment not in {"sandbox", "production"}:
            raise ValueError("broker_environment is invalid")
        if (
            type(runtime_binding) is not str
            or len(runtime_binding) != 64
            or any(
                character not in "0123456789abcdef"
                for character in runtime_binding
            )
        ):
            raise ValueError("runtime_binding is invalid")
        self.path = candidate
        self.max_source_age_seconds = max_source_age_seconds
        self.broker_environment = broker_environment
        self.runtime_binding = runtime_binding
        self._signing_key = signing_key

    def __repr__(self) -> str:
        return (
            "PositionsArtifactPublisher("
            f"path={self.path!r}, "
            f"max_source_age_seconds={self.max_source_age_seconds}, "
            f"broker_environment={self.broker_environment!r})"
        )

    @classmethod
    def from_runtime_config(
        cls,
        config: RuntimeConfig,
        *,
        signing_key: PositionsArtifactSigningKey,
    ) -> "PositionsArtifactPublisher":
        if type(config) is not RuntimeConfig:
            raise TypeError("config must be an exact RuntimeConfig")
        if not config.data.require_complete_snapshots:
            raise PositionsArtifactPublishError(
                "complete_portfolio_snapshots_required"
            )
        if (
            config.data.max_snapshot_age_seconds
            < MIN_PRODUCER_FRESHNESS_WINDOW_SECONDS
        ):
            raise PositionsArtifactPublishError(
                "positions_freshness_window_too_short"
            )
        if (
            config.broker_environment not in {"sandbox", "production"}
            or config.selected_account is None
        ):
            raise PositionsArtifactPublishError(
                "broker_backed_positions_runtime_required"
            )
        return cls(
            config.paths.positions_artifact_file,
            max_source_age_seconds=(
                config.data.max_snapshot_age_seconds
            ),
            signing_key=signing_key,
            broker_environment=config.broker_environment,
            runtime_binding=positions_artifact_runtime_binding(config),
        )

    def publish(
        self,
        snapshot: PositionsSnapshot,
        *,
        now: datetime | None = None,
    ) -> PublishReceipt:
        """Publish one fresh monotonic snapshot or fail closed."""

        if type(snapshot) is not PositionsSnapshot:
            raise TypeError("snapshot must be an exact PositionsSnapshot")
        if snapshot.broker_environment != self.broker_environment:
            raise PositionsArtifactPublishError(
                "positions_snapshot_environment_mismatch"
            )
        current_time = _utc_now(now)
        source_age = (
            current_time - snapshot.source_as_of
        ).total_seconds()
        if source_age < -MAX_ARTIFACT_FUTURE_SKEW_SECONDS:
            raise PositionsArtifactPublishError(
                "positions_snapshot_from_future"
            )
        if source_age > self.max_source_age_seconds:
            raise PositionsArtifactPublishError("positions_snapshot_stale")
        try:
            payload = render_positions_html(
                snapshot,
                signing_key=self._signing_key,
                runtime_binding=self.runtime_binding,
            )
            metadata = inspect_positions_html(
                payload,
                signing_key=self._signing_key,
                expected_broker_environment=self.broker_environment,
                expected_runtime_binding=self.runtime_binding,
            )
        except PositionsArtifactError as exc:
            raise PositionsArtifactPublishError(exc.code) from exc
        if (
            metadata.source_as_of != snapshot.source_as_of
            or metadata.source_generation != snapshot.source_generation
        ):
            raise PositionsArtifactPublishError(
                "rendered_positions_metadata_mismatch"
            )
        digest = hashlib.sha256(payload).hexdigest()
        parent_descriptor = _open_parent(self.path.parent)
        lock_descriptor = -1
        try:
            lock_descriptor = _acquire_lock(
                parent_descriptor,
                f".{self.path.name}.publish.lock",
            )
            lock_name = f".{self.path.name}.publish.lock"
            existing = _read_existing(
                parent_descriptor,
                self.path.name,
                signing_key=self._signing_key,
                broker_environment=self.broker_environment,
                runtime_binding=self.runtime_binding,
            )
            existing_state = None
            if existing is not None:
                (
                    existing_payload,
                    existing_metadata,
                    existing_state,
                ) = existing
                if existing_metadata is not None:
                    if (
                        existing_metadata.source_as_of
                        > snapshot.source_as_of
                    ):
                        raise PositionsArtifactPublishError(
                            "positions_snapshot_regression"
                        )
                    if (
                        existing_metadata.source_as_of
                        == snapshot.source_as_of
                        and existing_metadata.source_generation
                        != snapshot.source_generation
                    ):
                        raise PositionsArtifactPublishError(
                            "positions_snapshot_generation_conflict"
                        )
                    if (
                        existing_metadata.source_generation
                        == snapshot.source_generation
                        and existing_payload == payload
                    ):
                        _revalidate_target(
                            parent_descriptor,
                            self.path.name,
                            existing_state,
                        )
                        _revalidate_parent(
                            self.path.parent,
                            parent_descriptor,
                        )
                        _revalidate_lock(
                            parent_descriptor,
                            lock_name,
                            lock_descriptor,
                        )
                        return PublishReceipt(
                            state="already_current",
                            sha256=digest,
                            size_bytes=len(payload),
                            source_as_of=(
                                snapshot.source_as_of.isoformat()
                            ),
                            source_generation=(
                                snapshot.source_generation
                            ),
                        )
            return _replace_artifact(
                path=self.path,
                parent_descriptor=parent_descriptor,
                payload=payload,
                digest=digest,
                snapshot=snapshot,
                expected_target=existing_state,
                lock_descriptor=lock_descriptor,
                lock_name=lock_name,
            )
        finally:
            if lock_descriptor >= 0:
                try:
                    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(lock_descriptor)
            os.close(parent_descriptor)


def _utc_now(value: datetime | None) -> datetime:
    result = datetime.now(timezone.utc) if value is None else value
    if (
        type(result) is not datetime
        or result.tzinfo is None
        or result.utcoffset() is None
    ):
        raise TypeError("now must be an exact timezone-aware datetime")
    return result.astimezone(timezone.utc)


def _safe_flags(base: int) -> int:
    return base | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _same_snapshot(
    left: os.stat_result,
    right: os.stat_result,
) -> bool:
    return (
        _same_file(left, right)
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _private_directory(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and stat.S_IMODE(metadata.st_mode) == 0o700
    )


def _private_regular(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and stat.S_IMODE(metadata.st_mode) == 0o600
        and metadata.st_nlink == 1
    )


def _open_parent(path: Path) -> int:
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not _private_directory(before):
            raise PositionsArtifactPublishError(
                "positions_artifact_parent_unsafe"
            )
        descriptor = os.open(
            path,
            _safe_flags(
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            ),
        )
    except PositionsArtifactPublishError:
        raise
    except OSError as exc:
        raise PositionsArtifactPublishError(
            "positions_artifact_parent_unavailable"
        ) from exc
    try:
        after = os.fstat(descriptor)
        if not _same_file(before, after) or not _private_directory(after):
            raise PositionsArtifactPublishError(
                "positions_artifact_parent_unsafe"
            )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _revalidate_parent(path: Path, descriptor: int) -> None:
    try:
        by_path = os.lstat(path)
        by_descriptor = os.fstat(descriptor)
    except OSError as exc:
        raise PositionsArtifactPublishError(
            "positions_artifact_parent_changed"
        ) from exc
    if (
        not _same_file(by_path, by_descriptor)
        or not _private_directory(by_path)
        or not _private_directory(by_descriptor)
    ):
        raise PositionsArtifactPublishError(
            "positions_artifact_parent_changed"
        )


def _revalidate_target(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result | None,
) -> None:
    try:
        observed = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        if expected is None:
            return
        raise PositionsArtifactPublishError(
            "positions_artifact_target_changed"
        )
    except OSError as exc:
        raise PositionsArtifactPublishError(
            "positions_artifact_target_changed"
        ) from exc
    if (
        expected is None
        or not _private_regular(observed)
        or not _same_snapshot(expected, observed)
    ):
        raise PositionsArtifactPublishError(
            "positions_artifact_target_changed"
        )


def _acquire_lock(parent_descriptor: int, name: str) -> int:
    descriptor = -1
    created = False
    try:
        try:
            descriptor = os.open(
                name,
                _safe_flags(os.O_RDWR | os.O_CREAT | os.O_EXCL),
                0o600,
                dir_fd=parent_descriptor,
            )
            created = True
        except FileExistsError:
            descriptor = os.open(
                name,
                _safe_flags(os.O_RDWR),
                dir_fd=parent_descriptor,
            )
        if created:
            os.fchmod(descriptor, 0o600)
        metadata = os.fstat(descriptor)
        if not _private_regular(metadata):
            raise PositionsArtifactPublishError(
                "positions_publish_lock_unsafe"
            )
        try:
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise PositionsArtifactPublishError(
                "positions_publisher_busy"
            ) from exc
        _revalidate_lock(parent_descriptor, name, descriptor)
        return descriptor
    except PositionsArtifactPublishError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise PositionsArtifactPublishError(
            "positions_publish_lock_unavailable"
        ) from exc


def _revalidate_lock(
    parent_descriptor: int,
    name: str,
    descriptor: int,
) -> None:
    try:
        by_descriptor = os.fstat(descriptor)
        by_path = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise PositionsArtifactPublishError(
            "positions_publish_lock_changed"
        ) from exc
    if (
        not _private_regular(by_descriptor)
        or not _private_regular(by_path)
        or not _same_snapshot(by_descriptor, by_path)
    ):
        raise PositionsArtifactPublishError(
            "positions_publish_lock_changed"
        )


def _read_existing(
    parent_descriptor: int,
    name: str,
    *,
    signing_key: PositionsArtifactSigningKey,
    broker_environment: str,
    runtime_binding: str,
) -> tuple[
    bytes,
    ArtifactMetadata | None,
    os.stat_result,
] | None:
    try:
        before = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PositionsArtifactPublishError(
            "existing_positions_artifact_unsafe"
        ) from exc
    if (
        not _private_regular(before)
        or before.st_size > MAX_POSITIONS_ARTIFACT_BYTES
    ):
        raise PositionsArtifactPublishError(
            "existing_positions_artifact_unsafe"
        )
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            _safe_flags(
                os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
            ),
            dir_fd=parent_descriptor,
        )
        after = os.fstat(descriptor)
        if not _same_file(before, after) or not _private_regular(after):
            raise PositionsArtifactPublishError(
                "existing_positions_artifact_changed"
            )
        chunks: list[bytes] = []
        remaining = MAX_POSITIONS_ARTIFACT_BYTES + 1
        while remaining:
            try:
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, remaining),
                )
            except InterruptedError:
                continue
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        final = os.fstat(descriptor)
        try:
            path_after = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise PositionsArtifactPublishError(
                "existing_positions_artifact_changed"
            ) from exc
        if (
            not _private_regular(path_after)
            or not _same_snapshot(before, after)
            or not _same_snapshot(after, final)
            or not _same_snapshot(final, path_after)
            or len(payload) != final.st_size
            or len(payload) > MAX_POSITIONS_ARTIFACT_BYTES
        ):
            raise PositionsArtifactPublishError(
                "existing_positions_artifact_changed"
            )
    except PositionsArtifactPublishError:
        raise
    except OSError as exc:
        raise PositionsArtifactPublishError(
            "existing_positions_artifact_unsafe"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        metadata: ArtifactMetadata | None = inspect_positions_html(
            payload,
            signing_key=signing_key,
            expected_broker_environment=broker_environment,
            expected_runtime_binding=runtime_binding,
        )
    except PositionsArtifactError:
        metadata = None
    return payload, metadata, final


def _replace_artifact(
    *,
    path: Path,
    parent_descriptor: int,
    payload: bytes,
    digest: str,
    snapshot: PositionsSnapshot,
    expected_target: os.stat_result | None,
    lock_descriptor: int,
    lock_name: str,
) -> PublishReceipt:
    temporary_name: str | None = None
    temporary_descriptor = -1
    replaced = False
    try:
        for _ in range(16):
            candidate = f".{path.name}.{secrets.token_hex(16)}.tmp"
            try:
                temporary_descriptor = os.open(
                    candidate,
                    _safe_flags(os.O_RDWR | os.O_CREAT | os.O_EXCL),
                    0o600,
                    dir_fd=parent_descriptor,
                )
                temporary_name = candidate
                break
            except FileExistsError:
                continue
        if temporary_descriptor < 0 or temporary_name is None:
            raise PositionsArtifactPublishError(
                "positions_artifact_temp_unavailable"
            )
        os.fchmod(temporary_descriptor, 0o600)
        initial = os.fstat(temporary_descriptor)
        if not _private_regular(initial) or initial.st_size != 0:
            raise PositionsArtifactPublishError(
                "positions_artifact_temp_unsafe"
            )
        offset = 0
        while offset < len(payload):
            try:
                written = os.write(
                    temporary_descriptor,
                    payload[offset:],
                )
            except InterruptedError:
                continue
            if written <= 0:
                raise PositionsArtifactPublishError(
                    "positions_artifact_write_failed"
                )
            offset += written
        os.fsync(temporary_descriptor)
        written_metadata = os.fstat(temporary_descriptor)
        if (
            not _private_regular(written_metadata)
            or written_metadata.st_size != len(payload)
        ):
            raise PositionsArtifactPublishError(
                "positions_artifact_temp_unsafe"
            )
        _revalidate_parent(path.parent, parent_descriptor)
        _revalidate_target(
            parent_descriptor,
            path.name,
            expected_target,
        )
        _revalidate_lock(
            parent_descriptor,
            lock_name,
            lock_descriptor,
        )
        try:
            os.replace(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            replaced = True
            temporary_name = None
        except OSError as exc:
            raise PositionsArtifactCommitUnknown(digest) from exc
        try:
            published = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            descriptor_published = os.fstat(temporary_descriptor)
            if (
                not _same_file(descriptor_published, published)
                or not _private_regular(descriptor_published)
                or not _private_regular(published)
                or published.st_size != len(payload)
                or descriptor_published.st_size != len(payload)
            ):
                raise PositionsArtifactCommitUnknown(digest)
            os.lseek(temporary_descriptor, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = len(payload) + 1
            while remaining:
                try:
                    chunk = os.read(
                        temporary_descriptor,
                        min(64 * 1024, remaining),
                    )
                except InterruptedError:
                    continue
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            observed_payload = b"".join(chunks)
            after_read = os.fstat(temporary_descriptor)
            path_after = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                observed_payload != payload
                or hashlib.sha256(observed_payload).hexdigest() != digest
                or not _same_snapshot(
                    descriptor_published,
                    after_read,
                )
                or not _same_snapshot(after_read, path_after)
            ):
                raise PositionsArtifactCommitUnknown(digest)
            os.fsync(parent_descriptor)
            _revalidate_parent(path.parent, parent_descriptor)
            _revalidate_lock(
                parent_descriptor,
                lock_name,
                lock_descriptor,
            )
            final_descriptor_state = os.fstat(temporary_descriptor)
            final_path_state = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not _same_snapshot(after_read, final_descriptor_state)
                or not _same_snapshot(
                    final_descriptor_state,
                    final_path_state,
                )
            ):
                raise PositionsArtifactCommitUnknown(digest)
        except PositionsArtifactCommitUnknown:
            raise
        except (OSError, PositionsArtifactPublishError) as exc:
            raise PositionsArtifactCommitUnknown(digest) from exc
        return PublishReceipt(
            state="committed_durable",
            sha256=digest,
            size_bytes=len(payload),
            source_as_of=snapshot.source_as_of.isoformat(),
            source_generation=snapshot.source_generation,
        )
    except PositionsArtifactCommitUnknown:
        raise
    except PositionsArtifactPublishError:
        raise
    except OSError as exc:
        if replaced:
            raise PositionsArtifactCommitUnknown(digest) from exc
        raise PositionsArtifactPublishError(
            "positions_artifact_publish_failed"
        ) from exc
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_name is not None:
            try:
                os.unlink(
                    temporary_name,
                    dir_fd=parent_descriptor,
                )
            except OSError:
                pass


__all__ = [
    "MIN_PRODUCER_FRESHNESS_WINDOW_SECONDS",
    "PositionsArtifactCommitUnknown",
    "PositionsArtifactPublishError",
    "PositionsArtifactPublisher",
    "PublishReceipt",
]
