"""Owner-only append store for content-addressed Regime V2 observations.

Each prospective observation is an immutable file named by its sequence and
content hash.  There is no mutable head file and no update/delete API.  Readers
verify the complete hash chain before returning any entry.
"""

from __future__ import annotations

import errno
import fcntl
import os
import re
import secrets
import stat
from datetime import datetime, timezone
from pathlib import Path

from live_trading.regime_prospective import (
    ProspectiveJournalEntry,
    ProspectiveReviewError,
    ProspectiveReviewProtocol,
    ProviderEntitlementReceipt,
    validate_journal_chain,
)
from live_trading.regime_signal import RegimeSignal


MAX_JOURNAL_ENTRY_BYTES = 512 * 1024
_LOCK_NAME = ".append.lock"
_ENTRY_NAME = re.compile(
    r"^(?P<sequence>[0-9]{12})-(?P<sha256>[0-9a-f]{64})\.json$"
)
_PENDING_NAME = re.compile(r"^\.pending-[0-9a-f]{32}\.tmp$")


class ProspectiveJournalStoreError(RuntimeError):
    """Fail-closed local journal persistence error."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _trusted_utc_now() -> datetime:
    """Production-owned append clock; patched only in isolated unit tests."""

    return datetime.now(timezone.utc)


def _safe_open_flags(base: int) -> int:
    return base | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_file(left, right)
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _validate_directory(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise ProspectiveJournalStoreError("journal_directory_unsafe")


def _validate_regular_owner_file(
    metadata: os.stat_result,
    *,
    maximum_size: int | None = None,
    allowed_link_counts: tuple[int, ...] = (1,),
) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or metadata.st_nlink not in allowed_link_counts
    ):
        raise ProspectiveJournalStoreError("journal_file_unsafe")
    if maximum_size is not None and metadata.st_size > maximum_size:
        raise ProspectiveJournalStoreError("journal_entry_too_large")


class ProspectiveJournalStore:
    """Append and verify one dedicated prospective-journal directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not self.directory.name or self.directory.name in {".", ".."}:
            raise ValueError("directory must identify a dedicated journal")

    def _open_directory(self) -> int:
        try:
            before = os.lstat(self.directory)
        except FileNotFoundError as exc:
            raise ProspectiveJournalStoreError(
                "journal_directory_missing"
            ) from exc
        except OSError as exc:
            raise ProspectiveJournalStoreError(
                "journal_directory_unsafe"
            ) from exc
        if stat.S_ISLNK(before.st_mode):
            raise ProspectiveJournalStoreError("journal_directory_unsafe")
        try:
            descriptor = os.open(
                self.directory,
                _safe_open_flags(
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                ),
            )
        except OSError as exc:
            raise ProspectiveJournalStoreError(
                "journal_directory_unsafe"
            ) from exc
        try:
            after = os.fstat(descriptor)
            if not _same_file(before, after):
                raise ProspectiveJournalStoreError(
                    "journal_directory_unsafe"
                )
            _validate_directory(after)
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    @staticmethod
    def _open_lock(directory_descriptor: int, *, exclusive: bool) -> int:
        descriptor = -1
        try:
            descriptor = os.open(
                _LOCK_NAME,
                _safe_open_flags(os.O_RDWR | os.O_CREAT),
                0o600,
                dir_fd=directory_descriptor,
            )
            os.fchmod(descriptor, stat.S_IRUSR | stat.S_IWUSR)
            _validate_regular_owner_file(os.fstat(descriptor))
            fcntl.flock(
                descriptor,
                fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
            )
            opened = os.fstat(descriptor)
            path_after = os.stat(
                _LOCK_NAME,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if not _same_file(opened, path_after):
                raise ProspectiveJournalStoreError(
                    "journal_lock_unsafe"
                )
            _validate_regular_owner_file(path_after)
            return descriptor
        except OSError as exc:
            if descriptor >= 0:
                os.close(descriptor)
            raise ProspectiveJournalStoreError(
                "journal_lock_unsafe"
            ) from exc
        except ProspectiveJournalStoreError:
            if descriptor >= 0:
                os.close(descriptor)
            raise

    @staticmethod
    def _recover_pending(directory_descriptor: int) -> None:
        """Remove only store-owned pending links left by an interrupted append."""

        try:
            names = os.listdir(directory_descriptor)
        except OSError as exc:
            raise ProspectiveJournalStoreError(
                "journal_directory_unreadable"
            ) from exc
        pending_names = [
            name for name in names if _PENDING_NAME.fullmatch(name)
        ]
        changed = False
        for pending_name in pending_names:
            try:
                pending_metadata = os.stat(
                    pending_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ProspectiveJournalStoreError(
                    "journal_pending_unsafe"
                ) from exc
            _validate_regular_owner_file(
                pending_metadata,
                maximum_size=MAX_JOURNAL_ENTRY_BYTES,
                allowed_link_counts=(1, 2),
            )
            if pending_metadata.st_nlink == 2:
                linked_entries = []
                for candidate in names:
                    if _ENTRY_NAME.fullmatch(candidate) is None:
                        continue
                    try:
                        candidate_metadata = os.stat(
                            candidate,
                            dir_fd=directory_descriptor,
                            follow_symlinks=False,
                        )
                    except OSError as exc:
                        raise ProspectiveJournalStoreError(
                            "journal_pending_unsafe"
                        ) from exc
                    if _same_file(pending_metadata, candidate_metadata):
                        linked_entries.append(candidate)
                if len(linked_entries) != 1:
                    raise ProspectiveJournalStoreError(
                        "journal_pending_unsafe"
                    )
            try:
                os.unlink(
                    pending_name,
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise ProspectiveJournalStoreError(
                    "journal_pending_recovery_failed"
                ) from exc
            changed = True
        if changed:
            try:
                os.fsync(directory_descriptor)
            except OSError as exc:
                raise ProspectiveJournalStoreError(
                    "journal_pending_recovery_failed"
                ) from exc

    @staticmethod
    def _entry_names(directory_descriptor: int) -> tuple[str, ...]:
        try:
            names = os.listdir(directory_descriptor)
        except OSError as exc:
            raise ProspectiveJournalStoreError(
                "journal_directory_unreadable"
            ) from exc
        entries: list[tuple[int, str]] = []
        for name in names:
            if name == _LOCK_NAME:
                continue
            match = _ENTRY_NAME.fullmatch(name)
            if match is None:
                raise ProspectiveJournalStoreError(
                    "journal_directory_contains_unknown_file"
                )
            entries.append((int(match.group("sequence")), name))
        entries.sort()
        if [item[0] for item in entries] != list(
            range(1, len(entries) + 1)
        ):
            raise ProspectiveJournalStoreError(
                "journal_file_sequence_invalid"
            )
        return tuple(item[1] for item in entries)

    @staticmethod
    def _read_entry(
        directory_descriptor: int,
        filename: str,
    ) -> ProspectiveJournalEntry:
        match = _ENTRY_NAME.fullmatch(filename)
        if match is None:
            raise ProspectiveJournalStoreError(
                "journal_filename_invalid"
            )
        try:
            before = os.stat(
                filename,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ProspectiveJournalStoreError(
                "journal_entry_unreadable"
            ) from exc
        _validate_regular_owner_file(
            before,
            maximum_size=MAX_JOURNAL_ENTRY_BYTES,
        )
        try:
            descriptor = os.open(
                filename,
                _safe_open_flags(
                    os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                ),
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                raise ProspectiveJournalStoreError(
                    "journal_entry_unreadable"
                ) from exc
            raise ProspectiveJournalStoreError(
                "journal_file_unsafe"
            ) from exc
        try:
            after = os.fstat(descriptor)
            _validate_regular_owner_file(
                after,
                maximum_size=MAX_JOURNAL_ENTRY_BYTES,
            )
            if not _same_snapshot(before, after):
                raise ProspectiveJournalStoreError(
                    "journal_file_unsafe"
                )
            remaining = MAX_JOURNAL_ENTRY_BYTES + 1
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            final = os.fstat(descriptor)
            try:
                path_after = os.stat(
                    filename,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as exc:
                raise ProspectiveJournalStoreError(
                    "journal_file_unsafe"
                ) from exc
            if (
                not _same_snapshot(after, final)
                or not _same_snapshot(final, path_after)
                or len(raw) != final.st_size
            ):
                raise ProspectiveJournalStoreError(
                    "journal_file_unsafe"
                )
            if len(raw) > MAX_JOURNAL_ENTRY_BYTES:
                raise ProspectiveJournalStoreError(
                    "journal_entry_too_large"
                )
            try:
                payload = raw.decode("utf-8", errors="strict")
                entry = ProspectiveJournalEntry.from_json(payload)
            except (UnicodeDecodeError, ProspectiveReviewError) as exc:
                raise ProspectiveJournalStoreError(
                    "journal_entry_invalid"
                ) from exc
            if (
                entry.sequence_number != int(match.group("sequence"))
                or entry.entry_sha256 != match.group("sha256")
            ):
                raise ProspectiveJournalStoreError(
                    "journal_filename_hash_mismatch"
                )
            return entry
        finally:
            os.close(descriptor)

    def _load_locked(
        self,
        *,
        directory_descriptor: int,
        protocol: ProspectiveReviewProtocol,
    ) -> tuple[ProspectiveJournalEntry, ...]:
        entries = tuple(
            self._read_entry(directory_descriptor, filename)
            for filename in self._entry_names(directory_descriptor)
        )
        try:
            return validate_journal_chain(protocol, entries)
        except (ProspectiveReviewError, TypeError) as exc:
            raise ProspectiveJournalStoreError(
                "journal_chain_invalid"
            ) from exc

    def load_entries(
        self,
        *,
        protocol: ProspectiveReviewProtocol,
    ) -> tuple[ProspectiveJournalEntry, ...]:
        """Return the fully verified chain; never a partial prefix."""

        if type(protocol) is not ProspectiveReviewProtocol:
            raise TypeError(
                "protocol must be an exact ProspectiveReviewProtocol"
            )
        directory_descriptor = self._open_directory()
        lock_descriptor = -1
        try:
            lock_descriptor = self._open_lock(
                directory_descriptor,
                exclusive=True,
            )
            self._recover_pending(directory_descriptor)
            return self._load_locked(
                directory_descriptor=directory_descriptor,
                protocol=protocol,
            )
        finally:
            if lock_descriptor >= 0:
                os.close(lock_descriptor)
            os.close(directory_descriptor)

    def append(
        self,
        *,
        protocol: ProspectiveReviewProtocol,
        signal: RegimeSignal,
        entitlement_receipt: ProviderEntitlementReceipt,
    ) -> ProspectiveJournalEntry:
        """Atomically link one fsynced entry using the store-owned clock."""

        if type(protocol) is not ProspectiveReviewProtocol:
            raise TypeError(
                "protocol must be an exact ProspectiveReviewProtocol"
            )
        directory_descriptor = self._open_directory()
        lock_descriptor = -1
        pending_filename: str | None = None
        pending_descriptor = -1
        try:
            lock_descriptor = self._open_lock(
                directory_descriptor,
                exclusive=True,
            )
            self._recover_pending(directory_descriptor)
            chain = self._load_locked(
                directory_descriptor=directory_descriptor,
                protocol=protocol,
            )
            if chain and signal.as_of_session <= chain[-1].signal.as_of_session:
                raise ProspectiveJournalStoreError(
                    "journal_session_not_monotonic"
                )
            sequence = len(chain) + 1
            previous = chain[-1].entry_sha256 if chain else None
            recorded_at = _trusted_utc_now()
            try:
                entry = ProspectiveJournalEntry.create(
                    protocol=protocol,
                    sequence_number=sequence,
                    previous_entry_sha256=previous,
                    recorded_at=recorded_at,
                    signal=signal,
                    entitlement_receipt=entitlement_receipt,
                )
            except (ProspectiveReviewError, TypeError) as exc:
                raise ProspectiveJournalStoreError(
                    "journal_entry_rejected"
                ) from exc
            payload = entry.to_json().encode("utf-8")
            if len(payload) > MAX_JOURNAL_ENTRY_BYTES:
                raise ProspectiveJournalStoreError(
                    "journal_entry_too_large"
                )
            final_filename = (
                f"{entry.sequence_number:012d}-{entry.entry_sha256}.json"
            )
            try:
                for _ in range(16):
                    candidate = (
                        f".pending-{secrets.token_hex(16)}.tmp"
                    )
                    try:
                        pending_descriptor = os.open(
                            candidate,
                            _safe_open_flags(
                                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                            ),
                            0o600,
                            dir_fd=directory_descriptor,
                        )
                        pending_filename = candidate
                        break
                    except FileExistsError:
                        continue
                if pending_descriptor < 0 or pending_filename is None:
                    raise ProspectiveJournalStoreError(
                        "journal_append_failed"
                    )
                os.fchmod(
                    pending_descriptor,
                    stat.S_IRUSR | stat.S_IWUSR,
                )
                with os.fdopen(pending_descriptor, "wb") as handle:
                    pending_descriptor = -1
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(
                    pending_filename,
                    final_filename,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                os.unlink(
                    pending_filename,
                    dir_fd=directory_descriptor,
                )
                pending_filename = None
                os.fsync(directory_descriptor)
            except ProspectiveJournalStoreError:
                raise
            except OSError as exc:
                raise ProspectiveJournalStoreError(
                    "journal_append_failed"
                ) from exc
            return entry
        except ProspectiveJournalStoreError:
            if pending_filename is not None:
                try:
                    os.unlink(
                        pending_filename,
                        dir_fd=directory_descriptor,
                    )
                    os.fsync(directory_descriptor)
                except OSError:
                    pass
            raise
        finally:
            if pending_descriptor >= 0:
                os.close(pending_descriptor)
            if lock_descriptor >= 0:
                os.close(lock_descriptor)
            os.close(directory_descriptor)


__all__ = [
    "MAX_JOURNAL_ENTRY_BYTES",
    "ProspectiveJournalStore",
    "ProspectiveJournalStoreError",
]
