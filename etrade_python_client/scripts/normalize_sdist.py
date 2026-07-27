#!/usr/bin/env python3
"""Rewrite one inspected-source sdist with deterministic archive metadata."""

from __future__ import annotations

import argparse
import gzip
import io
import os
import stat
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Sequence


class SdistNormalizationError(RuntimeError):
    """The source distribution cannot be normalized safely."""


def _safe_name(name: str) -> str:
    candidate = name[:-1] if name.endswith("/") else name
    path = PurePosixPath(candidate)
    if (
        not candidate
        or "\\" in candidate
        or any(ord(character) < 32 or ord(character) == 127 for character in candidate)
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != candidate
    ):
        raise SdistNormalizationError(f"unsafe sdist path: {name!r}")
    return path.as_posix()


def normalize_sdist(path: Path, epoch: int) -> None:
    if epoch < 0:
        raise SdistNormalizationError("normalization epoch must be non-negative")
    members: list[tuple[str, bool, int, bytes]] = []
    seen: set[str] = set()
    seen_casefolded: set[str] = set()
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            if archive.pax_headers:
                raise SdistNormalizationError(
                    "sdist global PAX headers are not allowed"
                )
            for member in archive.getmembers():
                name = _safe_name(member.name)
                if name in seen or name.casefold() in seen_casefolded:
                    raise SdistNormalizationError(
                        f"duplicate or case-colliding sdist path: {name}"
                    )
                seen.add(name)
                seen_casefolded.add(name.casefold())
                if not member.isdir() and not member.isfile():
                    raise SdistNormalizationError(
                        f"non-regular sdist member: {name}"
                    )
                mode = stat.S_IMODE(member.mode)
                expected_mode = 0o755 if member.isdir() else 0o644
                if mode != expected_mode:
                    raise SdistNormalizationError(
                        f"unexpected sdist mode: {name} {mode:o}"
                    )
                data = b""
                if member.isfile():
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise SdistNormalizationError(
                            f"could not read sdist member: {name}"
                        )
                    data = extracted.read()
                    if len(data) != member.size:
                        raise SdistNormalizationError(
                            f"sdist member size differs: {name}"
                        )
                members.append((name, member.isdir(), mode, data))
    except (OSError, tarfile.TarError) as error:
        raise SdistNormalizationError(f"could not read sdist: {error}") from error

    members.sort(key=lambda value: (len(PurePosixPath(value[0]).parts), value[0]))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_output,
                mtime=epoch,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as output:
                    for name, is_directory, mode, data in members:
                        info = tarfile.TarInfo(name)
                        info.type = (
                            tarfile.DIRTYPE if is_directory else tarfile.REGTYPE
                        )
                        info.mode = mode
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = epoch
                        info.size = 0 if is_directory else len(data)
                        output.addfile(
                            info,
                            None if is_directory else io.BytesIO(data),
                        )
            raw_output.flush()
            os.fsync(raw_output.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        normalize_sdist(args.sdist.resolve(strict=True), args.epoch)
    except (OSError, SdistNormalizationError) as error:
        print(f"sdist normalization failed: {error}", file=sys.stderr)
        return 1
    print(f"normalized sdist: {args.sdist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
