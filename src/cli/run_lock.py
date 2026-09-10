"""Process-held, nonblocking ownership of a canonical CLI run directory."""

from contextlib import contextmanager
import errno
import hashlib
import os
from pathlib import Path


class RunDirectoryBusyError(FileExistsError):
    """Another owner is executing or finalizing in the run directory."""


@contextmanager
def run_directory_lock(run_dir: str | Path):
    """Hold an OS lock outside the managed directory until the caller returns.

    The sidecar is intentionally persistent. Unlinking it would allow another
    process to lock a new inode while an existing owner still holds the old one.
    """
    target = Path(run_dir).expanduser().resolve()
    identity = os.path.normcase(str(target))
    digest = hashlib.sha256(os.fsencode(identity)).hexdigest()
    lock_path = target.parent / f".audio-platform-cli-{digest}.lock"
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "r+b", buffering=0) as stream:
        try:
            if os.name == "nt":
                import msvcrt

                # Windows permits byte-range locks extending beyond EOF.
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise RunDirectoryBusyError(f"Run directory is already in use: {target}") from exc
            raise
        try:
            yield target
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
