"""Browser-owned POSIX artifact I/O. Never import Engine or model code.

FDs 3/4/5 are trusted staging/temporary/published directories opened by Browser
before Engine startup. All untrusted names are walked relative to these handles.
Run with the bundled interpreter, -I -S -B, and a fixed executable-search PATH only.
"""

import contextlib
import errno
import json
import os
import re
import stat
import sys
import threading
import time

CHUNK_BYTES = 1024 * 1024
MAX_BYTES = 20 * 1024**3
MAX_REQUEST_BYTES = 256 * 1024
MAX_ENTRIES = 10000
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK


class ArtifactFailure(Exception):
    pass


def components(value):
    if not isinstance(value, str) or len(value) > 4096 or "\0" in value:
        raise ArtifactFailure("ARTIFACT_PATH_INVALID")
    parts = value.split("/")
    if len(parts) > 32 or any(part in ("", ".", "..") for part in parts):
        raise ArtifactFailure("ARTIFACT_PATH_INVALID")
    return parts


def open_directory(root, parts):
    descriptor = os.dup(root)
    try:
        for part in parts:
            child = os.open(part, DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def file_identity(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def copy_output(output):
    parts = components(output["source"])
    if len(parts) < 3:
        raise ArtifactFailure("ARTIFACT_PATH_INVALID")
    name = output["object"]
    size = output["size"]
    if not isinstance(name, str) or not re.fullmatch(r"object_[A-Za-z0-9_-]{32}", name):
        raise ArtifactFailure("ARTIFACT_INVALID")
    if type(size) is not int or not 0 <= size <= MAX_BYTES:
        raise ArtifactFailure("ARTIFACT_INVALID")
    parent = open_directory(3, parts[:-1])
    source = destination = bucket = None
    partial = name + ".adopting"
    created = linked = False
    try:
        source = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        before = os.fstat(source)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ArtifactFailure("ARTIFACT_PATH_INVALID")
        if before.st_size != size:
            raise ArtifactFailure("ARTIFACT_INVALID")
        destination = os.open(
            partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=4
        )
        created = True
        remaining = size
        while remaining:
            chunk = os.read(source, min(CHUNK_BYTES, remaining))
            if not chunk:
                raise ArtifactFailure("ARTIFACT_INVALID")
            remaining -= len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination, view)
                if written <= 0:
                    raise ArtifactFailure("ARTIFACT_IO_ERROR")
                view = view[written:]
        # Detect ordinary concurrent changes. This is not a claim of an atomic
        # content snapshot: Engine already controls the bytes of its output.
        if os.read(source, 1) or file_identity(before) != file_identity(os.fstat(source)):
            raise ArtifactFailure("ARTIFACT_INVALID")
        copied = os.fstat(destination)
        if not stat.S_ISREG(copied.st_mode) or copied.st_nlink != 1 or copied.st_size != size:
            raise ArtifactFailure("ARTIFACT_INVALID")
        os.fchmod(destination, 0o400)
        os.fsync(destination)
        with contextlib.suppress(FileExistsError):
            os.mkdir(name[-2:], 0o700, dir_fd=5)
        bucket = open_directory(5, [name[-2:]])
        # Exclusive atomic installation inside Browser-only storage. Never move
        # the Engine inode, never overwrite an existing object, never use EXDEV
        # as a reason to reopen an Engine pathname.
        os.link(partial, name, src_dir_fd=4, dst_dir_fd=bucket, follow_symlinks=False)
        linked = True
        os.unlink(partial, dir_fd=4)
        created = False
        sync_directory(bucket)
        sync_directory(4)
    except BaseException:
        if linked:
            os.unlink(name, dir_fd=bucket)
        raise
    finally:
        for descriptor in (source, destination, bucket, parent):
            if descriptor is not None:
                os.close(descriptor)
        if created:
            os.unlink(partial, dir_fd=4)


def sync_directory(descriptor):
    try:
        os.fsync(descriptor)
    except OSError as error:
        # Some supported filesystems cannot flush directory descriptors. The
        # copied file itself is already fsynced; startup checks registry/file
        # consistency after a crash. Do not fall back to an untrusted pathname.
        if error.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise


def remove_entry(parent, name, budget, depth=0):
    budget[0] -= 1
    if budget[0] < 0 or depth > 32:
        raise ArtifactFailure("RESOURCE_LIMIT_EXCEEDED")
    try:
        child = os.open(name, DIRECTORY_FLAGS, dir_fd=parent)
    except FileNotFoundError:
        return
    except OSError as error:
        if error.errno not in (errno.ENOTDIR, errno.ELOOP):
            raise
        # unlinkat does not follow the leaf, even if it changes before unlink.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name, dir_fd=parent)
        return
    try:
        with os.scandir(child) as entries:
            for entry in entries:
                remove_entry(child, entry.name, budget, depth + 1)
    finally:
        os.close(child)
    with contextlib.suppress(FileNotFoundError):
        os.rmdir(name, dir_fd=parent)


def cleanup(request):
    budget = [MAX_ENTRIES]
    if request.get("all") is True:
        root = open_directory(3, ["."])
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    remove_entry(root, entry.name, budget)
        finally:
            os.close(root)
        return
    parts = components(request["job"])
    if len(parts) != 2:
        raise ArtifactFailure("ARTIFACT_PATH_INVALID")
    try:
        parent = open_directory(3, parts[:1])
    except FileNotFoundError:
        return
    try:
        files = request.get("files")
        if files is None:
            remove_entry(parent, parts[1], budget)
        else:
            if not isinstance(files, list) or len(files) > 64:
                raise ArtifactFailure("ARTIFACT_INVALID")
            for value in files:
                path = components(value)
                if path[:2] != parts or len(path) < 3:
                    raise ArtifactFailure("ARTIFACT_PATH_INVALID")
                try:
                    directory = open_directory(parent, path[1:-1])
                except FileNotFoundError:
                    continue
                try:
                    os.unlink(path[-1], dir_fd=directory)
                except FileNotFoundError:
                    pass
                finally:
                    os.close(directory)
    finally:
        os.close(parent)
    try:
        os.rmdir(parts[0], dir_fd=3)
    except OSError as error:
        if error.errno not in (errno.ENOENT, errno.ENOTEMPTY, errno.EEXIST):
            raise


def main():
    for descriptor in (3, 4, 5):
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ArtifactFailure("ARTIFACT_PATH_INVALID")
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ArtifactFailure("ARTIFACT_INVALID")
    request = json.loads(raw)
    parent_pid = request["parentPid"]
    if type(parent_pid) is not int or os.getppid() != parent_pid:
        raise ArtifactFailure("ARTIFACT_IO_ERROR")

    def watch_parent():
        while os.getppid() == parent_pid:
            time.sleep(0.25)
        # Browser crash: close inherited handles and stop touching private files.
        # Partial objects have no registry capability; startup reconciles them.
        os._exit(1)

    threading.Thread(target=watch_parent, daemon=True).start()
    if request["operation"] == "copy":
        outputs = request["outputs"]
        if not isinstance(outputs, list) or not 1 <= len(outputs) <= 64:
            raise ArtifactFailure("ARTIFACT_INVALID")
        if sum(item["size"] for item in outputs) > MAX_BYTES:
            raise ArtifactFailure("RESOURCE_LIMIT_EXCEEDED")
        for output in outputs:
            copy_output(output)
    elif request["operation"] == "cleanup":
        cleanup(request)
    else:
        raise ArtifactFailure("ARTIFACT_INVALID")


if __name__ == "__main__":
    try:
        main()
    except ArtifactFailure as error:
        print(str(error))
        sys.exit(1)
    except OSError as error:
        code = "ARTIFACT_IO_ERROR"
        if error.errno in (errno.ELOOP, errno.ENOTDIR, errno.EISDIR):
            code = "ARTIFACT_PATH_INVALID"
        elif error.errno in (errno.ENOSPC, errno.EDQUOT):
            code = "DISK_SPACE_INSUFFICIENT"
        print(code)
        sys.exit(1)
    except (KeyError, TypeError, ValueError):
        print("ARTIFACT_INVALID")
        sys.exit(1)
