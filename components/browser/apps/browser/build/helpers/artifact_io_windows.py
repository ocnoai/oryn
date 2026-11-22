"""Browser-owned Windows artifact I/O; stdlib only, launched with -I -S -B.

Bootstrap opens local NTFS roots before Engine startup and duplicates their
non-inheritable handles into Browser. Operations duplicate those handles back.
All untrusted names are single-component NtCreateFile opens relative to handles;
reparse points are never traversed. No pathname-copy or cross-volume fallback.
This does not implement the separate Windows Engine process sandbox.
"""

import contextlib
import ctypes as c
import json
import ntpath
import os
import re
import sys
import threading

CHUNK_BYTES = 1024 * 1024
MAX_BYTES = 20 * 1024**3
MAX_REQUEST_BYTES = 256 * 1024
MAX_ENTRIES = 10000
DIRECTORY = 0x10
REPARSE = 0x400
DELETE = 0x10000
SYNCHRONIZE = 0x100000
READ_ATTRIBUTES = 0x80
READ_DATA = 1
WRITE_DATA = 2
TRAVERSE = 0x20
OPEN_REPARSE = 0x200000
SYNC_IO = 0x20
SHARE_ALL = 7
U16, U32, I64, HANDLE = c.c_uint16, c.c_uint32, c.c_int64, c.c_void_p


class ArtifactFailure(Exception):
    pass


class UnicodeString(c.Structure):
    _fields_ = [("length", U16), ("maximum", U16), ("buffer", c.c_void_p)]


class ObjectAttributes(c.Structure):
    _fields_ = [
        ("length", U32),
        ("root", HANDLE),
        ("name", c.POINTER(UnicodeString)),
        ("attributes", U32),
        ("security", c.c_void_p),
        ("qos", c.c_void_p),
    ]


class IoStatus(c.Structure):
    _fields_ = [("status", c.c_void_p), ("information", c.c_size_t)]


class FileTime(c.Structure):
    _fields_ = [("low", U32), ("high", U32)]


class FileInfo(c.Structure):
    _fields_ = [
        ("attributes", U32),
        ("creation", FileTime),
        ("access", FileTime),
        ("write", FileTime),
        ("volume", U32),
        ("size_high", U32),
        ("size_low", U32),
        ("links", U32),
        ("index_high", U32),
        ("index_low", U32),
    ]


class BasicInfo(c.Structure):
    _fields_ = [
        ("creation", I64),
        ("access", I64),
        ("write", I64),
        ("change", I64),
        ("attributes", U32),
    ]


class RenameInfo(c.Structure):
    _fields_ = [("flags", U32), ("root", HANDLE), ("length", U32), ("name", U16 * 1)]


class DirectoryInfo(c.Structure):
    _fields_ = [
        ("next", U32),
        ("index", U32),
        ("creation", I64),
        ("access", I64),
        ("write", I64),
        ("change", I64),
        ("size", I64),
        ("allocation", I64),
        ("attributes", U32),
        ("length", U32),
        ("ea_size", U32),
        ("short_length", c.c_ubyte),
        ("short_name", U16 * 12),
        ("id", I64),
        ("name", U16 * 1),
    ]


def components(value):
    if not isinstance(value, str) or not 0 < len(value) <= 4096:
        raise ArtifactFailure("ARTIFACT_PATH_INVALID")
    parts = value.replace("\\", "/").split("/")
    if len(parts) > 32:
        raise ArtifactFailure("ARTIFACT_PATH_INVALID")
    for part in parts:
        # Reject streams, drive/device names, Win32 aliases and ambiguous names.
        if (
            not part
            or part in (".", "..")
            or part.endswith((".", " "))
            or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
            or len(part.encode("utf-16-le")) > 510
            or re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?", part)
        ):
            raise ArtifactFailure("ARTIFACT_PATH_INVALID")
    return parts


def output_spec(output):
    parts = components(output["source"])
    name, size = output["object"], output["size"]
    if len(parts) < 3:
        raise ArtifactFailure("ARTIFACT_PATH_INVALID")
    if not isinstance(name, str) or not re.fullmatch(r"object_[A-Za-z0-9_-]{32}", name):
        raise ArtifactFailure("ARTIFACT_INVALID")
    if type(size) is not int or not 0 <= size <= MAX_BYTES:
        raise ArtifactFailure("ARTIFACT_INVALID")
    return parts, name, size


def rename_buffer(root, name):
    if len(components(name)) != 1:
        raise ArtifactFailure("ARTIFACT_PATH_INVALID")
    encoded = name.encode("utf-16-le")
    buffer = c.create_string_buffer(c.sizeof(RenameInfo) + len(encoded) + 2)
    info = RenameInfo.from_buffer(buffer)
    info.flags, info.root, info.length = 0, root, len(encoded)  # Never replace.
    c.memmove(c.addressof(buffer) + RenameInfo.name.offset, encoded, len(encoded))
    return buffer


class WindowsIO:
    def __init__(self):
        if sys.platform != "win32":
            raise ArtifactFailure("ARTIFACT_IO_ERROR")
        self.kernel = c.WinDLL("kernel32", use_last_error=True)
        self.ntdll = c.WinDLL("ntdll", use_last_error=True)
        # Fixed-width Windows ABI types; ctypes.wintypes long is host-dependent.
        signatures = {
            "CloseHandle": ([HANDLE], c.c_int32),
            "GetCurrentProcess": ([], HANDLE),
            "OpenProcess": ([U32, c.c_int32, U32], HANDLE),
            "WaitForSingleObject": ([HANDLE, U32], U32),
            "DuplicateHandle": (
                [HANDLE, HANDLE, HANDLE, c.POINTER(HANDLE), U32, c.c_int32, U32],
                c.c_int32,
            ),
            "CreateFileW": ([c.c_wchar_p, U32, U32, c.c_void_p, U32, U32, HANDLE], HANDLE),
            "GetFileType": ([HANDLE], U32),
            "GetFileInformationByHandle": ([HANDLE, c.POINTER(FileInfo)], c.c_int32),
            "GetFileInformationByHandleEx": ([HANDLE, c.c_int32, c.c_void_p, U32], c.c_int32),
            "SetFileInformationByHandle": ([HANDLE, c.c_int32, c.c_void_p, U32], c.c_int32),
            "GetVolumeInformationByHandleW": (
                [
                    HANDLE,
                    c.c_wchar_p,
                    U32,
                    c.POINTER(U32),
                    c.POINTER(U32),
                    c.POINTER(U32),
                    c.c_wchar_p,
                    U32,
                ],
                c.c_int32,
            ),
            "GetDriveTypeW": ([c.c_wchar_p], U32),
            "ReadFile": ([HANDLE, c.c_void_p, U32, c.POINTER(U32), c.c_void_p], c.c_int32),
            "WriteFile": ([HANDLE, c.c_void_p, U32, c.POINTER(U32), c.c_void_p], c.c_int32),
            "FlushFileBuffers": ([HANDLE], c.c_int32),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result
        self.ntdll.NtCreateFile.argtypes = [
            c.POINTER(HANDLE),
            U32,
            c.POINTER(ObjectAttributes),
            c.POINTER(IoStatus),
            c.c_void_p,
            U32,
            U32,
            U32,
            U32,
            c.c_void_p,
            U32,
        ]
        self.ntdll.NtCreateFile.restype = c.c_int32
        self.ntdll.RtlNtStatusToDosError.argtypes = [c.c_int32]
        self.ntdll.RtlNtStatusToDosError.restype = U32

    def check(self, result):
        if not result:
            raise c.WinError(c.get_last_error())
        return result

    def close(self, handle):
        self.check(self.kernel.CloseHandle(handle))

    def duplicate(self, source_process, handle, destination_process, close_source=False):
        result = HANDLE()
        self.check(
            self.kernel.DuplicateHandle(
                source_process,
                handle,
                destination_process,
                c.byref(result),
                0,
                False,
                3 if close_source else 2,
            )
        )
        return result.value

    def info(self, handle):
        if self.kernel.GetFileType(handle) != 1:  # FILE_TYPE_DISK, never pipe/device.
            raise ArtifactFailure("ARTIFACT_PATH_INVALID")
        info, basic = FileInfo(), BasicInfo()
        self.check(self.kernel.GetFileInformationByHandle(handle, c.byref(info)))
        self.check(
            self.kernel.GetFileInformationByHandleEx(handle, 0, c.byref(basic), c.sizeof(basic))
        )
        return info, basic

    def identity(self, handle, size):
        info, basic = self.info(handle)
        actual_size = (info.size_high << 32) | info.size_low
        if info.attributes & (DIRECTORY | REPARSE | 0x40) or info.links != 1:
            raise ArtifactFailure("ARTIFACT_PATH_INVALID")
        if actual_size != size:
            raise ArtifactFailure("ARTIFACT_INVALID")
        return (
            info.volume,
            info.index_high,
            info.index_low,
            info.attributes,
            info.links,
            actual_size,
            basic.write,
            basic.change,
        )

    def open(
        self, root, name, *, directory=None, create=False, writable=False, delete=False, share=3
    ):
        if len(components(name)) != 1:
            raise ArtifactFailure("ARTIFACT_PATH_INVALID")
        encoded = name.encode("utf-16-le")
        buffer = c.create_string_buffer(encoded + b"\0\0")
        text = UnicodeString(len(encoded), len(encoded) + 2, c.addressof(buffer))
        attributes = ObjectAttributes(
            c.sizeof(ObjectAttributes), root, c.pointer(text), 0x40, None, None
        )
        result, status = HANDLE(), IoStatus()
        access = READ_DATA | READ_ATTRIBUTES | SYNCHRONIZE
        if directory is not False:
            access |= TRAVERSE
        if writable:
            access |= WRITE_DATA
        if delete:
            access |= DELETE
        options = OPEN_REPARSE | SYNC_IO
        if directory is not None:
            options |= 1 if directory else 0x40
        code = self.ntdll.NtCreateFile(
            c.byref(result),
            access,
            c.byref(attributes),
            c.byref(status),
            None,
            0x80,
            share,
            2 if create else 1,
            options,
            None,
            0,
        )
        if code < 0:
            raise c.WinError(self.ntdll.RtlNtStatusToDosError(code))
        return result.value

    @contextlib.contextmanager
    def directory(self, root, parts):
        handles = []
        try:
            current = root
            for part in parts:
                current = self.open(current, part, directory=True)
                handles.append(current)
                info, _ = self.info(current)
                if not info.attributes & DIRECTORY or info.attributes & REPARSE:
                    raise ArtifactFailure("ARTIFACT_PATH_INVALID")
            yield current
        finally:
            for handle in reversed(handles):
                self.close(handle)

    def root(self, path):
        if not isinstance(path, str):
            raise ArtifactFailure("ARTIFACT_PATH_INVALID")
        # No UNC/network/device roots. Canonical trusted paths are local drives.
        if path.startswith("\\\\?\\"):
            path = path[4:]
        drive, tail = ntpath.splitdrive(path)
        if not re.fullmatch(r"[A-Za-z]:", drive) or not tail.startswith("\\"):
            raise ArtifactFailure("ARTIFACT_PATH_INVALID")
        drive_root = drive + "\\"
        if self.kernel.GetDriveTypeW(drive_root) != 3:  # Local fixed disk only.
            raise ArtifactFailure("ARTIFACT_PATH_INVALID")
        handle = self.kernel.CreateFileW(
            drive_root,
            READ_DATA | TRAVERSE | READ_ATTRIBUTES | SYNCHRONIZE,
            3,
            None,
            3,
            0x02000000 | 0x00200000,
            None,
        )
        if handle == c.c_void_p(-1).value:
            raise c.WinError(c.get_last_error())
        try:
            filesystem = c.create_unicode_buffer(32)
            self.check(
                self.kernel.GetVolumeInformationByHandleW(
                    handle, None, 0, None, None, None, filesystem, len(filesystem)
                )
            )
            if filesystem.value != "NTFS":
                raise ArtifactFailure("ARTIFACT_PATH_INVALID")
            with self.directory(handle, components(tail[1:])) as leaf:
                return self.duplicate(
                    self.kernel.GetCurrentProcess(), leaf, self.kernel.GetCurrentProcess()
                )
        finally:
            self.close(handle)

    def read(self, handle, count):
        buffer, read = c.create_string_buffer(count), U32()
        self.check(self.kernel.ReadFile(handle, buffer, count, c.byref(read), None))
        return buffer.raw[: read.value]

    def write(self, handle, data):
        offset = 0
        while offset < len(data):
            written = U32()
            piece = data[offset:]
            self.check(self.kernel.WriteFile(handle, piece, len(piece), c.byref(written), None))
            if written.value == 0:
                raise ArtifactFailure("ARTIFACT_IO_ERROR")
            offset += written.value

    def flush(self, handle):
        self.check(self.kernel.FlushFileBuffers(handle))

    def rename(self, handle, root, name):
        buffer = rename_buffer(root, name)
        self.check(self.kernel.SetFileInformationByHandle(handle, 3, buffer, len(buffer)))

    def delete(self, handle):
        delete = c.c_ubyte(1)
        self.check(self.kernel.SetFileInformationByHandle(handle, 4, c.byref(delete), 1))

    def names(self, directory):
        buffer = c.create_string_buffer(64 * 1024)
        restart = True
        while True:
            if not self.kernel.GetFileInformationByHandleEx(
                directory, 11 if restart else 10, buffer, len(buffer)
            ):
                error = c.get_last_error()
                if error == 18:  # ERROR_NO_MORE_FILES
                    return
                raise c.WinError(error)
            restart = False
            offset = 0
            while True:
                if offset + c.sizeof(DirectoryInfo) > len(buffer):
                    raise ArtifactFailure("ARTIFACT_IO_ERROR")
                item = DirectoryInfo.from_buffer(buffer, offset)
                start = offset + DirectoryInfo.name.offset
                if item.length % 2 or start + item.length > len(buffer):
                    raise ArtifactFailure("ARTIFACT_IO_ERROR")
                name = buffer.raw[start : start + item.length].decode("utf-16-le")
                if name not in (".", ".."):
                    yield name
                if not item.next:
                    break
                if item.next < DirectoryInfo.name.offset + item.length or item.next % 8:
                    raise ArtifactFailure("ARTIFACT_IO_ERROR")
                offset += item.next


def copy_output(io, roots, output):
    parts, name, size = output_spec(output)
    staging, temporary, published = roots
    with io.directory(staging, parts[:-1]) as parent:
        source = io.open(parent, parts[-1], directory=False, share=SHARE_ALL)
        destination = bucket = None
        committed = False
        try:
            before = io.identity(source, size)
            destination = io.open(
                temporary,
                name + ".adopting",
                directory=False,
                create=True,
                writable=True,
                delete=True,
                share=0,
            )
            remaining = size
            while remaining:
                chunk = io.read(source, min(CHUNK_BYTES, remaining))
                if not chunk:
                    raise ArtifactFailure("ARTIFACT_INVALID")
                remaining -= len(chunk)
                io.write(destination, chunk)
            if io.read(source, 1) or io.identity(source, size) != before:
                raise ArtifactFailure("ARTIFACT_INVALID")
            io.identity(destination, size)
            io.flush(destination)
            try:
                bucket = io.open(published, name[-2:], directory=True, create=True)
            except OSError as error:
                if error.winerror not in (80, 183):
                    raise
                bucket = io.open(published, name[-2:], directory=True)
            info, _ = io.info(bucket)
            if info.attributes & REPARSE:
                raise ArtifactFailure("ARTIFACT_PATH_INVALID")
            # Rename the private copy by handle. The Engine inode is never moved.
            io.rename(destination, bucket, name)
            committed = True
        finally:
            if destination is not None:
                if not committed:
                    with contextlib.suppress(OSError):
                        io.delete(destination)
                io.close(destination)
            if bucket is not None:
                io.close(bucket)
            io.close(source)


def remove_entry(io, parent, name, budget, depth=0):
    budget[0] -= 1
    if budget[0] < 0 or depth > 32:
        raise ArtifactFailure("RESOURCE_LIMIT_EXCEEDED")
    try:
        handle = io.open(parent, name, delete=True)
    except OSError as error:
        if error.winerror in (2, 3):
            return
        raise
    try:
        info, _ = io.info(handle)
        if info.attributes & DIRECTORY and not info.attributes & REPARSE:
            for child in io.names(handle):
                remove_entry(io, handle, child, budget, depth + 1)
        # Delete the opened entry, including a reparse point itself, never its target.
        io.delete(handle)
    finally:
        io.close(handle)


def cleanup(io, roots, request):
    staging = roots[0]
    budget = [MAX_ENTRIES]
    if request.get("all") is True:
        for name in io.names(staging):
            remove_entry(io, staging, name, budget)
        return
    parts = components(request["job"])
    if len(parts) != 2:
        raise ArtifactFailure("ARTIFACT_PATH_INVALID")
    try:
        with io.directory(staging, parts[:1]) as parent:
            files = request.get("files")
            if files is None:
                remove_entry(io, parent, parts[1], budget)
            else:
                if not isinstance(files, list) or len(files) > 64:
                    raise ArtifactFailure("ARTIFACT_INVALID")
                for value in files:
                    path = components(value)
                    if path[:2] != parts or len(path) < 3:
                        raise ArtifactFailure("ARTIFACT_PATH_INVALID")
                    with io.directory(parent, path[1:-1]) as directory:
                        # Preview cleanup removes only a leaf, not a directory tree.
                        handle = io.open(directory, path[-1], directory=False, delete=True)
                        try:
                            io.delete(handle)
                        finally:
                            io.close(handle)
    except OSError as error:
        if error.winerror not in (2, 3):
            raise


def main():
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ArtifactFailure("ARTIFACT_INVALID")
    request = json.loads(raw)
    parent_pid = request["parentPid"]
    if type(parent_pid) is not int or parent_pid != os.getppid():
        raise ArtifactFailure("ARTIFACT_IO_ERROR")
    io = WindowsIO()
    parent = io.check(io.kernel.OpenProcess(0x40 | SYNCHRONIZE, False, parent_pid))
    current = io.kernel.GetCurrentProcess()

    def watch_parent():
        # Unlike getppid polling, a process handle cannot be confused by PID reuse.
        io.kernel.WaitForSingleObject(parent, 0xFFFFFFFF)
        os._exit(1)

    threading.Thread(target=watch_parent, daemon=True).start()
    operation = request["operation"]
    if operation == "initialize":
        paths = request["roots"]
        if not isinstance(paths, list) or len(paths) != 3:
            raise ArtifactFailure("ARTIFACT_INVALID")
        local, remote = [], []
        try:
            for path in paths:
                local.append(io.root(path))
            identities = [io.info(handle)[0] for handle in local]
            if identities[1].volume != identities[2].volume:
                raise ArtifactFailure("ARTIFACT_PATH_INVALID")
            if len({(item.volume, item.index_high, item.index_low) for item in identities}) != 3:
                raise ArtifactFailure("ARTIFACT_PATH_INVALID")
            for handle in local:
                remote.append(io.duplicate(current, handle, parent))
            print(json.dumps(remote), flush=True)
        except BaseException:
            for handle in remote:
                with contextlib.suppress(OSError):
                    io.close(io.duplicate(parent, handle, current, close_source=True))
            raise
        finally:
            for handle in local:
                io.close(handle)
        return
    handles = request["handles"]
    if (
        not isinstance(handles, list)
        or len(handles) != 3
        or len(set(handles)) != 3
        or any(type(handle) is not int or not 0 < handle < 2**53 for handle in handles)
    ):
        raise ArtifactFailure("ARTIFACT_INVALID")
    roots = []
    try:
        for handle in handles:
            roots.append(io.duplicate(parent, handle, current, close_source=operation == "close"))
        if operation == "close":
            return
        for handle in roots:
            info, _ = io.info(handle)
            if not info.attributes & DIRECTORY or info.attributes & REPARSE:
                raise ArtifactFailure("ARTIFACT_PATH_INVALID")
        if operation == "copy":
            outputs = request["outputs"]
            if not isinstance(outputs, list) or not 1 <= len(outputs) <= 64:
                raise ArtifactFailure("ARTIFACT_INVALID")
            specs = [output_spec(output) for output in outputs]
            if sum(spec[2] for spec in specs) > MAX_BYTES:
                raise ArtifactFailure("RESOURCE_LIMIT_EXCEEDED")
            for output in outputs:
                copy_output(io, roots, output)
        elif operation == "cleanup":
            cleanup(io, roots, request)
        else:
            raise ArtifactFailure("ARTIFACT_INVALID")
    finally:
        for handle in roots:
            io.close(handle)
        # The parent watcher owns its process handle until this helper exits.


if __name__ == "__main__":
    try:
        main()
    except ArtifactFailure as error:
        print(str(error))
        sys.exit(1)
    except OSError as error:
        code = "ARTIFACT_IO_ERROR"
        if error.winerror in (112, 1816):
            code = "DISK_SPACE_INSUFFICIENT"
        elif error.winerror in (123, 267, 1920, 4390, 4392, 4393, 4394):
            code = "ARTIFACT_PATH_INVALID"
        print(code)
        sys.exit(1)
    except (KeyError, TypeError, ValueError, UnicodeError):
        print("ARTIFACT_INVALID")
        sys.exit(1)
