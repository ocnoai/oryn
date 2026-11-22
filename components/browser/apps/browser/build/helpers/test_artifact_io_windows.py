"""Portable policy tests plus native Windows tests. Run only in an OS sandbox."""

import contextlib
import ctypes
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "windows_artifact_io", Path(__file__).with_name("artifact_io_windows.py")
)
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)
NAME = "object_" + "a" * 32


class MemoryIO:
    """Small test double for copy policy, not evidence of native Windows behavior."""

    def __init__(self, data=b"data"):
        self.source = data
        self.position = 0
        self.written = bytearray()
        self.deleted = []
        self.closed = []
        self.renamed = False
        self.flushed = False
        self.changed = False
        self.read_sizes = []
        self.identities = 0

    @contextlib.contextmanager
    def directory(self, root, parts):
        yield "parent"

    def open(self, root, name, **options):
        if root == "parent":
            return "source"
        if root == "temporary":
            if not options.get("create") or options.get("share") != 0:
                raise AssertionError("Destination must be exclusively created")
            return "destination"
        return "bucket"

    def identity(self, handle, size):
        if handle == "source":
            self.identities += 1
            return ("source", self.changed and self.identities > 1)
        if len(self.written) != size:
            raise backend.ArtifactFailure("ARTIFACT_INVALID")
        return ("destination",)

    def read(self, handle, count):
        self.read_sizes.append(count)
        data = self.source[self.position : self.position + count]
        self.position += len(data)
        return data

    def write(self, handle, data):
        self.written.extend(data)

    def flush(self, handle):
        self.flushed = True

    def info(self, handle):
        return SimpleNamespace(attributes=backend.DIRECTORY), None

    def rename(self, handle, bucket, name):
        if not self.flushed:
            raise AssertionError("Publication preceded flush")
        self.renamed = True

    def delete(self, handle):
        self.deleted.append(handle)

    def close(self, handle):
        self.closed.append(handle)


class PolicyTests(unittest.TestCase):
    def test_rejects_windows_aliases_and_escape_paths(self):
        for value in [
            "",
            "/a",
            "a/../b",
            "a/./b",
            "a//b",
            "C:\\a",
            "\\\\server\\share",
            "job/file:stream",
            "job/file.",
            "job/file ",
            "job/NUL",
            "job/con.txt",
            "job/COM¹",
            "job/a\0b",
            "job/a?b",
            "/".join(["a"] * 33),
        ]:
            with self.subTest(value=value), self.assertRaises(backend.ArtifactFailure):
                backend.components(value)
        self.assertEqual(
            backend.components("engine\\job\\résultat.bin"), ["engine", "job", "résultat.bin"]
        )

    def test_bounds_and_types(self):
        for size in [-1, True, 1.5, backend.MAX_BYTES + 1]:
            with self.subTest(size=size), self.assertRaises(backend.ArtifactFailure):
                backend.output_spec({"source": "e/j/file", "object": NAME, "size": size})

    def test_windows_64_bit_abi_layouts_and_exclusive_rename(self):
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            self.skipTest("64-bit layout")
        self.assertEqual(ctypes.sizeof(backend.FileInfo), 52)
        self.assertEqual(ctypes.sizeof(backend.ObjectAttributes), 48)
        self.assertEqual(ctypes.sizeof(backend.IoStatus), 16)
        self.assertEqual(backend.DirectoryInfo.name.offset, 104)
        self.assertEqual(backend.RenameInfo.name.offset, 20)
        buffer = backend.rename_buffer(1234, "résultat.bin")
        info = backend.RenameInfo.from_buffer(buffer)
        self.assertEqual(info.flags, 0)
        self.assertEqual(info.root, 1234)
        self.assertEqual(info.length, len("résultat.bin".encode("utf-16-le")))
        self.assertEqual(buffer.raw[20 : 20 + info.length].decode("utf-16-le"), "résultat.bin")

    def copy(self, io, size):
        backend.copy_output(
            io,
            ["staging", "temporary", "published"],
            {"source": "e/j/file", "object": NAME, "size": size},
        )

    def test_multichunk_copy_is_flushed_before_publication(self):
        data = b"x" * (backend.CHUNK_BYTES * 2 + 137)
        io = MemoryIO(data)
        self.copy(io, len(data))
        self.assertEqual(io.written, data)
        self.assertLessEqual(max(io.read_sizes), backend.CHUNK_BYTES)
        self.assertTrue(io.renamed)
        self.assertEqual(io.deleted, [])
        self.assertCountEqual(io.closed, ["source", "destination", "bucket"])

    def test_empty_file(self):
        io = MemoryIO(b"")
        self.copy(io, 0)
        self.assertTrue(io.renamed)

    def test_short_growing_and_changed_source_never_publish(self):
        for data, size, changed in [(b"x", 2, False), (b"xx", 1, False), (b"x", 1, True)]:
            io = MemoryIO(data)
            io.changed = changed
            with (
                self.subTest(data=data, size=size, changed=changed),
                self.assertRaises(backend.ArtifactFailure),
            ):
                self.copy(io, size)
            self.assertFalse(io.renamed)
            self.assertEqual(io.deleted, ["destination"])
            self.assertCountEqual(io.closed, ["source", "destination"])

    def test_failed_rename_deletes_copy_by_handle(self):
        io = MemoryIO()

        def fail(*args):
            raise OSError("collision")

        io.rename = fail
        with self.assertRaises(OSError):
            self.copy(io, 4)
        self.assertEqual(io.deleted, ["destination"])
        self.assertCountEqual(io.closed, ["source", "destination", "bucket"])

    def test_cleanup_does_not_enumerate_reparse_target(self):
        io = MemoryIO()
        io.info = lambda handle: (
            SimpleNamespace(attributes=backend.DIRECTORY | backend.REPARSE),
            None,
        )
        io.names = lambda handle: self.fail("Traversed a reparse point")
        backend.remove_entry(io, "parent", "junction", [1])
        self.assertEqual(io.deleted, ["source"])
        self.assertEqual(io.closed, ["source"])

    def test_cleanup_budget_precedes_open(self):
        io = MemoryIO()
        io.open = lambda *args, **kwargs: self.fail("Opened past budget")
        for budget, depth in [(0, 0), (10, 33)]:
            with self.assertRaises(backend.ArtifactFailure):
                backend.remove_entry(io, "parent", "child", [budget], depth)


@unittest.skipUnless(os.name == "nt", "Native Windows APIs require Windows")
class NativeWindowsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="oryn-windows-io-")
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.paths = [self.base / name for name in ["staging", "temporary", "published"]]
        for path in self.paths:
            path.mkdir()
        self.io = backend.WindowsIO()
        self.roots = []
        self.addCleanup(lambda: [self.io.close(handle) for handle in reversed(self.roots)])
        for path in self.paths:
            self.roots.append(self.io.root(str(path)))
        self.job = self.paths[0] / "engine" / "job"
        self.job.mkdir(parents=True)
        self.source = self.job / "output.bin"

    def copy(self, size, name=NAME):
        backend.copy_output(
            self.io, self.roots, {"source": "engine\\job\\output.bin", "object": name, "size": size}
        )
        return self.paths[2] / name[-2:] / name

    def test_real_copy_retains_identity_and_cleanup(self):
        data = b"a" * (2 * backend.CHUNK_BYTES + 137)
        self.source.write_bytes(data)
        with self.source.open("r+b") as writer:
            destination = self.copy(len(data))
            self.assertNotEqual(self.source.stat().st_ino, destination.stat().st_ino)
            writer.write(b"changed")
            writer.flush()
            self.assertEqual(destination.read_bytes(), data)
        backend.cleanup(self.io, self.roots, {"job": "engine\\job"})
        self.assertFalse(self.job.exists())
        self.assertEqual(list(self.paths[1].iterdir()), [])
        destination.unlink()

    def test_hard_link_rejected(self):
        self.source.write_bytes(b"data")
        os.link(self.source, self.job / "second.bin")
        with self.assertRaises(backend.ArtifactFailure):
            self.copy(4)
        self.assertEqual(list(self.paths[1].iterdir()), [])

    def test_existing_object_is_not_replaced(self):
        self.source.write_bytes(b"first")
        destination = self.copy(5)
        self.source.write_bytes(b"other")
        with self.assertRaises(OSError):
            self.copy(5)
        self.assertEqual(destination.read_bytes(), b"first")
        self.assertEqual(list(self.paths[1].iterdir()), [])

    def test_native_wrong_size_removes_partial(self):
        self.source.write_bytes(b"data")
        with self.assertRaises(backend.ArtifactFailure):
            self.copy(5)
        self.assertEqual(list(self.paths[1].iterdir()), [])

    def test_native_helper_retains_roots_in_parent_across_processes(self):
        helper = Path(__file__).with_name("artifact_io_windows.py")

        def run(request):
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", str(helper)],
                input=json.dumps({**request, "parentPid": os.getpid()}),
                text=True,
                capture_output=True,
                timeout=10,
                env={"SystemRoot": os.environ["SYSTEMROOT"], "PATH": ""},
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            return result.stdout

        handles = json.loads(run({"operation": "initialize", "roots": list(map(str, self.paths))}))
        try:
            self.assertEqual(len(set(handles)), 3)
            self.source.write_bytes(b"native helper copy")
            run(
                {
                    "operation": "copy",
                    "handles": handles,
                    "outputs": [
                        {
                            "source": "engine\\job\\output.bin",
                            "object": NAME,
                            "size": 18,
                        }
                    ],
                }
            )
            self.assertEqual((self.paths[2] / NAME[-2:] / NAME).read_bytes(), b"native helper copy")
            run({"operation": "cleanup", "handles": handles, "job": "engine\\job"})
            self.assertFalse(self.job.exists())
        finally:
            run({"operation": "close", "handles": handles})

    def test_native_reparse_ancestor_rejected_and_link_cleanup_confined(self):
        outside = self.base / "dummy-outside"
        outside.mkdir()
        target = outside / "output.bin"
        target.write_bytes(b"unchanged")
        link = self.job / "linked"
        try:
            os.symlink(outside, link, target_is_directory=True)
        except OSError as error:
            if error.winerror == 1314:
                self.skipTest("Native symlink fixture requires Developer Mode or symlink privilege")
            raise
        with self.assertRaises(backend.ArtifactFailure):
            backend.copy_output(
                self.io,
                self.roots,
                {
                    "source": "engine/job/linked/output.bin",
                    "object": NAME,
                    "size": 9,
                },
            )
        backend.cleanup(self.io, self.roots, {"job": "engine/job"})
        self.assertEqual(target.read_bytes(), b"unchanged")
        self.assertFalse(self.job.exists())


if __name__ == "__main__":
    unittest.main()
