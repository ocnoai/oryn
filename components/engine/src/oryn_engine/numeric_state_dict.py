"""Closed tensor-directory grammar without object deserialization.

GLOBAL/REDUCE/BUILD are parsed as inert format tags. Nothing named in the file is
imported, called, constructed, or assigned attributes. Only plain dense tensor
descriptors and validated module-version bookkeeping can result.
"""

import math
import pickletools
import re
from dataclasses import dataclass

MAX_BYTES = 8 * 1024**2
MAX_TENSORS = 20000
MAX_OPS = 1000000
MAX_MEMO = 200000
DTYPES = {
    "FloatStorage": ("F32", 4),
    "HalfStorage": ("F16", 2),
    "BFloat16Storage": ("BF16", 2),
    "DoubleStorage": ("F64", 8),
    "LongStorage": ("I64", 8),
    "IntStorage": ("I32", 4),
    "ShortStorage": ("I16", 2),
    "CharStorage": ("I8", 1),
    "ByteStorage": ("U8", 1),
    "BoolStorage": ("BOOL", 1),
}


@dataclass(frozen=True)
class Symbol:
    module: str
    name: str


@dataclass(frozen=True)
class Storage:
    key: str
    dtype: str
    elements: int
    itemsize: int


@dataclass(frozen=True)
class Tensor:
    storage: Storage
    offset: int
    shape: tuple
    stride: tuple


class Map(dict):
    def __init__(self, *, ordered=False):
        super().__init__()
        self.ordered = ordered


def parse(raw):
    if not isinstance(raw, bytes) or not 3 <= len(raw) <= MAX_BYTES or raw[:2] != b"\x80\x02":
        raise ValueError("NUMERIC_ARCHIVE_GRAMMAR")
    stack, marks, memo, storages = [], [], {}, {}
    marker = object()
    count = 0
    metadata_seen = set()

    def assign(target, pairs):
        nonlocal count
        if type(target) is not Map or len(pairs) % 2:
            raise ValueError("NUMERIC_ARCHIVE_MAP")
        for i in range(0, len(pairs), 2):
            key, value = pairs[i : i + 2]
            if not isinstance(key, str) or len(key) > 512 or key in target:
                raise ValueError("NUMERIC_ARCHIVE_KEY")
            count += 1
            if count > 4 * MAX_TENSORS:
                raise ValueError("NUMERIC_ARCHIVE_LIMIT")
            target[key] = value

    def marked():
        if not marks:
            raise ValueError("NUMERIC_ARCHIVE_MARK")
        index = marks.pop()
        if index >= len(stack) or stack[index] is not marker:
            raise ValueError("NUMERIC_ARCHIVE_MARK")
        result = tuple(stack[index + 1 :])
        del stack[index:]
        return result

    def storage(value):
        if (
            not isinstance(value, tuple)
            or len(value) != 5
            or value[0] != "storage"
            or not isinstance(value[1], Symbol)
            or value[1].module != "torch"
            or value[1].name not in DTYPES
            or not isinstance(value[2], str)
            or not re.fullmatch(r"[0-9]{1,64}", value[2])
            or not isinstance(value[3], str)
            or not re.fullmatch(r"cpu|mps|cuda(?::[0-9]{1,3})?", value[3])
            or type(value[4]) is not int
            or not 0 <= value[4] <= 2**36
        ):
            raise ValueError("NUMERIC_ARCHIVE_STORAGE")
        dtype, width = DTYPES[value[1].name]
        item = Storage(value[2], dtype, value[4], width)
        if item.elements * item.itemsize > 256 * 1024**3:
            raise ValueError("NUMERIC_ARCHIVE_LIMIT")
        if item.key in storages and item != storages[item.key]:
            raise ValueError("NUMERIC_ARCHIVE_STORAGE_CONFLICT")
        storages[item.key] = item
        if len(storages) > MAX_TENSORS:
            raise ValueError("NUMERIC_ARCHIVE_LIMIT")
        return item

    def reduce(symbol, args):
        if not isinstance(symbol, Symbol) or not isinstance(args, tuple):
            raise ValueError("NUMERIC_ARCHIVE_REDUCE")
        if symbol == Symbol("collections", "OrderedDict") and args == ():
            return Map(ordered=True)
        if symbol != Symbol("torch._utils", "_rebuild_tensor_v2") or len(args) != 6:
            raise ValueError("NUMERIC_ARCHIVE_REDUCE")
        store, offset, shape, stride, gradients, hooks = args
        if (
            not isinstance(store, Storage)
            or type(offset) is not int
            or offset < 0
            or not isinstance(shape, tuple)
            or not isinstance(stride, tuple)
            or len(shape) != len(stride)
            or len(shape) > 8
            or any(type(x) is not int or not 1 <= x <= 2**32 for x in shape)
            or any(type(x) is not int or not 0 <= x <= 2**36 for x in stride)
            or gradients is not False
            or type(hooks) is not Map
            or hooks
        ):
            raise ValueError("NUMERIC_ARCHIVE_TENSOR")
        size = math.prod(shape)
        if size > 2**36 or offset + size > store.elements:
            raise ValueError("NUMERIC_ARCHIVE_TENSOR_RANGE")
        expected = 1
        # Dense permutations (including transposed linear weights) occupy exactly
        # prod(shape) elements. Reject overlapping, broadcast and gapped views.
        for step, dimension in sorted((s, d) for d, s in zip(shape, stride, strict=True) if d > 1):
            if step != expected:
                raise ValueError("NUMERIC_ARCHIVE_TENSOR_STRIDE")
            expected *= dimension
        return Tensor(store, offset, shape, stride)

    try:
        for n, (op, arg, position) in enumerate(pickletools.genops(raw)):
            if n >= MAX_OPS or len(stack) > 4 * MAX_TENSORS or len(memo) > MAX_MEMO:
                raise ValueError("NUMERIC_ARCHIVE_LIMIT")
            name = op.name
            if name == "PROTO":
                if position != 0 or arg != 2:
                    raise ValueError("NUMERIC_ARCHIVE_GRAMMAR")
            elif name == "GLOBAL":
                module, target = arg.split(" ")
                if not (
                    (module, target)
                    in {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2")}
                    or module == "torch"
                    and target in DTYPES
                ):
                    raise ValueError("NUMERIC_ARCHIVE_GLOBAL")
                stack.append(Symbol(module, target))
            elif name in {"BININT", "BININT1", "BININT2", "LONG1", "LONG4"}:
                if type(arg) is not int or not -(2**63) <= arg < 2**63:
                    raise ValueError("NUMERIC_ARCHIVE_INTEGER")
                stack.append(arg)
            elif name == "BINUNICODE":
                if not isinstance(arg, str) or len(arg) > 512:
                    raise ValueError("NUMERIC_ARCHIVE_STRING")
                stack.append(arg)
            elif name in {"NEWFALSE", "NEWTRUE"}:
                stack.append(name == "NEWTRUE")
            elif name == "EMPTY_TUPLE":
                stack.append(())
            elif name == "EMPTY_DICT":
                stack.append(Map())
            elif name == "MARK":
                marks.append(len(stack))
                if len(marks) > 16:
                    raise ValueError("NUMERIC_ARCHIVE_DEPTH")
                stack.append(marker)
            elif name == "TUPLE":
                stack.append(marked())
            elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
                length = int(name[-1])
                if len(stack) < length or (marks and len(stack) - length <= marks[-1]):
                    raise ValueError("NUMERIC_ARCHIVE_TUPLE")
                value = tuple(stack[-length:])
                del stack[-length:]
                stack.append(value)
            elif name in {"BINPUT", "LONG_BINPUT"}:
                if not stack or stack[-1] is marker or type(arg) is not int or arg != len(memo):
                    raise ValueError("NUMERIC_ARCHIVE_MEMO")
                memo[arg] = stack[-1]
            elif name in {"BINGET", "LONG_BINGET"}:
                stack.append(memo[arg])
            elif name == "BINPERSID":
                stack.append(storage(stack.pop()))
            elif name == "REDUCE":
                args, symbol = stack.pop(), stack.pop()
                stack.append(reduce(symbol, args))
            elif name == "SETITEM":
                value, key = stack.pop(), stack.pop()
                assign(stack[-1], (key, value))
            elif name == "SETITEMS":
                values = marked()
                assign(stack[-1], values)
            elif name == "BUILD":
                state, target = stack.pop(), stack[-1]
                if (
                    type(target) is not Map
                    or not target.ordered
                    or id(target) in metadata_seen
                    or type(state) is not Map
                    or set(state) != {"_metadata"}
                    or type(state["_metadata"]) is not Map
                ):
                    raise ValueError("NUMERIC_ARCHIVE_BOOKKEEPING")
                for key, item in state["_metadata"].items():
                    if (
                        not re.fullmatch(r"[A-Za-z0-9_.]{0,512}", key)
                        or type(item) is not Map
                        or set(item) != {"version"}
                        or type(item["version"]) is not int
                        or not 0 <= item["version"] <= 100
                    ):
                        raise ValueError("NUMERIC_ARCHIVE_BOOKKEEPING")
                metadata_seen.add(id(target))
            elif name == "STOP":
                if position != len(raw) - 1 or marks or len(stack) != 1:
                    raise ValueError("NUMERIC_ARCHIVE_TRAILING")
                result = stack[0]
                if type(result) is not Map or not 1 <= len(result) <= MAX_TENSORS:
                    raise ValueError("NUMERIC_ARCHIVE_ROOT")
                if any(
                    not re.fullmatch(r"[A-Za-z0-9_.]{1,512}", key) or not isinstance(value, Tensor)
                    for key, value in result.items()
                ):
                    raise ValueError("NUMERIC_ARCHIVE_ROOT")
                if {x.storage.key for x in result.values()} != set(storages):
                    raise ValueError("NUMERIC_ARCHIVE_UNUSED_STORAGE")
                return dict(result)
            else:
                raise ValueError("NUMERIC_ARCHIVE_OPCODE")
    except (KeyError, IndexError, TypeError, OverflowError) as exc:
        raise ValueError("NUMERIC_ARCHIVE_GRAMMAR") from exc
    raise ValueError("NUMERIC_ARCHIVE_TRUNCATED")
