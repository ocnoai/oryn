"""Data-only CPython parser. Launched with -I -S, no site, credentials or network.

This module intentionally has no Oryn/third-party imports. Candidate bytes are
never compiled into bytecode, imported or evaluated. AST equivalence identifies
a reviewed contract; it does not prove arbitrary Python semantically equivalent.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import re
import sys
import tokenize

NORMALIZER = "oryn.pyast.cpython3.12.16-grammar3.11.v1"
MAX_SOURCE_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 1400 * 1024
MAX_RESULT_BYTES = 64 * 1024
MAX_NODES = 100_000
MAX_DEPTH = 100
MAX_CANONICAL_BYTES = 8 * 1024 * 1024
AST_SCHEMA = {
    "Add": [],
    "And": [],
    "AnnAssign": ["target", "annotation", "value", "simple"],
    "Assert": ["test", "msg"],
    "Assign": ["targets", "value", "type_comment"],
    "AsyncFor": ["target", "iter", "body", "orelse", "type_comment"],
    "AsyncFunctionDef": [
        "name",
        "args",
        "body",
        "decorator_list",
        "returns",
        "type_comment",
        "type_params",
    ],
    "AsyncWith": ["items", "body", "type_comment"],
    "Attribute": ["value", "attr", "ctx"],
    "AugAssign": ["target", "op", "value"],
    "AugLoad": [],
    "AugStore": [],
    "Await": ["value"],
    "BinOp": ["left", "op", "right"],
    "BitAnd": [],
    "BitOr": [],
    "BitXor": [],
    "BoolOp": ["op", "values"],
    "Break": [],
    "Call": ["func", "args", "keywords"],
    "ClassDef": ["name", "bases", "keywords", "body", "decorator_list", "type_params"],
    "Compare": ["left", "ops", "comparators"],
    "Constant": ["value", "kind"],
    "Continue": [],
    "Del": [],
    "Delete": ["targets"],
    "Dict": ["keys", "values"],
    "DictComp": ["key", "value", "generators"],
    "Div": [],
    "Eq": [],
    "ExceptHandler": ["type", "name", "body"],
    "Expr": ["value"],
    "Expression": ["body"],
    "ExtSlice": [],
    "FloorDiv": [],
    "For": ["target", "iter", "body", "orelse", "type_comment"],
    "FormattedValue": ["value", "conversion", "format_spec"],
    "FunctionDef": [
        "name",
        "args",
        "body",
        "decorator_list",
        "returns",
        "type_comment",
        "type_params",
    ],
    "FunctionType": ["argtypes", "returns"],
    "GeneratorExp": ["elt", "generators"],
    "Global": ["names"],
    "Gt": [],
    "GtE": [],
    "If": ["test", "body", "orelse"],
    "IfExp": ["test", "body", "orelse"],
    "Import": ["names"],
    "ImportFrom": ["module", "names", "level"],
    "In": [],
    "Index": [],
    "Interactive": ["body"],
    "Invert": [],
    "Is": [],
    "IsNot": [],
    "JoinedStr": ["values"],
    "LShift": [],
    "Lambda": ["args", "body"],
    "List": ["elts", "ctx"],
    "ListComp": ["elt", "generators"],
    "Load": [],
    "Lt": [],
    "LtE": [],
    "MatMult": [],
    "Match": ["subject", "cases"],
    "MatchAs": ["pattern", "name"],
    "MatchClass": ["cls", "patterns", "kwd_attrs", "kwd_patterns"],
    "MatchMapping": ["keys", "patterns", "rest"],
    "MatchOr": ["patterns"],
    "MatchSequence": ["patterns"],
    "MatchSingleton": ["value"],
    "MatchStar": ["name"],
    "MatchValue": ["value"],
    "Mod": [],
    "Module": ["body", "type_ignores"],
    "Mult": [],
    "Name": ["id", "ctx"],
    "NamedExpr": ["target", "value"],
    "Nonlocal": ["names"],
    "Not": [],
    "NotEq": [],
    "NotIn": [],
    "Or": [],
    "Param": [],
    "ParamSpec": ["name"],
    "Pass": [],
    "Pow": [],
    "RShift": [],
    "Raise": ["exc", "cause"],
    "Return": ["value"],
    "Set": ["elts"],
    "SetComp": ["elt", "generators"],
    "Slice": ["lower", "upper", "step"],
    "Starred": ["value", "ctx"],
    "Store": [],
    "Sub": [],
    "Subscript": ["value", "slice", "ctx"],
    "Suite": [],
    "Try": ["body", "handlers", "orelse", "finalbody"],
    "TryStar": ["body", "handlers", "orelse", "finalbody"],
    "Tuple": ["elts", "ctx"],
    "TypeAlias": ["name", "type_params", "value"],
    "TypeIgnore": ["lineno", "tag"],
    "TypeVar": ["name", "bound"],
    "TypeVarTuple": ["name"],
    "UAdd": [],
    "USub": [],
    "UnaryOp": ["op", "operand"],
    "While": ["test", "body", "orelse"],
    "With": ["items", "body", "type_comment"],
    "Yield": ["value"],
    "YieldFrom": ["value"],
    "_ast_Ellipsis": [],
    "alias": ["name", "asname"],
    "arg": ["arg", "annotation", "type_comment"],
    "arguments": [
        "posonlyargs",
        "args",
        "vararg",
        "kwonlyargs",
        "kw_defaults",
        "kwarg",
        "defaults",
    ],
    "boolop": [],
    "cmpop": [],
    "comprehension": ["target", "iter", "ifs", "is_async"],
    "excepthandler": [],
    "expr": [],
    "expr_context": [],
    "keyword": ["arg", "value"],
    "match_case": ["pattern", "guard", "body"],
    "mod": [],
    "operator": [],
    "pattern": [],
    "slice": [],
    "stmt": [],
    "type_ignore": [],
    "type_param": [],
    "unaryop": [],
    "withitem": ["context_expr", "optional_vars"],
}


class InspectionError(ValueError):
    pass


def _literal(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return ["int", hex(value)]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, complex):
        return ["complex", value.real.hex(), value.imag.hex()]
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if value is Ellipsis:
        return ["ellipsis"]
    raise InspectionError("Unsupported AST literal")


def inspect_source(raw: bytes) -> dict:
    if sys.version_info[:3] != (3, 12, 16):
        raise InspectionError("Unqualified parser version")
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_SOURCE_BYTES:
        raise InspectionError("Source byte limit")
    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    if encoding not in {"utf-8", "utf-8-sig"}:
        raise InspectionError("Only UTF-8 source is qualified")
    text = raw.decode(encoding)
    # Type comments are AST fields. Line-scoped type ignores and tool directives
    # have no general position-independent equivalence: exact recognition only.
    sensitive = False
    for tokens, token in enumerate(tokenize.generate_tokens(io.StringIO(text).readline), 1):
        if tokens > 200_000:
            raise InspectionError("Token limit")
        if token.type == tokenize.COMMENT and re.match(
            r"#\s*(?:type:\s*ignore|noqa\b|pylint:|mypy:|pyright:|pragma:|jit:|triton:)",
            token.string,
            re.IGNORECASE,
        ):
            sensitive = True
    tree = ast.parse(text, filename="<inert-source>", type_comments=True, feature_version=(3, 11))
    nodes = 0
    imports: list[dict] = []
    exports = sorted(n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)))

    def canonical(value, depth=0):
        nonlocal nodes
        if depth > MAX_DEPTH:
            raise InspectionError("AST depth limit")
        if isinstance(value, ast.AST):
            nodes += 1
            if nodes > MAX_NODES:
                raise InspectionError("AST node limit")
            name = type(value).__name__
            fields = AST_SCHEMA.get(name)
            if fields is None or tuple(value._fields) != tuple(fields):
                raise InspectionError("Unknown AST kind or fields")
            if isinstance(value, ast.ImportFrom) and value.level:
                imports.append(
                    {
                        "level": value.level,
                        "module": value.module,
                        "names": [item.name for item in value.names],
                    }
                )
                if len(imports) > 256:
                    raise InspectionError("Import limit")
            # Source-position attributes are the ONLY omitted AST attributes.
            return [
                name,
                [[field, canonical(getattr(value, field), depth + 1)] for field in fields],
            ]
        if isinstance(value, list):
            return [canonical(item, depth + 1) for item in value]
        return _literal(value)

    representation = [NORMALIZER, "utf-8", canonical(tree)]
    digest = hashlib.sha256()
    size = 0
    for chunk in json.JSONEncoder(ensure_ascii=True, separators=(",", ":")).iterencode(
        representation
    ):
        encoded = chunk.encode("ascii")
        size += len(encoded)
        if size > MAX_CANONICAL_BYTES:
            raise InspectionError("Canonical byte limit")
        digest.update(encoded)
    result = {
        "schemaVersion": 1,
        "normalizer": NORMALIZER,
        "rawSha256": hashlib.sha256(raw).hexdigest(),
        "structuralSha256": None if sensitive else digest.hexdigest(),
        "sizeBytes": len(raw),
        "imports": imports,
        "exports": exports,
    }
    if len(json.dumps(result).encode()) > MAX_RESULT_BYTES:
        raise InspectionError("Inspection result limit")
    return result


def main():
    import resource

    # Apply limits in the fixed helper, before reading any hostile bytes. Avoid
    # preexec_fn: the authenticated broker is multithreaded.
    for kind, amount in (
        # Darwin rejects finite AS/DATA limits. The parent enforces resident
        # memory independently; this helper cannot disable that watchdog.
        (resource.RLIMIT_CPU, 3),
        (resource.RLIMIT_NOFILE, 32),
        (resource.RLIMIT_FSIZE, 0),
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_NPROC, 1),
    ):
        _, hard = resource.getrlimit(kind)
        limit = amount if hard == resource.RLIM_INFINITY else min(amount, hard)
        try:
            resource.setrlimit(kind, (limit, limit))
        except (ValueError, OSError) as exc:
            raise RuntimeError(
                f"Parser resource limit unavailable: {kind}, {limit}, {hard}"
            ) from exc
    try:
        frame = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(frame) > MAX_REQUEST_BYTES:
            raise InspectionError("Request limit")
        request = json.loads(frame)
        if not isinstance(request, dict) or set(request) != {"source"}:
            raise InspectionError("Invalid request")
        raw = base64.b64decode(request["source"], validate=True)
        result = {"ok": True, "result": inspect_source(raw)}
    except (
        ValueError,
        TypeError,
        SyntaxError,
        UnicodeError,
        MemoryError,
        RecursionError,
        tokenize.TokenError,
        LookupError,
    ):
        result = {"ok": False, "error": "SOURCE_INSPECTION_REJECTED"}
    sys.stdout.write(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
