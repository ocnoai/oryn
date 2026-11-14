"""Reviewed chat framing shared by every engine.

A model's own chat template is read as text only, to choose one of these reviewed
formats by its markers. It is never compiled or rendered. Oryn builds every prompt
from the signed table in ``build-inputs/chat-formats.json``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

NAME = re.compile(r"[a-z][a-z0-9-]{1,31}")
ROLES = ("system", "user", "assistant")
SYSTEM_MODES = ("turn", "merge-into-next-user", "lead")
FIELDS = {
    "recognize",
    "turn",
    "turns",
    "roles",
    "system",
    "trim",
    "begin",
    "reply",
    "thinkingOff",
    "image",
    "stop",
    "reserved",
    "llamaCppBuiltin",
    "reviewedFrom",
}
MAX_TABLE_BYTES = 64 * 1024
MAX_FORMATS = 64


def _text(value: object, *, empty: bool = False, limit: int = 256) -> str:
    if not isinstance(value, str) or len(value) > limit or (not value and not empty):
        raise ValueError("Chat format text is invalid")
    return value


def _markers(value: object, *, empty: bool = True) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > 16
        or (not value and not empty)
        or len(set(value)) != len(value)
    ):
        raise ValueError("Chat format marker list is invalid")
    for marker in value:
        _text(marker, limit=64)
    return value


def _pattern(value: object, *, role: bool) -> str:
    text = _text(value)
    if "{content}" not in text or (role and "{role}" not in text):
        raise ValueError("Chat format turn must place its content")
    rest = re.sub(r"\{(?:role|content)\}", "", text)
    if "{" in rest or "}" in rest:
        raise ValueError("Chat format turn has an unknown placeholder")
    return text


def validate_table(table: object) -> dict[str, dict[str, Any]]:
    """Validate the closed table at build time and again when the signed copy loads."""

    if (
        not isinstance(table, dict)
        or set(table) != {"schemaVersion", "formats"}
        or table["schemaVersion"] != 1
        or not isinstance(table["formats"], dict)
        or not 1 <= len(table["formats"]) <= MAX_FORMATS
    ):
        raise ValueError("Chat format table is invalid")
    formats: dict[str, dict[str, Any]] = {}
    for name, entry in table["formats"].items():
        if not isinstance(name, str) or not NAME.fullmatch(name) or not isinstance(entry, dict):
            raise ValueError("Chat format name is invalid")
        if set(entry) - FIELDS or "reply" not in entry or "recognize" not in entry:
            raise ValueError(f"Chat format {name} has unknown or missing fields")
        recognize = _markers(entry["recognize"], empty=name == "plain")
        if name == "plain" and recognize:
            raise ValueError("The plain format is chosen only when a model has no template")
        turns = entry.get("turns", {})
        if not isinstance(turns, dict) or set(turns) - set(ROLES):
            raise ValueError(f"Chat format {name} turns are invalid")
        for pattern in turns.values():
            _pattern(pattern, role=False)
        if "turn" in entry:
            _pattern(entry["turn"], role=True)
        elif set(turns) != set(ROLES):
            raise ValueError(f"Chat format {name} does not cover every role")
        roles = entry.get("roles", {})
        if not isinstance(roles, dict) or set(roles) - set(ROLES):
            raise ValueError(f"Chat format {name} role labels are invalid")
        for label in roles.values():
            if not re.fullmatch(r"[A-Za-z][A-Za-z_ -]{0,31}", _text(label, limit=32)):
                raise ValueError(f"Chat format {name} role label is invalid")
        if entry.get("system", "turn") not in SYSTEM_MODES:
            raise ValueError(f"Chat format {name} system mode is invalid")
        trim = entry.get("trim", False)
        if not isinstance(trim, bool) and (
            not isinstance(trim, list) or set(trim) - set(ROLES) or len(set(trim)) != len(trim)
        ):
            raise ValueError(f"Chat format {name} trim is invalid")
        _text(entry["reply"])
        for field in ("begin", "thinkingOff", "image"):
            if field in entry:
                _text(entry[field])
        for field in ("stop", "reserved"):
            if field in entry:
                _markers(entry[field])
        if "llamaCppBuiltin" in entry and not NAME.fullmatch(_text(entry["llamaCppBuiltin"])):
            raise ValueError(f"Chat format {name} llama.cpp renderer is invalid")
        if "reviewedFrom" in entry:
            for reference in _markers(entry["reviewedFrom"], empty=False):
                _text(reference, limit=160)
        formats[name] = entry
    if "plain" not in formats:
        raise ValueError("Chat format table must define the plain format")
    return formats


@lru_cache(maxsize=1)
def formats() -> dict[str, dict[str, Any]]:
    from oryn_engine.generated_capabilities import generated_artifact

    return validate_table(generated_artifact()["buildInputs"]["chatFormats"])


def names() -> tuple[str, ...]:
    return tuple(sorted(formats()))


def reserved_markers(name: str) -> tuple[str, ...]:
    entry = formats()[name]
    found = [*entry["recognize"], *entry.get("stop", []), *entry.get("reserved", [])]
    if entry.get("begin"):
        found.append(entry["begin"])
    found.extend(re.findall(r"<[^<>\s]{1,62}>", entry.get("image", "")))
    return tuple(dict.fromkeys(found))


def select(template: object, *, allowed: tuple[str, ...] | list[str] | None = None) -> str | None:
    """Choose a reviewed format from a template's text. Exactly one format must match."""

    if not isinstance(template, str) or not template:
        return None
    matches = [
        name
        for name, entry in formats().items()
        if entry["recognize"]
        and (allowed is None or name in allowed)
        and all(marker in template for marker in entry["recognize"])
    ]
    return matches[0] if len(matches) == 1 else None


def select_by_vocabulary(vocabulary, *, allowed=None) -> str | None:
    """Choose a reviewed format from special tokens when a tokenizer ships no template."""

    matches = [
        name
        for name, entry in formats().items()
        if entry["recognize"]
        and (allowed is None or name in allowed)
        and all(marker in vocabulary for marker in entry["recognize"])
    ]
    return matches[0] if len(matches) == 1 else None


def template_text(*sources: object) -> str | None:
    """Return the first template string a loaded tokenizer or processor carries, as data."""

    for source in sources:
        value = getattr(source, "chat_template", None) if source is not None else None
        if isinstance(value, dict):
            value = value.get("default")
        if isinstance(value, str) and value:
            return value
    return None


def template_from_config(config: object) -> str | None:
    """Return the template text a repository's metadata carries, as data, or None."""

    if not isinstance(config, dict):
        return None
    value = config.get("chat_template_jinja")
    if not isinstance(value, str) or not value:
        tokenizer = config.get("tokenizer_config")
        value = tokenizer.get("chat_template") if isinstance(tokenizer, dict) else None
    if isinstance(value, list):
        value = next(
            (
                item.get("template")
                for item in value
                if isinstance(item, dict) and item.get("name") == "default"
            ),
            None,
        )
    return value if isinstance(value, str) and value else None


def supports_image(name: str, vocabulary) -> bool:
    placeholder = formats()[name].get("image")
    tokens = re.findall(r"<[^<>\s]{1,62}>", placeholder or "")
    return bool(tokens) and all(token in vocabulary for token in tokens)


def render(
    name: str,
    messages: list[dict[str, str]],
    *,
    image_index: int | None = None,
    thinking_off: bool = False,
    bos: str = "",
) -> str:
    """Build a prompt from reviewed markers. Message text is data and cannot add framing."""

    entry = formats()[name]
    if not isinstance(messages, list) or not messages:
        raise ValueError("Chat messages are invalid")
    reserved = reserved_markers(name)
    # Without special tokens a role is plain text ("System: "), so a message line that starts
    # with a role label would be indistinguishable from a real turn.
    text_prefixes = (
        tuple(f"{entry.get('roles', {}).get(role, role)}:".lower() for role in ROLES)
        if not entry["recognize"]
        else ()
    )
    normalized = []
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ROLES
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError("Chat messages are invalid")
        if any(marker in message["content"] for marker in reserved) or any(
            line.lstrip().lower().startswith(prefix)
            for line in message["content"].splitlines()
            for prefix in text_prefixes
        ):
            raise ValueError("CHAT_CONTENT_RESERVED_FRAMING")
        normalized.append({"role": message["role"], "content": message["content"]})
    trim = entry.get("trim", False)
    trimmed = set(ROLES) if trim is True else set(trim or ())
    for message in normalized:
        if message["role"] in trimmed:
            message["content"] = message["content"].strip()
    if image_index is not None:
        if (
            "image" not in entry
            or not 0 <= image_index < len(normalized)
            or normalized[image_index]["role"] != "user"
        ):
            raise ValueError("This chat format cannot place an image here")
        normalized[image_index]["content"] = entry["image"] + normalized[image_index]["content"]
    mode = entry.get("system", "turn")
    if mode == "merge-into-next-user":
        merged, pending = [], ""
        for message in normalized:
            if message["role"] == "system":
                pending = f"{pending}{message['content'].strip()}\n\n"
                continue
            if message["role"] == "user" and pending:
                message = {"role": "user", "content": pending + message["content"]}
                pending = ""
            merged.append(message)
        if pending:
            merged.append({"role": "user", "content": pending.strip()})
        normalized = merged
    elif mode == "lead":
        system = [m["content"] for m in normalized if m["role"] == "system"]
        normalized = [m for m in normalized if m["role"] != "system"]
        if system:
            normalized.insert(0, {"role": "system", "content": "\n\n".join(system)})
    prompt = bos + entry.get("begin", "")
    for message in normalized:
        pattern = entry.get("turns", {}).get(message["role"], entry.get("turn"))
        label = entry.get("roles", {}).get(message["role"], message["role"])
        prompt += pattern.replace("{role}", label).replace("{content}", message["content"])
    prompt += entry["reply"]
    if thinking_off:
        if "thinkingOff" not in entry:
            raise ValueError("This chat format has no reviewed non-thinking reply")
        prompt += entry["thinkingOff"]
    return prompt


def llama_cpp_vision_arguments(name: str) -> tuple[str, ...]:
    """Return llama-mtmd-cli template arguments: its built-in renderer or an engine template."""

    entry = formats()[name]
    if entry.get("llamaCppBuiltin"):
        return ("--chat-template", entry["llamaCppBuiltin"], "--no-jinja")
    if (
        "image" not in entry
        or entry.get("turns")
        or entry.get("begin")
        or entry.get("system", "turn") != "turn"
    ):
        raise ValueError("This chat format has no reviewed llama.cpp vision renderer")
    role = "message['role']"
    for source, label in entry.get("roles", {}).items():
        role = f"'{label}' if message['role'] == '{source}' else {role}"
    content = "message['content'] | trim" if entry.get("trim") is True else "message['content']"
    turn = entry["turn"].replace("{role}", "{{ " + role + " }}")
    turn = turn.replace("{content}", "{{ " + content + " }}")
    # Engine-generated from reviewed markers; message text is template data, never code.
    return (
        "--jinja",
        "--chat-template",
        "{{- bos_token -}}"
        "{%- for message in messages -%}"
        f"{turn}"
        "{% endfor -%}"
        f"{{%- if add_generation_prompt -%}}{entry['reply']}{{% endif -%}}",
    )
