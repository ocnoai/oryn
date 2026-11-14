from __future__ import annotations

from typing import Any


def protocol_error(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
    diagnostic_id: str | None = None,
    category: str = "protocol",
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "category": category,
        "message": message,
        "retryable": retryable,
    }
    if details is not None:
        error["details"] = details
    if diagnostic_id is not None:
        error["diagnosticId"] = diagnostic_id
    return error
