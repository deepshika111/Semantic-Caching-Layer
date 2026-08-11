from __future__ import annotations

import hashlib
import hmac

from fastapi import Header, HTTPException, Request


def admin_keys_match(provided: str, expected: str) -> bool:
    left = hashlib.sha256(provided.encode("utf-8")).digest()
    right = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(left, right)


async def require_admin(
    request: Request,
    x_admin_key: str = Header(default="", alias="X-Admin-Key"),
) -> None:
    expected = request.app.state.settings.admin_api_key
    if not expected:
        raise HTTPException(status_code=503, detail="Admin API is not configured")
    if not admin_keys_match(x_admin_key, expected):
        raise HTTPException(status_code=401, detail="Invalid admin key")
