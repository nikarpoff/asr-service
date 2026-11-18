from secrets import compare_digest
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import get_settings, Settings

INVALID_TOKEN = "Invalid token"

security = HTTPBearer(auto_error=False)


def get_auth_config(settings: Settings = Depends(get_settings)) -> Optional[str]:
    """Auth token from the config; None when auth is off."""
    return settings.AUTH_TOKEN


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_token: Optional[str] = Depends(get_auth_config),
) -> str:
    """Bearer token check: 401 without a token or with a wrong one."""
    # If auth is disabled, allow access
    if auth_token is None:
        return "anonymous"

    # Constant-time comparison: no timing hints about the token. Bytes, since
    # compare_digest refuses str with non-ASCII characters
    if credentials is None or not compare_digest(credentials.credentials.encode(), auth_token.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=INVALID_TOKEN,
            headers={"WWW-Authenticate": "Bearer"},
        )

    return "user"
