"""
Shared dependencies for API routes
"""
import os
from uuid import uuid4
from time import perf_counter
from fastapi import HTTPException, Request, Depends
from fastapi.security import APIKeyHeader
from typing import Optional

from utils.logging_helper import create_correlation_logger
from utils.auth import AuthenticationManager

# Load authentication configuration
SECRET_KEY = os.getenv("SECRET_KEY")
max_failed_attempts = int(os.getenv("MAX_AUTH_ATTEMPTS", "10"))
auth_lockout_time = int(os.getenv("AUTH_LOCKOUT_MINUTES", "15")) * 60

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
auth_manager = AuthenticationManager()


async def validate_api_request(request: Request, api_key: Optional[str] = Depends(api_key_header)):
    """Enhanced API key validation with rate limiting and security logging"""
    correlation_id = str(uuid4())
    auth_logger = create_correlation_logger(__name__, correlation_id)
    
    client_id = auth_manager._get_client_id(request)
    key_hash = auth_manager._hash_key(api_key) if api_key else "none"
    
    # Log all authentication attempts
    auth_logger.info(
        "API authentication attempt",
        client_id=client_id,
        key_hash=key_hash,
        endpoint=request.url.path,
        method=request.method,
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    # Check if client is locked out
    if auth_manager._is_locked_out(client_id):
        auth_logger.warning(
            "API request blocked - client locked out",
            client_id=client_id,
            lockout_time_remaining=auth_lockout_time - (perf_counter() - auth_manager.lockout_times[client_id])
        )
        raise HTTPException(
            status_code=429, 
            detail="Too many failed authentication attempts. Please try again later."
        )
    
    # Validate API key
    if not api_key:
        auth_manager._record_failed_attempt(client_id)
        auth_logger.warning("API request failed - missing API key", client_id=client_id)
        raise HTTPException(status_code=401, detail="API key required")
    
    if not SECRET_KEY:
        auth_logger.error("Server misconfiguration - SECRET_KEY not set")
        raise HTTPException(status_code=503, detail="Server configuration error")
    
    if api_key != SECRET_KEY:
        auth_manager._record_failed_attempt(client_id)
        auth_logger.warning(
            "API request failed - invalid API key",
            client_id=client_id,
            key_hash=key_hash,
            failed_attempts_count=len(auth_manager.failed_attempts.get(client_id, []))
        )
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Successful authentication
    auth_manager._clear_failed_attempts(client_id)
    auth_logger.info(
        "API request authenticated successfully",
        client_id=client_id,
        key_hash=key_hash
    )
    
    return api_key 