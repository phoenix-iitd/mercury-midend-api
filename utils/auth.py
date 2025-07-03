from fastapi import Request
import hashlib
from time import perf_counter
import os


import structlog

logger = structlog.get_logger(__name__)

max_failed_attempts = int(os.getenv("MAX_AUTH_ATTEMPTS", "10"))
auth_lockout_time = int(os.getenv("AUTH_LOCKOUT_MINUTES", "15")) * 60

class AuthenticationManager:
    def __init__(self):
        self.failed_attempts = {}
        self.lockout_times = {}
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Use X-Forwarded-For if behind proxy, otherwise remote IP
        client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        if not client_ip:
            client_ip = getattr(request.client, "host", "unknown")
        return client_ip
    
    def _hash_key(self, key: str) -> str:
        """Hash API key for secure logging"""
        if not key:
            return "empty"
        return hashlib.sha256(key.encode()).hexdigest()[:12]
    
    def _is_locked_out(self, client_id: str) -> bool:
        """Check if client is currently locked out"""
        if client_id not in self.lockout_times:
            return False
        
        time_since_lockout = perf_counter() - self.lockout_times[client_id]
        if time_since_lockout > auth_lockout_time:
            # Lockout expired, reset
            del self.lockout_times[client_id]
            self.failed_attempts.pop(client_id, None)
            return False
        
        return True
    
    def _record_failed_attempt(self, client_id: str):
        """Record a failed authentication attempt"""
        current_time = perf_counter()
        
        if client_id not in self.failed_attempts:
            self.failed_attempts[client_id] = []
        
        # Clean old attempts (older than 1 hour)
        hour_ago = current_time - 3600
        self.failed_attempts[client_id] = [
            attempt for attempt in self.failed_attempts[client_id] 
            if attempt > hour_ago
        ]
        
        # Add current attempt
        self.failed_attempts[client_id].append(current_time)
        
        # Check if should be locked out
        if len(self.failed_attempts[client_id]) >= max_failed_attempts:
            self.lockout_times[client_id] = current_time
            logger.warning(
                "Client locked out due to too many failed auth attempts",
                client_id=client_id,
                failed_attempts=len(self.failed_attempts[client_id])
            )
    
    def _clear_failed_attempts(self, client_id: str):
        """Clear failed attempts for successful authentication"""
        self.failed_attempts.pop(client_id, None)
        self.lockout_times.pop(client_id, None)

