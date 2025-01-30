from fastapi import FastAPI, HTTPException, Request, Depends, Security, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, AnyUrl, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Union
import httpx
import asyncio
import base64
import os
import time
import logging
import hmac
import hashlib
import json
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
import random
from contextlib import asynccontextmanager
from uuid import uuid4
from time import perf_counter
from functools import wraps
from fastapi.exceptions import RequestValidationError

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# Add development mode check
IS_DEV = os.getenv("ENVIRONMENT", "development").lower() == "development"

# Simplified logging configuration
logging.basicConfig(
    level=logging.DEBUG if IS_DEV else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add correlation ID to logger
class ContextFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = getattr(record, 'correlation_id', 'NO_CORR_ID')
        return True

logger.addFilter(ContextFilter())

# Simplified performance decorator
def log_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = perf_counter()
        try:
            result = await func(*args, **kwargs)
            logger.info(f"{func.__name__} completed in {(perf_counter() - start_time)*1000:.0f}ms")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {str(e)}")
            raise
    return wrapper

# Create connection pool
connection_pool = None

# Update client pool with async state management
@asynccontextmanager
async def get_client_pool():
    transport = httpx.AsyncHTTPTransport(
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        retries=3
    )
    
    async with httpx.AsyncClient(
        verify=False,
        timeout=httpx.Timeout(
            connect=2.0,    # Connection timeout
            read=5.0,       # Read timeout
            write=5.0,      # Write timeout
            pool=5.0        # Pool timeout
        ),
        transport=transport,
        http2=False,
        limits=httpx.Limits(max_connections=100)
    ) as client:
        yield client

# Initialize FastAPI app with lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources
    app.state.client_pool = httpx.AsyncClient(
        verify=True,
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        http2=True
    )
    yield
    # Cleanup
    await app.state.client_pool.aclose()

# Update FastAPI initialization
app = FastAPI(
    docs_url="/docs" if IS_DEV else None,
    redoc_url="/redoc" if IS_DEV else None,
    lifespan=lifespan
)

# Enhanced security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[host.strip() for host in os.getenv("ALLOWED_HOSTS", "").split(",") if host.strip()]
)

# Update rate limiting configuration
limiter = Limiter(key_func=get_remote_address)

# Conditional rate limiting decorator
def conditional_limit(limit_string: str):
    def decorator(func):
        if IS_DEV:
            return func
        return limiter.limit(limit_string)(func)
    return decorator

# Enhanced API Configuration with validation
API_CONFIG = {
    "base_url": os.getenv("API_BASE_URL"),
    "port": int(os.getenv("API_PORT", 443)),
    "auth": base64.b64encode(
        f"{os.getenv('API_USER')}:{os.getenv('API_PASS')}".encode()
    ).decode() if os.getenv('API_USER') and os.getenv('API_PASS') else None
}

SECRET_KEY = os.getenv("SECRET_KEY")

if not all([API_CONFIG["base_url"], API_CONFIG["auth"]]):
    raise ValueError("Missing required API configuration")

if not SECRET_KEY:
    raise ValueError("SECRET_KEY environment variable must be set")

# Enhanced security setup
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
# hmac_header = APIKeyHeader(name="X-HMAC-Signature", auto_error=True)

async def validate_request_timing(request: Request):
    """Prevent replay attacks by validating request timing"""
    if IS_DEV:
        return True
    
    timestamp = request.headers.get("X-Request-Time")
    if not timestamp:
        raise HTTPException(status_code=400, detail="Missing timestamp")
    
    try:
        request_time = int(timestamp)
        if abs(time.time() - request_time) > 300:  # 5 minute window
            raise HTTPException(status_code=400, detail="Request expired")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp")

def create_error_response(error_message: str) -> dict:
    """Utility function to create consistent error responses"""
    return {
        "success": False,
        "message": str(error_message)
    }

# Keep API key header but remove HMAC header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
# hmac_header = APIKeyHeader(name="X-HMAC-Signature", auto_error=True)

async def validate_api_request(request: Request):
    """Validate secret key and request timing"""
    secret_key = request.headers.get("X-API-Key")
    
    if not secret_key:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": "Secret key required"}
        )

    try:
        if not hmac.compare_digest(secret_key, SECRET_KEY):
            logger.warning("Invalid secret key")
            return JSONResponse(
                status_code=401,
                content={"success": False, "message": "Invalid secret key"}
            )

        if not IS_DEV:  # Skip timestamp validation in dev mode
            await validate_request_timing(request)
        
        return True
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": str(e)}
        )

# Comment out old HMAC validation function
"""
async def validate_hmac(request: Request, api_key: str = Security(api_key_header)):
    # ... existing HMAC validation code ...
"""

# Enhanced Models with Pydantic v2 validation
class Group(BaseModel):
    id: str = Field(min_length=10, max_length=50, pattern=r'^[0-9]+@[sg]\.(whatsapp\.net|us)$')
    name: str = Field(min_length=1, max_length=100)
    
    @field_validator('id')
    @classmethod
    def validate_whatsapp_id(cls, v: str) -> str:
        v = v.lower()
        parts = v.split('@')
        if len(parts) != 2:
            raise ValueError("Invalid WhatsApp ID format")
        
        prefix, suffix = parts
        
        if not prefix.isdigit():
            raise ValueError("ID prefix must be numeric")
            
        if suffix == "s.whatsapp.net":
            if len(prefix) < 10:
                raise ValueError("Invalid user ID format")
        elif suffix == "g.us":
            if len(prefix) < 15:
                raise ValueError("Invalid group ID format")
        else:
            raise ValueError("Invalid WhatsApp ID suffix")
            
        return v

class QueueRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    time: int
    data: List[Group]
    filePath: str = ""  # Changed from AnyUrl to str
    fileName: str = ""

    @field_validator('time')
    @classmethod
    def validate_time(cls, v: int) -> int:
        if abs(time.time() - v) > 300 and not IS_DEV:
            raise ValueError("Invalid timestamp")
        return v

    @model_validator(mode='after')
    def validate_file_fields(self) -> 'QueueRequest':
        filePath = self.filePath or ""
        fileName = self.fileName or ""
        
        if filePath == "" and fileName == "":
            return self
            
        if bool(filePath) != bool(fileName):
            raise ValueError("Both filePath and fileName must be provided together")
        
        if filePath and not filePath.startswith('https://'):
            raise ValueError("Only HTTPS URLs are allowed")
            
        return self

class QueueResponse(BaseModel):
    success: bool
    executionTime: float
    summary: Dict[str, int]

# Update the ErrorResponse model to be simpler
class ErrorResponse(BaseModel):
    success: bool = False
    message: str

# Optimized API request implementation
async def secure_api_request(client: httpx.AsyncClient, endpoint: str, data: dict, group_name: str) -> dict:
    """Secure API request implementation with request signing and retry logic"""
    api_url = f"http://{API_CONFIG['base_url']}:{API_CONFIG['port']}/{endpoint}"  # Use HTTP for internal network
    
    # Add random delay between 1 and 3 seconds
    delay = random.uniform(4.0, 6.0)
    
    headers = {
        "Authorization": f"Basic {API_CONFIG['auth']}",
        "Content-Type": "application/json",
        "X-Request-Time": str(int(time.time()))
    }

    try:
        response = await client.post(
            api_url,
            json=data,
            headers=headers,
            timeout=10.0,
            follow_redirects=True
        )
        response.raise_for_status()
        await asyncio.sleep(delay)
        logger.info(f"Message sent to {group_name} ({data['phone']}) after {delay:.1f}s delay")
        return response.json()
    except Exception as error:
        logger.error(f"Failed to send message to {group_name} ({data['phone']}): {str(error)}")
        raise

# Update the execute_queue endpoint
@app.post("/executeQueue", response_model=Union[QueueResponse, ErrorResponse])
@conditional_limit("5/minute")
@log_performance
async def execute_queue(
    request_data: QueueRequest,
    request: Request
):
    # Validate API key first
    validation_result = await validate_api_request(request)
    if isinstance(validation_result, JSONResponse):
        return validation_result

    # Rest of the execute_queue function remains the same
    start_time = perf_counter()
    
    try:
        batch_size = 5 if IS_DEV else 3  # Reduced batch size due to delays
        success_count = 0
        failed_count = 0
        
        async with get_client_pool() as client:
            # Process in larger chunks with concurrent execution
            chunks = [request_data.data[i:i + batch_size] 
                     for i in range(0, len(request_data.data), batch_size)]
            
            for chunk in chunks:
                tasks = []
                for group in chunk:
                    endpoint = "send/image" if len(request_data.filePath) > 0 else "send/message"
                    payload = {
                        "phone": group.id,
                        "message": request_data.message if not len(request_data.filePath) > 0 else None,
                        "caption": request_data.message if len(request_data.filePath) > 0 else None,
                        **({"image_url": request_data.filePath} if len(request_data.filePath) > 0 else {})
                    }
                    tasks.append(secure_api_request(client, endpoint, payload, group.name))
                
                # Execute batch concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                success_count += sum(1 for r in results if not isinstance(r, Exception))
                failed_count += sum(1 for r in results if isinstance(r, Exception))

                # Random delay between batches (0.5 to 1.5 seconds)
                await asyncio.sleep(random.uniform(0.5, 1.5))

        execution_time = (perf_counter() - start_time) * 1000
        
        return QueueResponse(
            success=failed_count == 0,
            executionTime=execution_time,
            summary={
                "total": len(request_data.data),
                "successful": success_count,
                "failed": failed_count
            }
        )

    except HTTPException as http_error:
        logger.error(f"HTTP error in queue execution: {str(http_error)}")
        return {"success": False, "message": str(http_error.detail)}
    except Exception as error:
        logger.error(f"Queue execution failed: {str(error)}")
        return {"success": False, "message": f"Internal server error: {str(error)}"}

# Add error handler for rate limit exceeded
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"success": False, "message": "Rate limit exceeded"}
    )

# Update FastAPI exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(exc.detail)
    )

# Update other exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            str(exc) if IS_DEV else "An unexpected error occurred"
        )
    )

# Update validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if errors:
        error = errors[0]
        try:
            # Better error message extraction for Pydantic v2
            error_message = error.get('msg', '') if isinstance(error, dict) else str(error)
            if isinstance(error_message, str) and 'none is not an allowed value' in error_message.lower():
                error_message = "Missing required field"
        except:
            error_message = str(error)
    else:
        error_message = "Validation error"

    return JSONResponse(
        status_code=422,
        content={"success": False, "message": error_message}
    )

# Enhanced security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.update({
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=()",
        "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    })
    return response

# Update request logging middleware with safe field names
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    correlation_id = str(uuid4())
    start_time = perf_counter()
    
    logger.info(
        "Request started",
        extra={
            'correlation_id': correlation_id,
            'request_method': request.method,
            'request_url': str(request.url),
            'client_ip': request.client.host,
            'request_headers': dict(request.headers)
        }
    )
    
    try:
        response = await call_next(request)
        execution_time = (perf_counter() - start_time) * 1000
        
        logger.info(
            "Request completed",
            extra={
                'correlation_id': correlation_id,
                'status_code': response.status_code,
                'execution_time_ms': execution_time
            }
        )
        
        return response
    except Exception as error:
        execution_time = (perf_counter() - start_time) * 1000
        logger.error(
            "Request failed",
            extra={
                'correlation_id': correlation_id,
                'execution_time_ms': execution_time,
                'error_class': error.__class__.__name__,
                'error_message': str(error)
            },
            exc_info=True
        )
        raise

if __name__ == "__main__":
    # Development settings with live reload
    is_dev = os.getenv("ENVIRONMENT", "development").lower() == "development"
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        ssl_keyfile=os.getenv("SSL_KEYFILE") if not IS_DEV else None,
        ssl_certfile=os.getenv("SSL_CERTFILE") if not IS_DEV else None,
        log_level="debug" if IS_DEV else "info",
        access_log=True,
        reload=IS_DEV,  # Enable auto-reload in development
        reload_dirs=["./"] if IS_DEV else None,  # Watch current directory
        reload_delay=0.25,  # Delay between reloads
        workers=1 if IS_DEV else None,  # Single worker in dev mode
        loop="uvloop"  # Use uvloop for better async performance
    )