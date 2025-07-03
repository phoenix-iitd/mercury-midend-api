import os, base64, datetime
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from utils.logging_helper import configure_structlog, get_logger, create_correlation_logger
from utils.database import init_database, close_database
from utils.memory_queue import init_memory_queue, close_memory_queue
from utils.global_queue import init_global_queue, close_global_queue

from routes import queue_router, user_router, system_router

# Load env and setup config
load_dotenv()
IS_DEV = os.getenv("ENVIRONMENT", "development").lower() == "development"

# Configure structured logging
configure_structlog("development" if IS_DEV else "production")
logger = get_logger(__name__)

# Load API config
API_CONFIG = {
    "base_url": os.getenv("API_BASE_URL", "localhost"),
    "port": int(os.getenv("API_PORT", "8080")),
    "auth": base64.b64encode(
        f"{os.getenv('API_USER', 'admin')}:{os.getenv('API_PASS', 'password')}".encode()
    ).decode(),
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting application", environment="development" if IS_DEV else "production")
    try:
        await init_database()
        await init_memory_queue()
        await init_global_queue(
            api_base_url=API_CONFIG["base_url"],
            api_port=API_CONFIG["port"],
            api_auth=API_CONFIG["auth"]
        )
        logger.info("All systems initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize systems", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    try:
        await close_global_queue()
        await close_memory_queue()
        await close_database()
        logger.info("All systems shut down successfully")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))

# Create FastAPI instance with lifespan
app = FastAPI(
    docs_url="/docs" if IS_DEV else None,
    lifespan=lifespan
)

"""
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        h.strip() for h in os.getenv("ALLOWED_HOSTS", "*").split(",") if h.strip()
    ],
) 
"""

# Include the route modules
app.include_router(queue_router)
app.include_router(user_router)
app.include_router(system_router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    correlation_id = str(uuid4())
    correlation_logger = create_correlation_logger(__name__, correlation_id)
    
    correlation_logger.error("Unhandled exception", 
                           error=str(exc), 
                           path=str(request.url.path),
                           method=request.method)
    
    return JSONResponse(
        status_code=500, 
        content={
            "success": False, 
            "message": "Internal server error",
            "correlation_id": correlation_id
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    correlation_id = str(uuid4())
    correlation_logger = create_correlation_logger(__name__, correlation_id)
    
    correlation_logger.warning("HTTP exception", 
                             status_code=exc.status_code,
                             detail=exc.detail,
                             path=str(request.url.path),
                             method=request.method)
    
    return JSONResponse(
        status_code=exc.status_code, 
        content={
            "success": False, 
            "message": exc.detail,
            "correlation_id": correlation_id
        }
    )


if __name__ == "__main__":
    print(os.getenv("MIDEND_BASE_URL"))
    print(os.getenv("MIDEND_PORT"))
    uvicorn.run(
        "server:app",
        host=os.getenv("MIDEND_BASE_URL", "127.0.0.1"),
        port=int(os.getenv("MIDEND_PORT", 8000)),
        reload=IS_DEV,
    )
