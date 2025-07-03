"""
System-related API endpoints
"""
import datetime
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from utils.logging_helper import get_logger
from utils.database import db
from utils.memory_queue import memory_queue

logger = get_logger(__name__)
router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health")
async def health_check():
    """System health check endpoint"""
    try:
        # Check database connectivity with a simple query instead of user lookup
        async with db.get_connection() as conn:
            await conn.fetchval("SELECT 1")
        
        # Check queue status
        queue_status = memory_queue.get_queue_status()
        
        return {
            "status": "healthy",
            "database": "connected",
            "queue": queue_status,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
        ) 