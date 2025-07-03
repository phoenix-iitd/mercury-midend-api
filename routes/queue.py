"""
Queue-related API endpoints
"""
from uuid import uuid4
from time import perf_counter
from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Optional

from utils.logging_helper import create_correlation_logger, get_logger, PerformanceTimer
from utils.global_queue import global_queue
from utils.models import QueueRequest, QueueResponse, QueueStatusRequest, QueueStatusResponse, RevokeRequest
from .dependencies import validate_api_request

logger = get_logger(__name__)
router = APIRouter(prefix="/queue", tags=["queue"])


@router.post("/execute", response_model=QueueResponse)
async def execute_queue(request_data: QueueRequest, request: Request, api_key: str = Depends(validate_api_request)):
    """Execute a queue request"""
    correlation_id = str(uuid4())
    correlation_logger = create_correlation_logger(__name__, correlation_id)
    
    correlation_logger.info("Queue execution request received", 
                          user_id=request_data.user_id, 
                          group_count=len(request_data.data))
    
    start = perf_counter()
    
    try:
        with PerformanceTimer(logger, "execute_queue", correlation_id=correlation_id, user_id=request_data.user_id):
            # Convert groups to simple dict format
            groups = [{"id": group.id, "name": group.name} for group in request_data.data]
            
            # Submit to global queue
            result = await global_queue.submit_queue_request(
                user_id=request_data.user_id,
                device_id=request_data.device_id,
                message_content=request_data.message,
                groups=groups,
                file_path=request_data.filePath,
                file_name=request_data.fileName,
                correlation_id=correlation_id
            )
            
            exec_time = (perf_counter() - start) * 1000
            
            if result["success"]:
                correlation_logger.info("Queue request submitted successfully",
                                      total_messages=result["total_messages"],
                                      credits_deducted=result["credits_deducted"],
                                      execution_time_ms=exec_time)
                
                return QueueResponse(
                    success=True,
                    executionTime=exec_time,
                    summary={
                        "total": result["total_messages"],
                        "queued": result["total_messages"],
                        "failed": 0,
                        "credits_deducted": result["credits_deducted"]
                    },
                    correlation_id=correlation_id
                )
            else:
                correlation_logger.warning("Queue request failed", 
                                         error=result.get("error"),
                                         execution_time_ms=exec_time)
                
                raise HTTPException(
                    status_code=400, 
                    detail=result.get("error", "Queue request failed")
                )
                
    except Exception as e:
        exec_time = (perf_counter() - start) * 1000
        correlation_logger.error("Queue execution failed", 
                               error=str(e), 
                               execution_time_ms=exec_time)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/status", response_model=QueueStatusResponse)
async def get_queue_status(request_data: QueueStatusRequest, request: Request, api_key: str = Depends(validate_api_request)):
    """Get real-time status of a queue request"""
    correlation_logger = create_correlation_logger(__name__, request_data.correlation_id)
    
    try:
        status = await global_queue.get_queue_status(
            user_id=request_data.user_id,
            correlation_id=request_data.correlation_id
        )
        
        correlation_logger.debug("Queue status retrieved", status=status)
        
        return QueueStatusResponse(
            success=True,
            status=status
        )
        
    except Exception as e:
        correlation_logger.error("Failed to get queue status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/revoke")
async def revoke_queue(request_data: RevokeRequest, request: Request, api_key: str = Depends(validate_api_request)):
    """Revoke messages for a user"""
    correlation_id = str(uuid4())
    correlation_logger = create_correlation_logger(__name__, correlation_id)
    
    correlation_logger.info("Message revocation request received",
                          user_id=request_data.user_id,
                          message_text=request_data.message_text,
                          max_age_hours=request_data.max_age_hours)
    
    try:
        with PerformanceTimer(logger, "revoke_messages", correlation_id=correlation_id, user_id=request_data.user_id):
            result = await global_queue.revoke_messages(
                user_id=request_data.user_id,
                device_id=request_data.device_id,
                message_text=request_data.message_text,
                max_age_hours=request_data.max_age_hours
            )
            
            correlation_logger.info("Message revocation completed", result=result)
            return result
            
    except Exception as e:
        correlation_logger.error("Message revocation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) 