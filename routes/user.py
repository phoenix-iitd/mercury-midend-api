"""
User-related API endpoints
"""
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Request, Depends

from utils.logging_helper import create_correlation_logger, get_logger
from utils.database import db
from utils.models import CreditsResponse
from .dependencies import validate_api_request

logger = get_logger(__name__)
router = APIRouter(prefix="/user", tags=["user"])


@router.get("/{user_identifier}/credits", response_model=CreditsResponse)
async def get_user_credits(user_identifier: str, request: Request, api_key: str = Depends(validate_api_request)):
    """Get user credit information. Accepts username or UUID."""
    correlation_id = str(uuid4())
    correlation_logger = create_correlation_logger(__name__, correlation_id)
    
    try:
        credits = await db.get_user_credits(user_identifier)
        
        if credits:
            correlation_logger.info("Credits retrieved", user_identifier=user_identifier, credits=credits["left"])
            return CreditsResponse(
                success=True,
                credits=credits
            )
        else:
            correlation_logger.warning("User not found in credits table", user_identifier=user_identifier)
            raise HTTPException(status_code=404, detail="User not found")
            
    except HTTPException:
        raise
    except Exception as e:
        correlation_logger.error("Failed to get user credits", error=str(e), user_identifier=user_identifier)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_identifier}")
async def get_user_info(user_identifier: str, request: Request, api_key: str = Depends(validate_api_request)):
    """Get user information by username or UUID"""
    correlation_id = str(uuid4())
    correlation_logger = create_correlation_logger(__name__, correlation_id)
    
    try:
        # Try to get user info
        if db._is_uuid(user_identifier):
            user_info = await db.get_user_by_id(user_identifier)
        else:
            user_info = await db.get_user_by_username(user_identifier)
        
        if user_info:
            # Remove sensitive data
            user_info.pop('password_hash', None)
            correlation_logger.info("User info retrieved", user_identifier=user_identifier)
            return {
                "success": True,
                "user": user_info
            }
        else:
            correlation_logger.warning("User not found", user_identifier=user_identifier)
            raise HTTPException(status_code=404, detail="User not found")
            
    except HTTPException:
        raise
    except Exception as e:
        correlation_logger.error("Failed to get user info", error=str(e), user_identifier=user_identifier)
        raise HTTPException(status_code=500, detail=str(e)) 