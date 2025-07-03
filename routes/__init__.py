"""
Routes package for Mercury API endpoints
"""
from .queue import router as queue_router
from .user import router as user_router
from .system import router as system_router

__all__ = ["queue_router", "user_router", "system_router"] 