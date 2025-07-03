import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
from datetime import datetime
import asyncpg
from asyncpg.pool import Pool
import structlog

logger = structlog.get_logger(__name__)

class Database:
    def __init__(self):
        self.pool: Optional[Pool] = None
        
    async def init_pool(self) -> None:
        """Initialize database connection pool"""
        dsn = self._get_database_url()
        try:
            logger.info("Initializing database connection pool", dsn=dsn)
            self.pool = await asyncpg.create_pool(
                dsn,
                min_size=2,
                max_size=10,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error("Failed to initialize database pool", error=str(e))
            logger.error("Database URL", dsn=dsn)
            raise
            
    def _get_database_url(self) -> str:
        """Get database URL from environment variables"""
        return os.getenv("DATABASE_URL", "")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def close_pool(self) -> None:
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
            
    # Helper functions to handle username/UUID resolution
    def _is_uuid(self, value: str) -> bool:
        """Check if string is a valid UUID"""
        try:
            UUID(value)
            return True
        except ValueError:
            return False
    
    async def _resolve_user_id(self, user_identifier: str) -> Optional[str]:
        """Resolve username or UUID to UUID string. Returns None if user not found."""
        if self._is_uuid(user_identifier):
            # Already a UUID, verify it exists
            async with self.get_connection() as conn:
                result = await conn.fetchval(
                    "SELECT id FROM users WHERE id = $1",
                    UUID(user_identifier)
                )
                return str(result) if result else None
        else:
            # It's a username, look up the UUID
            async with self.get_connection() as conn:
                result = await conn.fetchval(
                    "SELECT id FROM users WHERE username = $1",
                    user_identifier
                )
                return str(result) if result else None
    
    async def _resolve_message_id(self, message_identifier: str) -> Optional[str]:
        """Resolve message identifier to UUID string. Returns None if message not found."""
        if self._is_uuid(message_identifier):
            return message_identifier
        else:
            # Might be a WhatsApp message ID, look up the UUID
            async with self.get_connection() as conn:
                result = await conn.fetchval(
                    "SELECT id FROM messages WHERE whatsapp_message_id = $1",
                    message_identifier
                )
                return str(result) if result else None
            
    # User and Credit operations
    async def get_user_credits(self, user_identifier: str) -> Optional[Dict[str, Any]]:
        """Get user credit information. Accepts username or UUID."""
        user_id = await self._resolve_user_id(user_identifier)
        if not user_id:
            logger.warning("User not found", user_identifier=user_identifier)
            return None
            
        async with self.get_connection() as conn:
            result = await conn.fetchrow(
                "SELECT user_id, \"limit\", \"left\", updated_at FROM credits WHERE user_id = $1",
                UUID(user_id)
            )
            if result:
                return dict(result)
            return None
    
    async def deduct_credit(self, user_identifier: str, amount: int = 1) -> bool:
        """Deduct credits from user account. Accepts username or UUID. Returns True if successful, False if insufficient credits"""
        user_id = await self._resolve_user_id(user_identifier)
        if not user_id:
            logger.warning("User not found for credit deduction", user_identifier=user_identifier)
            return False
            
        async with self.get_connection() as conn:
            async with conn.transaction():
                # Get current credits with FOR UPDATE lock
                result = await conn.fetchrow(
                    "SELECT \"left\" FROM credits WHERE user_id = $1 FOR UPDATE",
                    UUID(user_id)
                )
                
                if not result:
                    logger.warning("User not found in credits table", user_id=user_id)
                    return False
                    
                current_credits = result["left"]
                if current_credits < amount:
                    logger.warning("Insufficient credits", user_id=user_id, current=current_credits, needed=amount)
                    return False
                
                # Deduct credits
                await conn.execute(
                    "UPDATE credits SET \"left\" = \"left\" - $1, updated_at = NOW() WHERE user_id = $2",
                    amount,
                    UUID(user_id)
                )
                
                logger.info("Credits deducted", user_id=user_id, amount=amount, remaining=current_credits-amount)
                return True
    
    # Group operations
    async def get_or_create_group(self, whatsapp_group_id: str, name: str) -> str:
        """Get existing group or create new one. Returns group UUID"""
        async with self.get_connection() as conn:
            # Try to get existing group
            result = await conn.fetchrow(
                "SELECT id FROM groups WHERE whatsapp_group_id = $1",
                whatsapp_group_id
            )
            
            if result:
                return str(result["id"])
            
            # Create new group
            group_id = uuid4()
            await conn.execute(
                "INSERT INTO groups (id, name, whatsapp_group_id) VALUES ($1, $2, $3)",
                group_id,
                name,
                whatsapp_group_id
            )
            
            logger.info("Created new group", group_id=str(group_id), name=name, whatsapp_id=whatsapp_group_id)
            return str(group_id)
    
    # Message operations
    async def create_message(
        self,
        group_id: str,
        sender_identifier: str,  # Can be username or UUID
        content: str,
        media_url: Optional[str] = None,
        whatsapp_message_id: Optional[str] = None
    ) -> str:
        """Create a new message. Returns message UUID"""
        # Resolve sender to UUID
        sender_id = await self._resolve_user_id(sender_identifier)
        if not sender_id:
            raise ValueError(f"User not found: {sender_identifier}")
            
        message_id = uuid4()
        
        async with self.get_connection() as conn:
            await conn.execute(
                """INSERT INTO messages 
                   (id, whatsapp_message_id, group_id, sender_id, content, media_url, status) 
                   VALUES ($1, $2, $3, $4, $5, $6, 'pending')""",
                message_id,
                whatsapp_message_id,
                UUID(group_id),
                UUID(sender_id),
                content,
                media_url
            )
        
        logger.info("Message created", message_id=str(message_id), group_id=group_id, sender_id=sender_id)
        return str(message_id)
    
    async def update_message_status(
        self,
        message_identifier: str,  # Can be message UUID or WhatsApp message ID
        status: str,
        whatsapp_message_id: Optional[str] = None,
        sent_at: Optional[datetime] = None
    ) -> None:
        """Update message status"""
        # Resolve message ID to UUID
        message_id = await self._resolve_message_id(message_identifier)
        if not message_id:
            raise ValueError(f"Message not found: {message_identifier}")
            
        async with self.get_connection() as conn:
            query_parts = ["UPDATE messages SET status = $1, updated_at = NOW()"]
            params: List[Any] = [status]
            param_count = 1
            
            if whatsapp_message_id:
                param_count += 1
                query_parts.append(f"whatsapp_message_id = ${param_count}")
                params.append(whatsapp_message_id)
            
            if sent_at:
                param_count += 1
                query_parts.append(f"sent_at = ${param_count}")
                params.append(sent_at)
            
            param_count += 1
            query = " SET ".join(query_parts) + f" WHERE id = ${param_count}"
            params.append(UUID(message_id))
            
            await conn.execute(query, *params)
        
        logger.info("Message status updated", message_id=message_id, status=status)
    
    async def get_messages_for_revoke(
        self,
        user_identifier: str,  # Can be username or UUID
        device_id: str,
        content: str,
        hours_back: int = 8
    ) -> List[Dict[str, Any]]:
        """Get messages that can be revoked"""
        # Resolve user to UUID
        user_id = await self._resolve_user_id(user_identifier)
        if not user_id:
            logger.warning("User not found for message revocation", user_identifier=user_identifier)
            return []
            
        cutoff_time = datetime.now().timestamp() - (hours_back * 3600)
        
        async with self.get_connection() as conn:
            results = await conn.fetch(
                """SELECT m.id, m.whatsapp_message_id, g.whatsapp_group_id 
                   FROM messages m 
                   JOIN groups g ON m.group_id = g.id 
                   WHERE m.sender_id = $1 AND m.content = $2 
                   AND m.status = 'success' 
                   AND EXTRACT(EPOCH FROM m.sent_at) > $3""",
                UUID(user_id),
                content,
                cutoff_time
            )
            
            return [dict(row) for row in results]
    
    # Additional helper methods
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user information by username"""
        async with self.get_connection() as conn:
            result = await conn.fetchrow(
                "SELECT id, username, phone_no, device_flag, created_at FROM users WHERE username = $1",
                username
            )
            if result:
                return dict(result)
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information by UUID"""
        try:
            uuid_obj = UUID(user_id)
        except ValueError:
            return None
            
        async with self.get_connection() as conn:
            result = await conn.fetchrow(
                "SELECT id, username, phone_no, device_flag, created_at FROM users WHERE id = $1",
                uuid_obj
            )
            if result:
                return dict(result)
            return None

# Global database instance
db = Database()

async def init_database():
    """Initialize database connection"""
    await db.init_pool()

async def close_database():
    """Close database connection"""
    await db.close_pool() 