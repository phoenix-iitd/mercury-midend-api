import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)

class MessageStatus(Enum):
    PENDING = "pending"
    SUCCESS = "success" 
    FAILED = "failed"

@dataclass
class QueueMessage:
    """In-memory representation of a message similar to DB table structure"""
    id: str = field(default_factory=lambda: str(uuid4()))
    whatsapp_message_id: Optional[str] = None
    group_id: str = ""
    group_name: str = ""
    whatsapp_group_id: str = ""
    sender_id: str = ""
    content: str = ""
    media_url: Optional[str] = None
    status: MessageStatus = MessageStatus.PENDING
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Additional fields for queue management
    retry_count: int = 0
    correlation_id: str = ""
    device_id: str = ""
    file_path: str = ""
    file_name: str = ""

@dataclass
class QueueStats:
    """Statistics for queue monitoring"""
    total_messages: int = 0
    pending_messages: int = 0
    successful_messages: int = 0
    failed_messages: int = 0
    queue_size: int = 0
    processing_rate: float = 0.0
    
class MemoryQueue:
    """In-memory queue system that handles message processing with real-time updates"""
    
    def __init__(self):
        self.messages: Dict[str, QueueMessage] = {}
        self.pending_queue: asyncio.Queue = asyncio.Queue()
        self.processing: Dict[str, QueueMessage] = {}
        self.completed: Dict[str, QueueMessage] = {}
        self.failed: Dict[str, QueueMessage] = {}
        
        # Stats tracking
        self.stats = QueueStats()
        self._last_stats_update = datetime.now()
        self._processed_count_since_last = 0
        
        # Callbacks for real-time updates
        self.status_callbacks: List[Callable[[QueueMessage], None]] = []
        
        # Queue processing task
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start_processing(self):
        """Start the queue processing task"""
        if self._running:
            return
            
        self._running = True
        self._processing_task = asyncio.create_task(self._process_queue())
        logger.info("Memory queue processing started")
        
    async def stop_processing(self):
        """Stop the queue processing task"""
        self._running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Memory queue processing stopped")
        
    def add_status_callback(self, callback: Callable[[QueueMessage], None]):
        """Add a callback for status updates"""
        self.status_callbacks.append(callback)
        
    async def enqueue_message(self, message: QueueMessage) -> str:
        """Add a message to the queue"""
        self.messages[message.id] = message
        await self.pending_queue.put(message.id)
        
        self._update_stats()
        logger.info("Message enqueued", message_id=message.id, correlation_id=message.correlation_id)
        
        return message.id
        
    async def enqueue_bulk(self, messages: List[QueueMessage]) -> List[str]:
        """Add multiple messages to the queue"""
        message_ids = []
        for message in messages:
            self.messages[message.id] = message
            await self.pending_queue.put(message.id)
            message_ids.append(message.id)
            
        self._update_stats()
        logger.info("Bulk messages enqueued", count=len(messages))
        
        return message_ids
        
    def get_message(self, message_id: str) -> Optional[QueueMessage]:
        """Get a message by ID"""
        return self.messages.get(message_id)
        
    def update_message_status(
        self, 
        message_id: str, 
        status: MessageStatus,
        whatsapp_message_id: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update message status and trigger callbacks"""
        message = self.messages.get(message_id)
        if not message:
            return False
            
        old_status = message.status
        message.status = status
        message.updated_at = datetime.now()
        
        if whatsapp_message_id:
            message.whatsapp_message_id = whatsapp_message_id
            
        if status == MessageStatus.SUCCESS:
            message.sent_at = datetime.now()
            self.processing.pop(message_id, None)
            self.completed[message_id] = message
            
        elif status == MessageStatus.FAILED:
            message.retry_count += 1
            self.processing.pop(message_id, None)
            self.failed[message_id] = message
            
        # Update stats
        self._update_stats()
        
        # Notify callbacks
        for callback in self.status_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.error("Status callback failed", error=str(e), message_id=message_id)
                
        logger.info(
            "Message status updated", 
            message_id=message_id, 
            old_status=old_status.value,
            new_status=status.value
        )
        
        return True
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "queue_size": self.pending_queue.qsize(),
            "processing_count": len(self.processing),
            "completed_count": len(self.completed),
            "failed_count": len(self.failed),
            "total_messages": len(self.messages),
            "stats": {
                "total": self.stats.total_messages,
                "pending": self.stats.pending_messages,
                "successful": self.stats.successful_messages,
                "failed": self.stats.failed_messages,
                "processing_rate": self.stats.processing_rate
            }
        }
        
    def get_messages_by_correlation(self, correlation_id: str) -> List[QueueMessage]:
        """Get all messages for a correlation ID"""
        return [
            msg for msg in self.messages.values() 
            if msg.correlation_id == correlation_id
        ]
        
    def get_user_queue_status(self, user_id: str, correlation_id: str) -> Dict[str, Any]:
        """Get queue status for a specific user request"""
        user_messages = [
            msg for msg in self.messages.values()
            if msg.sender_id == user_id and msg.correlation_id == correlation_id
        ]
        
        if not user_messages:
            return {"error": "No messages found for this request"}
            
        total = len(user_messages)
        pending = len([m for m in user_messages if m.status == MessageStatus.PENDING])
        processing = len([m for m in user_messages if m.id in self.processing])
        successful = len([m for m in user_messages if m.status == MessageStatus.SUCCESS])
        failed = len([m for m in user_messages if m.status == MessageStatus.FAILED])
        
        return {
            "correlation_id": correlation_id,
            "total": total,
            "pending": pending,
            "processing": processing,
            "successful": successful,
            "failed": failed,
            "progress_percentage": ((successful + failed) / total * 100) if total > 0 else 0,
            "messages": [
                {
                    "id": msg.id,
                    "group_name": msg.group_name,
                    "status": msg.status.value,
                    "sent_at": msg.sent_at.isoformat() if msg.sent_at else None,
                    "retry_count": msg.retry_count
                }
                for msg in user_messages
            ]
        }
        
    def cleanup_old_messages(self, hours: int = 24):
        """Clean up old completed/failed messages"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        to_remove = []
        for message_id, message in self.messages.items():
            if (message.status in [MessageStatus.SUCCESS, MessageStatus.FAILED] and
                message.updated_at.timestamp() < cutoff_time):
                to_remove.append(message_id)
                
        for message_id in to_remove:
            self.messages.pop(message_id, None)
            self.completed.pop(message_id, None)
            self.failed.pop(message_id, None)
            
        if to_remove:
            logger.info("Cleaned up old messages", count=len(to_remove))
            
        self._update_stats()
        
    def _update_stats(self):
        """Update internal statistics"""
        self.stats.total_messages = len(self.messages)
        self.stats.pending_messages = self.pending_queue.qsize()
        self.stats.successful_messages = len(self.completed)
        self.stats.failed_messages = len(self.failed)
        self.stats.queue_size = self.pending_queue.qsize()
        
        # Calculate processing rate
        now = datetime.now()
        time_diff = (now - self._last_stats_update).total_seconds()
        if time_diff >= 60:  # Update rate every minute
            self.stats.processing_rate = self._processed_count_since_last / time_diff
            self._last_stats_update = now
            self._processed_count_since_last = 0
            
    async def _process_queue(self):
        """Internal queue processing loop - this will be customized per use case"""
        logger.info("Queue processing loop started")
        
        while self._running:
            try:
                # Wait for a message with timeout to allow checking _running flag
                message_id = await asyncio.wait_for(
                    self.pending_queue.get(), 
                    timeout=1.0
                )
                
                message = self.messages.get(message_id)
                if not message:
                    continue
                    
                # Move to processing
                self.processing[message_id] = message
                message.status = MessageStatus.PENDING  # Mark as being processed
                
                # This is where external processing would happen
                # For now, we just mark as processed
                logger.debug("Processing message", message_id=message_id)
                
                # Mark task as done
                self.pending_queue.task_done()
                self._processed_count_since_last += 1
                
            except asyncio.TimeoutError:
                # Continue loop to check _running flag
                continue
            except Exception as e:
                logger.error("Error in queue processing", error=str(e))
                
        logger.info("Queue processing loop ended")

# Global memory queue instance
memory_queue = MemoryQueue()

async def init_memory_queue():
    """Initialize and start the memory queue"""
    await memory_queue.start_processing()
    
async def close_memory_queue():
    """Stop and cleanup the memory queue"""
    await memory_queue.stop_processing() 