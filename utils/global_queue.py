import asyncio
import random
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from uuid import uuid4
import httpx
import structlog

from .database import db
from .memory_queue import memory_queue, QueueMessage, MessageStatus

logger = structlog.get_logger(__name__)

class GlobalQueue:
    """
    Global queue that handles multiple traffic lanes and chokes them into one processing lane.
    Includes credit management and message processing with delays.
    """
    
    def __init__(self):
        self.processing_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Delay configurations
        self.initial_delay_range = (2, 3)
        self.inter_message_delay_range = (4, 6)
        self.revoke_delay_range = (1, 2)
        
        # API configuration
        self.api_config = {
            "base_url": "",
            "port": 0,
            "auth": "",
            "timeout": 1000.0
        }
        
    def configure_api(self, base_url: str, port: int, auth: str):
        """Configure API settings"""
        self.api_config.update({
            "base_url": base_url,
            "port": port,
            "auth": auth
        })
        
    async def start(self):
        """Start the global queue processing"""
        if self.running:
            return
            
        self.running = True
        self.processing_task = asyncio.create_task(self._process_messages())
        logger.info("Global queue started")
        
    async def stop(self):
        """Stop the global queue processing"""
        self.running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Global queue stopped")
        
    async def submit_queue_request(
        self,
        user_id: str,
        device_id: str,
        message_content: str,
        groups: List[Dict[str, str]],
        file_path: str = "",
        file_name: str = "",
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit a queue request with credit checking and message queuing.
        Returns correlation_id and initial status.
        """
        if not correlation_id:
            correlation_id = str(uuid4())
            
        logger.info("Processing queue request", user_id=user_id, correlation_id=correlation_id, group_count=len(groups))
        
        # Check user credits first
        credits_needed = len(groups)
        if not await db.deduct_credit(user_id, credits_needed):
            logger.warning("Insufficient credits", user_id=user_id, needed=credits_needed)
            return {
                "success": False,
                "error": "Insufficient credits",
                "correlation_id": correlation_id,
                "credits_needed": credits_needed
            }
        
        # Create queue messages for each group
        queue_messages = []
        for group_data in groups:
            # Get or create group in database
            group_id = await db.get_or_create_group(
                whatsapp_group_id=group_data["id"],
                name=group_data["name"]
            )
            
            # Create queue message
            queue_message = QueueMessage(
                group_id=group_id,
                group_name=group_data["name"],
                whatsapp_group_id=group_data["id"],
                sender_id=user_id,
                content=message_content,
                media_url=file_path if file_path else None,
                correlation_id=correlation_id,
                device_id=device_id,
                file_path=file_path,
                file_name=file_name
            )
            queue_messages.append(queue_message)
        
        # Add messages to memory queue
        message_ids = await memory_queue.enqueue_bulk(queue_messages)
        
        logger.info(
            "Queue request submitted", 
            user_id=user_id, 
            correlation_id=correlation_id,
            message_count=len(message_ids),
            credits_deducted=credits_needed
        )
        
        return {
            "success": True,
            "correlation_id": correlation_id,
            "message_ids": message_ids,
            "total_messages": len(message_ids),
            "credits_deducted": credits_needed
        }
        
    async def get_queue_status(self, user_id: str, correlation_id: str) -> Dict[str, Any]:
        """Get real-time status of a queue request"""
        return memory_queue.get_user_queue_status(user_id, correlation_id)
        
    async def _process_messages(self):
        """Main processing loop that handles the single-lane message processing"""
        logger.info("Global queue processing started")
        first_message = True
        
        while self.running:
            try:
                # Get next message from memory queue
                if memory_queue.pending_queue.empty():
                    await asyncio.sleep(0.1)  # Small delay when queue is empty
                    continue
                    
                message_id = await memory_queue.pending_queue.get()
                message = memory_queue.get_message(message_id)
                
                if not message:
                    memory_queue.pending_queue.task_done()
                    continue
                
                # Apply delays - initial delay for first message, inter-message delay for others
                if first_message:
                    delay = random.uniform(*self.initial_delay_range)
                    logger.debug("Initial delay applied", delay=f"{delay:.1f}s", correlation_id=message.correlation_id)
                    first_message = False
                else:
                    delay = random.uniform(*self.inter_message_delay_range)
                    logger.debug("Inter-message delay applied", delay=f"{delay:.1f}s", correlation_id=message.correlation_id)
                    
                await asyncio.sleep(delay)
                
                # Move message to processing state
                memory_queue.processing[message_id] = message
                
                # Process the message
                success = await self._send_message(message)
                
                # Update message status based on result
                if success:
                    # Save to database
                    await self._save_message_to_db(message)
                    memory_queue.update_message_status(
                        message_id, 
                        MessageStatus.SUCCESS,
                        message.whatsapp_message_id
                    )
                else:
                    memory_queue.update_message_status(message_id, MessageStatus.FAILED)
                
                memory_queue.pending_queue.task_done()
                
            except Exception as e:
                logger.error("Error in global queue processing", error=str(e))
                await asyncio.sleep(1)  # Delay before retry
                
        logger.info("Global queue processing ended")
        
    async def _send_message(self, message: QueueMessage) -> bool:
        """Send a message via the API"""
        try:
            endpoint = "send/image" if message.file_path else "send/message"
            url = f"http://{self.api_config['base_url']}:{self.api_config['port']}/{endpoint}"
            
            # Prepare payload
            if message.file_path:
                payload = {
                    "phone": message.whatsapp_group_id,
                    "caption": message.content,
                    "image_url": message.file_path
                }
            else:
                payload = {
                    "phone": message.whatsapp_group_id,
                    "message": message.content
                }
            
            headers = {
                "Authorization": f"Basic {self.api_config['auth']}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=self.api_config["timeout"]) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                # Extract WhatsApp message ID from response
                result = response.json()
                whatsapp_message_id = result.get("results", {}).get("message_id")
                message.whatsapp_message_id = whatsapp_message_id
                message.sent_at = datetime.now()
                
                logger.info(
                    "Message sent successfully",
                    message_id=message.id,
                    group_name=message.group_name,
                    whatsapp_id=message.whatsapp_group_id,
                    correlation_id=message.correlation_id
                )
                
                return True
                
        except Exception as e:
            logger.error(
                "Failed to send message",
                message_id=message.id,
                group_name=message.group_name,
                error=str(e),
                correlation_id=message.correlation_id
            )
            return False
            
    async def _save_message_to_db(self, message: QueueMessage):
        """Save successful message to database"""
        try:
            message_id = await db.create_message(
                group_id=message.group_id,
                sender_identifier=message.sender_id,
                content=message.content,
                media_url=message.media_url,
                whatsapp_message_id=message.whatsapp_message_id
            )
            
            # Update with success status and sent time
            await db.update_message_status(
                message_identifier=message_id,
                status="success",
                sent_at=message.sent_at
            )
            
            logger.debug(
                "Message saved to database",
                message_id=message_id,
                correlation_id=message.correlation_id
            )
            
        except Exception as e:
            logger.error(
                "Failed to save message to database",
                error=str(e),
                correlation_id=message.correlation_id
            )
            
    async def revoke_messages(
        self,
        user_id: str,
        device_id: str,
        message_text: str,
        max_age_hours: int = 8
    ) -> Dict[str, Any]:
        """Revoke messages for a user"""
        logger.info("Starting message revocation", user_id=user_id, message_text=message_text)
        
        # Get messages to revoke from database
        messages_to_revoke = await db.get_messages_for_revoke(
            user_identifier=user_id,
            device_id=device_id,
            content=message_text,
            hours_back=max_age_hours
        )
        
        if not messages_to_revoke:
            logger.info("No messages found for revocation", user_id=user_id)
            return {
                "success": True,
                "summary": {
                    "revoked": 0,
                    "failed": 0,
                    "skipped": 0,
                    "time_window": f"Last {max_age_hours} hours"
                }
            }
        
        revoked_count = 0
        failed_count = 0
        skipped_count = 0
        
        for message_data in messages_to_revoke:
            try:
                whatsapp_message_id = message_data.get("whatsapp_message_id")
                whatsapp_group_id = message_data.get("whatsapp_group_id")
                
                if not whatsapp_message_id or not whatsapp_group_id:
                    skipped_count += 1
                    continue
                
                # Send revoke request
                success = await self._revoke_single_message(whatsapp_message_id, whatsapp_group_id)
                
                if success:
                    revoked_count += 1
                    # Update message status in database
                    await db.update_message_status(
                        message_identifier=str(message_data["id"]),
                        status="failed"  # Mark as failed since it was revoked
                    )
                else:
                    failed_count += 1
                
                # Apply delay between revocations
                delay = random.uniform(*self.revoke_delay_range)
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error("Error revoking message", error=str(e), message_id=message_data.get("id"))
                failed_count += 1
        
        logger.info(
            "Message revocation completed",
            user_id=user_id,
            revoked=revoked_count,
            failed=failed_count,
            skipped=skipped_count
        )
        
        return {
            "success": failed_count == 0,
            "summary": {
                "revoked": revoked_count,
                "failed": failed_count,
                "skipped": skipped_count,
                "time_window": f"Last {max_age_hours} hours"
            }
        }
        
    async def _revoke_single_message(self, whatsapp_message_id: str, whatsapp_group_id: str) -> bool:
        """Revoke a single message"""
        try:
            url = f"http://{self.api_config['base_url']}:{self.api_config['port']}/message/{whatsapp_message_id}/revoke"
            payload = {"phone": whatsapp_group_id}
            headers = {
                "Authorization": f"Basic {self.api_config['auth']}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=100.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
            logger.debug("Message revoked successfully", message_id=whatsapp_message_id)
            return True
            
        except Exception as e:
            logger.error("Failed to revoke message", message_id=whatsapp_message_id, error=str(e))
            return False

# Global instance
global_queue = GlobalQueue()

async def init_global_queue(api_base_url: str, api_port: int, api_auth: str):
    """Initialize and start the global queue"""
    global_queue.configure_api(api_base_url, api_port, api_auth)
    await global_queue.start()
    
async def close_global_queue():
    """Stop the global queue"""
    await global_queue.stop() 