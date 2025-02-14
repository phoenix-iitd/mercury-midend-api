import os, base64, asyncio, random, datetime
from uuid import uuid4
from time import perf_counter
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, model_validator
import uvicorn
import logging
from dotenv import load_dotenv

from utils.logging_helper import add_correlation_id
from utils.firebase import initialize_firebase
from utils.queue_helpers import log_message_firestore, create_rt_queue
from google.cloud.firestore_v1.base_query import FieldFilter

# Load env and setup config
load_dotenv()
IS_DEV = os.getenv("ENVIRONMENT", "development").lower() == "development"

# Initialize logger
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG if IS_DEV else logging.INFO)

# Load API config
API_CONFIG = {
    "base_url": os.getenv("API_BASE_URL"),
    "port": int(os.getenv("API_PORT")),
    "auth": base64.b64encode(f"{os.getenv('API_USER')}:{os.getenv('API_PASS')}".encode()).decode()
}
SECRET_KEY = os.getenv("SECRET_KEY")

# Initialize Firebase and get Firestore client
db_firestore = initialize_firebase()

# Create FastAPI instance once
app = FastAPI(docs_url="/docs" if IS_DEV else None)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[h.strip() for h in os.getenv("ALLOWED_HOSTS", "*").split(",") if h.strip()]
)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def validate_api_request(request: Request):
    key = request.headers.get("X-API-Key")
    if not key or key != SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key

# Pydantic Models
class Group(BaseModel):
    id: str = Field(..., pattern=r'^[0-9]+(-[0-9]+)?@[sg]\.(whatsapp\.net|us)$')
    name: str

class QueueRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    data: list[Group]
    filePath: str = ""
    fileName: str = ""
    device_id: str
    user_id: str

    @model_validator(mode="after")
    def check_file_fields(self):
        if bool(self.filePath) != bool(self.fileName):
            raise ValueError("Both filePath and fileName must be provided together")
        return self

class QueueResponse(BaseModel):
    success: bool
    executionTime: float
    summary: dict

class RevokeRequest(BaseModel):
    user_id: str
    device_id: str
    message_text: str
    date_str: str
    max_age_hours: int = Field(default=8, ge=1, le=24)

# Define delay ranges
INITIAL_DELAY_RANGE = (2, 3)
INTER_MESSAGE_DELAY_RANGE = (4, 6)
REVOKE_DELAY_RANGE = (1, 2)

# Helper: send API request
async def async_send_message(endpoint: str, data: dict, group_name: str) -> dict:
    url = f"http://{API_CONFIG['base_url']}:{API_CONFIG['port']}/{endpoint}"
    headers = {
        "Authorization": f"Basic {API_CONFIG['auth']}",
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient(timeout=1000.0) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()
        logger.info(add_correlation_id(f"Message sent to {group_name} ({data['phone']})", "sync", logging.INFO))
        return response.json()

# New helper to revoke a message asynchronously
async def revoke_message(message_id: str, phone: str, correlation_id: str) -> bool:
    revoke_url = f"http://{API_CONFIG['base_url']}:{API_CONFIG['port']}/message/{message_id}/revoke"
    payload = {"phone": phone}
    logger.debug(add_correlation_id(f"Sending revoke request to {revoke_url} with payload {payload}", correlation_id, logging.DEBUG))
    try:
        async with httpx.AsyncClient(timeout=100.0) as client:
            response = await client.post(
                revoke_url,
                json=payload,
                headers={
                    "Authorization": f"Basic {API_CONFIG['auth']}",
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
        logger.debug(add_correlation_id(f"Revoke request successful for message {message_id}", correlation_id, logging.DEBUG))
        return True
    except Exception as e:
        logger.error(add_correlation_id(f"Failed to revoke {message_id} for {phone}: {str(e)}", correlation_id, logging.ERROR))
        return False

# New helper to process message revocations.
async def process_revocations(request_data: RevokeRequest, correlation_id: str) -> tuple[int, int, int]:
    """
    Query Firestore for messages within the time window and revoke them.
    Returns a tuple of (revoked_count, failed_count, skipped_count).
    """
    docs = []
    now = datetime.datetime.now()
    current_time = int(now.timestamp() * 1000)
    cutoff_time = int((now - datetime.timedelta(hours=request_data.max_age_hours)).timestamp() * 1000)
    
    try:
        logs_ref = db_firestore.collection("log").document(request_data.date_str).collection(request_data.user_id)
        simple_docs = list(
            logs_ref.where(
                filter=FieldFilter("device_id", "==", request_data.device_id)
            ).stream()
        )
        for doc in simple_docs:
            data = doc.to_dict()
            if (data.get("message") == request_data.message_text and
                cutoff_time <= int(data.get("exactTime", "0")) <= current_time and data.get("status") != "REVOKED"):
                docs.append(doc)
        logger.debug(add_correlation_id(f"Found {len(docs)} messages within time window", correlation_id, logging.DEBUG))
    except Exception as e:
        logger.error(add_correlation_id(f"Failed to query Firestore: {str(e)}", correlation_id, logging.ERROR))
        raise HTTPException(status_code=500, detail="Failed to query messages")
    
    revoked_count = 0
    failed_count = 0
    skipped_count = 0

    for doc in docs:
        data = doc.to_dict()
        message_id = data.get("message_id")
        phone = data.get("toWhomID", [])[0] if data.get("toWhomID") else None
        if not message_id or not phone:
            logger.warning(add_correlation_id(f"Missing message_id or phone for doc {doc.id}", correlation_id, logging.WARNING))
            skipped_count += 1
            continue
        logger.debug(add_correlation_id(f"Revoking message {message_id} for phone {phone}", correlation_id, logging.DEBUG))
        if await revoke_message(message_id, phone, correlation_id):
            doc.reference.update({
                "status": "REVOKED",
                "revokedAt": datetime.datetime.now().strftime("%H:%M:%S %d %b, %Y")
            })
            revoked_count += 1
            logger.info(add_correlation_id(f"Revoked message {message_id} for {phone}", correlation_id, logging.INFO))
        else:
            failed_count += 1
        # Delay to mitigate rate limiting.
        delay = random.uniform(*REVOKE_DELAY_RANGE)
        logger.debug(add_correlation_id(f"Delaying {delay:.1f}s before next revocation", correlation_id, logging.DEBUG))
        await asyncio.sleep(delay)

    return revoked_count, failed_count, skipped_count

# Endpoints
@app.post("/executeQueue", response_model=QueueResponse)
async def execute_queue(request_data: QueueRequest, request: Request):
    validate_api_request(request)
    correlation_id = str(uuid4())
    logger.info(add_correlation_id("Starting queue execution", correlation_id, logging.INFO))
    start = perf_counter()
    success_count = 0
    failed_count = 0
    total = len(request_data.data)
    device_id = request_data.device_id
    user_id = request_data.user_id

    rt_queue_ref = create_rt_queue(str(int(datetime.datetime.now().timestamp()*1000)), request_data)

    for idx, group in enumerate(request_data.data, 1):
        try:
            if idx == 1:
                delay = random.uniform(*INITIAL_DELAY_RANGE)
                logger.debug(add_correlation_id(f"Initial delay: {delay:.1f}s", correlation_id, logging.DEBUG))
                await asyncio.sleep(delay)
            logger.info(add_correlation_id(f"Processing {idx}/{total} for group {group.name}", correlation_id, logging.INFO))
            endpoint = "send/image" if request_data.filePath else "send/message"
            payload = {
                "phone": group.id,
                **({"caption": request_data.message, "image_url": request_data.filePath} if request_data.filePath else {"message": request_data.message})
            }
            # Use the new async send function
            response = await async_send_message(endpoint, payload, group.name)
            message_id = response.get("results", {}).get("message_id")
            success_count += 1
            log_message_firestore(user_id, group, request_data.message, device_id, message_id,
                                  file_path=request_data.filePath, file_name=request_data.fileName)
            rt_queue_ref.child("data").child(str(idx-1)).delete()
        except Exception as e:
            failed_count += 1
            logger.error(add_correlation_id(f"Error for {group.name}: {str(e)}", correlation_id, logging.ERROR))
        if idx < total:
            delay = random.uniform(*INTER_MESSAGE_DELAY_RANGE)
            logger.debug(add_correlation_id(f"Delay between messages: {delay:.1f}s", correlation_id, logging.DEBUG))
            await asyncio.sleep(delay)

    exec_time = (perf_counter() - start) * 1000
    logger.info(add_correlation_id(f"Completed queue in {exec_time:.0f}ms | Success: {success_count}, Failed: {failed_count}", correlation_id, logging.INFO))
    rt_queue_ref.delete()
    return QueueResponse(success=(failed_count == 0), executionTime=exec_time,
                         summary={"total": total, "successful": success_count, "failed": failed_count})

# Modified revocation endpoint using the new helper.
@app.post("/revokeQueue")
async def revoke_queue(request_data: RevokeRequest, request: Request):
    validate_api_request(request)
    correlation_id = str(uuid4())
    logger.info(add_correlation_id(f"Revoking messages for {request_data.user_id}", correlation_id, logging.INFO))
    
    revoked_count, failed_count, skipped_count = await process_revocations(request_data, correlation_id)
    
    return {
        "success": failed_count == 0,
        "summary": {
            "revoked": revoked_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "time_window": f"Last {request_data.max_age_hours} hours"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(add_correlation_id(f"Unhandled exception: {str(exc)}", "NO_CORR_ID", logging.ERROR))
    return JSONResponse(status_code=500, content={"success": False, "message": "Internal server error"})

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "message": exc.detail})

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.getenv("MIDEND_BASE_URL", "127.0.0.1"),
        port=int(os.getenv("MIDEND_PORT", 8000)),
        reload=IS_DEV
    )