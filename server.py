import os, time, random, base64
from uuid import uuid4
from time import perf_counter
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, model_validator
import uvicorn
from dotenv import load_dotenv
import datetime
import firebase_admin
from firebase_admin import credentials, firestore, db
from google.cloud.firestore_v1.base_query import FieldFilter, BaseCompositeFilter
from typing import Optional
import logging

# Load environment variables and setup initial config
load_dotenv()
IS_DEV = os.getenv("ENVIRONMENT", "development").lower() == "development"

# Updated helper to prepend colored correlation id to messages based on log level
def add_correlation_id(msg: str, correlation_id: str = "NO_CORR_ID", level: int = logging.INFO) -> str:
    if level == logging.DEBUG:
        color = "\033[90m"  # gray
    elif level == logging.INFO:
        color = "\033[32m"  # green
    elif level == logging.WARNING:
        color = "\033[38;5;208m"  # orange
    elif level == logging.ERROR:
        color = "\033[31m"  # red
    else:
        color = ""
    reset = "\033[0m"
    return f"{color}[{correlation_id}]{reset} {msg}"

# Initialize FastAPI and get Uvicorn logger
app = FastAPI(title='Mercury Midend API', docs_url="/docs" if IS_DEV else None)
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG if IS_DEV else logging.INFO)

# Load environment variables from .env and verify required ones
API_CONFIG = {
    "base_url": os.getenv("API_BASE_URL"),
    "port": int(os.getenv("API_PORT")),
    "auth": base64.b64encode(f"{os.getenv('API_USER')}:{os.getenv('API_PASS')}".encode()).decode()
}
SECRET_KEY = os.getenv("SECRET_KEY")

# ---- Update Firebase Initialization for Realtime DB ----
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
if not project_id:
    raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")
database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise ValueError("DATABASE_URL environment variable is required for realtime database")
cred_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not cred_json:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required for realtime database")


def check_firebase_auth():
    try:
        # Try to access Firestore to verify credentials
        db_firestore = firestore.client()
        db_firestore.collection('_check_auth').limit(1).get()
        # Try to access Realtime Database
        db.reference('/_check_auth').get()
        logger.info(add_correlation_id("Firebase authentication successful", "firebase_init", logging.INFO))
        return True
    except Exception as e:
        logger.error(add_correlation_id(f"Firebase authentication failed: {str(e)}", "firebase_init", logging.ERROR))
        raise ValueError("Firebase authentication failed. Please check your credentials.")

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_json)
    firebase_admin.initialize_app(cred, {'projectId': project_id, 'databaseURL': database_url})
    check_firebase_auth()  # Verify authentication after initialization

db_firestore = firestore.client()
# Rename Firestore client variable to db_firestore for clarity
# ---- End Firebase Initialization ----


# Initialize FastAPI with trusted hosts; allows auto-reload in dev from command-line
app = FastAPI(docs_url="/docs" if IS_DEV else None)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[h.strip() for h in os.getenv("ALLOWED_HOSTS", "*").split(",") if h.strip()]
)

# Security: simple API key check from header "X-API-Key"
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
    filePath: str = ""  # if provided, sends image; otherwise sends text
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
    max_age_hours: Optional[int] = Field(default=8, ge=1, le=24)  # WhatsApp generally allows ~8 hours

# Helper: send API request (synchronous)
def secure_api_request(endpoint: str, data: dict, group_name: str) -> dict:
    url = f"http://{API_CONFIG['base_url']}:{API_CONFIG['port']}/{endpoint}"
    headers = {
        "Authorization": f"Basic {API_CONFIG['auth']}",
        "Content-Type": "application/json"
    }
    try:
        response = httpx.post(url, json=data, headers=headers, timeout=10.0)
        response.raise_for_status()
        logger.info(add_correlation_id(f"Message sent to {group_name} ({data['phone']})", "sync", logging.INFO))
        return response.json()
    except Exception as error:
        logger.error(add_correlation_id(f"Failed to send to {group_name}: {str(error)}", "sync", logging.ERROR))
        raise

# ---- New Helper: Log message in Firestore under structured "data" ----
def log_message_firestore(user_id: str, group: Group, msg_text: str, device_id: str, message_id: str, file_path: str = None, file_name: str = None):
    if not user_id:
        user_id = "unknown"   # Prevent empty document id
    now = datetime.datetime.now()
    date_str = now.strftime("%d %b, %Y")  # e.g. "09 Feb, 2025"
    time_str = now.strftime("%H:%M:%S %d %b, %Y")  # e.g. "11:44:24 09 Feb, 2025"
    message_data = {
        "exactTime": str(int(now.timestamp()*1000)),
        "message": msg_text,
        "user_id": user_id,
        "time": time_str,
        "device_id": device_id,
        "message_id": message_id,  # Store message_id from API response
        "toWhomID": [group.id],  # Store ID instead of name for revocation
        "toWhom": [group.name],
        **({"filePath": file_path, "fileName": file_name} if file_path else {})
    }
    db_firestore.collection("log")\
        .document(date_str)\
        .collection(user_id)\
        .add(message_data)
# ---- End New Helper ----

# ---- New Helper: Update Realtime Database Queue ----
def create_rt_queue(queue_id: str, request_data: QueueRequest):
    now = datetime.datetime.now()
    # Store entire queue in realtime database
    rt_ref = db.reference("queue").child(queue_id)
    rt_ref.set({
        "device_id": request_data.device_id,
        "user_id": request_data.user_id,
        "message": request_data.message,
        "filePath": request_data.filePath,
        "fileName": request_data.fileName,
        "time": now.strftime("%H:%M:%S %d %b, %Y"),
        "exactTime": str(int(now.timestamp()*1000)),
        "data": [{"id": grp.id, "name": grp.name} for grp in request_data.data]
    })
    return rt_ref
# ---- End Realtime Queue Helper ----

# executeQueue endpoint: synchronous processing with delays, logging, and Firestore tracking
@app.post("/executeQueue", response_model=QueueResponse)
async def execute_queue(request_data: QueueRequest, request: Request):
    # Validate API key
    validate_api_request(request)
    correlation_id = str(uuid4())
    logger.info(add_correlation_id("Starting queue execution", correlation_id, logging.INFO))
    start = perf_counter()
    success_count = 0
    failed_count = 0
    total = len(request_data.data)
    device_id = request_data.device_id
    user_id = request_data.user_id
    current_time = str(int(datetime.datetime.now().timestamp() * 1000))

    # ---- Create realtime database queue ----
    rt_queue_ref = create_rt_queue(current_time, request_data)
    # ---- End realtime queue creation ----

    for idx, group in enumerate(request_data.data, 1):
        try:
            if idx == 1:
                delay = random.uniform(2, 3)
                logger.debug(add_correlation_id(f"Initial delay: {delay:.1f}s", correlation_id, logging.DEBUG))
                time.sleep(delay)
            logger.info(add_correlation_id(f"Processing {idx}/{total} for group {group.name}", correlation_id, logging.INFO))
            endpoint = "send/image" if request_data.filePath else "send/message"
            payload = {
                "phone": group.id,
                ** ({"caption": request_data.message, "image_url": request_data.filePath} if request_data.filePath else {"message": request_data.message}),
            }
            response = secure_api_request(endpoint, payload, group.name)
            message_id = response.get("results", {}).get("message_id")
            success_count += 1
            log_message_firestore(user_id, group, request_data.message, device_id, message_id, file_path=request_data.filePath, file_name=request_data.fileName)
            # Remove individual message from realtime db queue (using list index as key)
            rt_queue_ref.child("data").child(str(idx-1)).delete()
        except Exception as e:
            failed_count += 1
            logger.error(add_correlation_id(f"Error for {group.name}: {str(e)}", correlation_id, logging.ERROR))
        if idx < total:
            delay = random.uniform(4, 6)
            logger.debug(add_correlation_id(f"Delay between messages: {delay:.1f}s", correlation_id, logging.DEBUG))
            time.sleep(delay)

    exec_time = (perf_counter() - start) * 1000
    logger.info(add_correlation_id(f"Completed queue in {exec_time:.0f}ms | Success: {success_count}, Failed: {failed_count}", correlation_id, logging.INFO))
    
    # ---- Remove realtime database queue node completely if processing is done ----
    rt_queue_ref.delete()
    # ---- End realtime removal ----

    return QueueResponse(success=(failed_count == 0), executionTime=exec_time, summary={"total": total, "successful": success_count, "failed": failed_count})

# New revokeQueue endpoint
@app.post("/revokeQueue")
async def revoke_queue(request_data: RevokeRequest, request: Request):
    validate_api_request(request)
    correlation_id = str(uuid4())
    logger.info(add_correlation_id(f"Revoking messages for {request_data.user_id}", correlation_id, logging.INFO))
    
    # Calculate time window for revocation
    now = datetime.datetime.now()
    current_time = int(now.timestamp() * 1000)
    cutoff_time = int((now - datetime.timedelta(hours=request_data.max_age_hours)).timestamp() * 1000)
    
    logger.debug(add_correlation_id(f"Time window: {cutoff_time} to {current_time}", correlation_id, logging.DEBUG))
    logger.debug(add_correlation_id(f"Search criteria: device_id={request_data.device_id}, message={request_data.message_text}", correlation_id, logging.DEBUG))
    
    try:
        # First try with compound query (requires index)
        logs_ref = db_firestore.collection("log")\
            .document(request_data.date_str)\
            .collection(request_data.user_id)
            
        try:
            # Using only a simple query to filter results by device_id
            simple_docs = list(logs_ref.where(filter=FieldFilter("device_id", "==", request_data.device_id)).stream())
            logger.debug(add_correlation_id(f"Simple query found {len(simple_docs)} documents", correlation_id, logging.DEBUG))
            
            docs = []
            for doc in simple_docs:
                data = doc.to_dict()
                if (data.get("message") == request_data.message_text and
                    cutoff_time <= int(data.get("exactTime", "0")) <= current_time and data.get("status") != "REVOKED"):
                    docs.append(doc)
                    logger.debug(add_correlation_id(f"Matched document {doc.id}", correlation_id, logging.DEBUG))
                else:
                    logger.debug(add_correlation_id(f"Filtered out document {doc.id}", correlation_id, logging.DEBUG))
        except Exception as e:
            logger.error(add_correlation_id(f"Failed to execute simple query: {str(e)}", correlation_id, logging.ERROR))
            raise
    
    except Exception as e:
        logger.error(add_correlation_id(f"Failed to query Firestore: {str(e)}", correlation_id, logging.ERROR))
        raise HTTPException(status_code=500, detail="Failed to query messages")
    
    revoked_count = 0
    failed_count = 0
    skipped_count = 0
    
    try:
        logger.debug(add_correlation_id(f"Found {len(docs)} messages within time window", correlation_id, logging.DEBUG))
        
        for doc in docs:
            data = doc.to_dict()
            message_id = data.get("message_id")
            phone = data.get("toWhomID", [])[0] if data.get("toWhomID") else None
            
            if not message_id or not phone:
                logger.warning(add_correlation_id(f"Missing message_id or phone for doc {doc.id}", correlation_id, logging.WARNING))
                skipped_count += 1
                continue
            
            # Rest of revocation logic remains the same
            logger.debug(add_correlation_id(f"Processing revocation for message_id: {message_id}, phone: {phone}", correlation_id, logging.DEBUG))
            
            revoke_url = f"http://{API_CONFIG['base_url']}:{API_CONFIG['port']}/message/{message_id}/revoke"
            payload = {"phone": phone}
            
            logger.debug(add_correlation_id(f"Sending revoke request to {revoke_url} with payload {payload}", correlation_id, logging.DEBUG))
            try:
                response = httpx.post(
                    revoke_url, 
                    json=payload,
                    headers={
                        "Authorization": f"Basic {API_CONFIG['auth']}",
                        "Content-Type": "application/json"
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                logger.debug(add_correlation_id(f"Revoke request successful for message {message_id}", correlation_id, logging.DEBUG))
                doc.reference.update({
                    "status": "REVOKED",
                    "revokedAt": datetime.datetime.now().strftime("%H:%M:%S %d %b, %Y")
                })
                logger.debug(add_correlation_id(f"Updated document {doc.id} with revoked status", correlation_id, logging.DEBUG))
                revoked_count += 1
                logger.info(add_correlation_id(f"Revoked message {message_id} for {phone}", correlation_id, logging.INFO))
            except Exception as e:
                failed_count += 1
                logger.error(add_correlation_id(f"Failed to revoke {message_id} for {phone}: {str(e)}", correlation_id, logging.ERROR))
    except Exception as e:
        logger.error(add_correlation_id(f"Failed to query Firestore: {str(e)}", correlation_id, logging.ERROR))
        raise HTTPException(status_code=500, detail="Failed to query messages")
    
    return {
        "success": failed_count == 0,
        "summary": {
            "revoked": revoked_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "time_window": f"Last {request_data.max_age_hours} hours"
        }
    }

# Global exception handler for unexpected errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(add_correlation_id(f"Unhandled exception: {str(exc)}", "NO_CORR_ID", logging.ERROR))
    return JSONResponse(status_code=500, content={"success": False, "message": "Internal server error"})

# ---- HTTPException handler for error responses with success: false ----
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "message": exc.detail})
# ---- End HTTPException handler ----

# Run with auto-reload in dev mode if executed directly
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.getenv("MIDEND_BASE_URL", "127.0.0.1"),
        port=int(os.getenv("MIDEND_PORT", 8000)),
        reload=IS_DEV  # auto-reload enabled in development
    )