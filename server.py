import os, time, random, base64, logging
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
from firebase_admin import credentials, firestore, db  # updated to import realtime db

# Load environment variables from .env and verify required ones
load_dotenv()

REQUIRED_VARS = ["API_BASE_URL", "API_PORT", "API_USER", "API_PASS", "SECRET_KEY"]
missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
if missing:
    raise ValueError(f"Missing environment variables: {', '.join(missing)}")

IS_DEV = os.getenv("ENVIRONMENT", "development").lower() == "development"

# API configuration
API_CONFIG = {
    "base_url": os.getenv("API_BASE_URL"),
    "port": int(os.getenv("API_PORT")),
    "auth": base64.b64encode(f"{os.getenv('API_USER')}:{os.getenv('API_PASS')}".encode()).decode()
}
SECRET_KEY = os.getenv("SECRET_KEY")


# Custom logging formatter to safely handle missing correlation_id
import logging

class SafeFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'NO_CORR_ID'
        # Get formatted time and bold it for INFO; otherwise use unmodified
        timestamp = self.formatTime(record)
        bold_timestamp = f"\033[1m{timestamp}\033[0m"

        # Color the level name based on the log level.
        level = record.levelname.strip()
        if record.levelno == logging.DEBUG:
            colored_level = f"\033[90m{level}\033[0m"
        elif record.levelno == logging.INFO:
            colored_level = f"\033[32m{level}\033[0m"
        elif record.levelno == logging.WARNING:
            colored_level = f"\033[38;5;208m{level}\033[0m"
        elif record.levelno == logging.ERROR:
            colored_level = f"\033[31m{level}\033[0m"
        else:
            colored_level = level
        # Preserve fixed width
        record.levelname = colored_level.ljust(8)

        # For INFO logs, also color "Success:" if present.
        if record.levelno == logging.INFO and isinstance(record.msg, str):
            record.msg = record.msg.replace("Success:", "\033[32mSuccess:\033[0m")

        if isinstance(record.msg, str):
            record.msg = record.msg.replace("Failed:", "\033[31mFailed:\033[0m")

        message = record.getMessage()

        # Assemble the final text.
        text = f"{bold_timestamp} - {record.levelname} - [{record.correlation_id}] {message}"

        # For DEBUG logs, wrap the whole text in gray.
        if record.levelno == logging.DEBUG:
            text = f"\033[90m{text}\033[0m"
        return text

handler = logging.StreamHandler()
# Use fixed width for level output as shown in formatter string below
formatter = SafeFormatter('%(asctime)s - %(levelname)s - [%(correlation_id)s] %(message)s')
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if IS_DEV else logging.INFO)
logger.addHandler(handler)

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
        logger.info("Firebase authentication successful")
        return True
    except Exception as e:
        logger.error(f"Firebase authentication failed: {str(e)}")
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

# Helper: send API request (synchronous)
def secure_api_request(endpoint: str, data: dict, group_name: str) -> dict:
    url = f"http://{API_CONFIG['base_url']}:{API_CONFIG['port']}/{endpoint}"
    headers = {
        "Authorization": f"Basic {API_CONFIG['auth']}",
        "Content-Type": "application/json"
    }
    try:
        response = httpx.post(url, json=data, headers=headers, timeout=1000.0)
        response.raise_for_status()
        logger.info(f"Message sent to {group_name} ({data['phone']})", extra={'correlation_id': 'sync'})
        return response.json()
    except Exception as error:
        logger.error(f"Failed to send to {group_name}: {str(error)}", extra={'correlation_id': 'sync'})
        raise

# ---- New Helper: Log message in Firestore under structured "data" ----
def log_message_firestore(user_id: str, group: Group, msg_text: str, device_id: str, file_path: str = None, file_name: str = None):
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
def execute_queue(request_data: QueueRequest, request: Request):
    # Validate API key
    validate_api_request(request)
    correlation_id = str(uuid4())
    logger.info("Starting queue execution", extra={'correlation_id': correlation_id})
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
                logger.debug(f"Initial delay: {delay:.1f}s", extra={'correlation_id': correlation_id})
                time.sleep(delay)
            logger.info(f"Processing {idx}/{total} for group {group.name}", extra={'correlation_id': correlation_id})
            endpoint = "send/image" if request_data.filePath else "send/message"
            payload = {
                "phone": group.id,
                ** ({"caption": request_data.message, "image_url": request_data.filePath} if request_data.filePath else {"message": request_data.message}),
            }
            secure_api_request(endpoint, payload, group.name)
            success_count += 1
            log_message_firestore(user_id, group, request_data.message, device_id, file_path=request_data.filePath, file_name=request_data.fileName)
            # Remove individual message from realtime db queue (using list index as key)
            rt_queue_ref.child("data").child(str(idx-1)).delete()
        except Exception as e:
            failed_count += 1
            logger.error(f"Error for {group.name}: {str(e)}", extra={'correlation_id': correlation_id})
        if idx < total:
            delay = random.uniform(4, 6)
            logger.debug(f"Delay between messages: {delay:.1f}s", extra={'correlation_id': correlation_id})
            time.sleep(delay)

    exec_time = (perf_counter() - start) * 1000
    logger.info(f"Completed queue in {exec_time:.0f}ms | Success: {success_count}, Failed: {failed_count}", extra={'correlation_id': correlation_id})
    
    # ---- Remove realtime database queue node completely if processing is done ----
    rt_queue_ref.delete()
    # ---- End realtime removal ----

    return QueueResponse(success=(failed_count == 0), executionTime=exec_time, summary={"total": total, "successful": success_count, "failed": failed_count})

# Global exception handler for unexpected errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
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
        log_level="debug" if IS_DEV else "info",
        reload=IS_DEV  # auto-reload enabled in development
    )