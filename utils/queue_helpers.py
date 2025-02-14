import datetime
from firebase_admin import firestore, db
from utils.logging_helper import add_correlation_id
import logging

def log_message_firestore(user_id: str, group, msg_text: str, device_id: str, message_id: str, file_path: str = None, file_name: str = None):
    if not user_id:
        user_id = "unknown"
    now = datetime.datetime.now()
    date_str = now.strftime("%d %b, %Y")
    time_str = now.strftime("%H:%M:%S %d %b, %Y")
    message_data = {
        "exactTime": str(int(now.timestamp()*1000)),
        "message": msg_text,
        "user_id": user_id,
        "time": time_str,
        "device_id": device_id,
        "message_id": message_id,
        "toWhomID": [group.id],
        "toWhom": [group.name],
    }
    if file_path:
        message_data.update({"filePath": file_path, "fileName": file_name})
    firestore.client().collection("log").document(date_str).collection(user_id).add(message_data)

def create_rt_queue(queue_id: str, request_data):
    now = datetime.datetime.now()
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
