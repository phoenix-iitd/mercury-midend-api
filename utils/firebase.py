import os
import datetime
import firebase_admin
from firebase_admin import credentials, firestore, db
from dotenv import load_dotenv
from utils.logging_helper import add_correlation_id
import logging

load_dotenv()
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
database_url = os.getenv("DATABASE_URL")
cred_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not project_id or not database_url or not cred_json:
    raise ValueError("Required Firebase environment variables missing")


def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_json)
        firebase_admin.initialize_app(
            cred, {"projectId": project_id, "databaseURL": database_url}
        )
        check_firebase_auth()
    return firestore.client()


def check_firebase_auth():
    try:
        fs_client = firestore.client()
        fs_client.collection("_check_auth").limit(1).get()
        db.reference("/_check_auth").get()
        logging.getLogger("uvicorn.error").info(
            add_correlation_id(
                "Firebase authentication successful", "firebase_init", logging.INFO
            )
        )
        return True
    except Exception as e:
        logging.getLogger("uvicorn.error").error(
            add_correlation_id(
                f"Firebase authentication failed: {str(e)}",
                "firebase_init",
                logging.ERROR,
            )
        )
        raise ValueError(
            "Firebase authentication failed. Please check your credentials."
        )
