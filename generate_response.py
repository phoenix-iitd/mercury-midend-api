import hmac
import hashlib
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


# Get the secret key from environment
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY not set")

# Create payload
payload = {
    "message": "HMAC Signature test",  # Changed from msg to message
    "device_id": "12345678sdadas90",
    "user_id": "tt1230494",
    "time": int(time.time()),
    "data": [
        {"id": "120363336981499418@g.us"}
    ],
    "filePath": "https://cdn.promptden.com/images/e65d092d-ea84-4381-a0af-dfec0a44308a.jpg?class=standard",
    "fileName": "minion.jpg"
}

# Convert to JSON string - ensure consistent formatting
json_str = json.dumps(payload, separators=(',', ':'))

# Generate HMAC
signature = hmac.new(
    SECRET_KEY.encode(),
    json_str.encode(),
    hashlib.sha256
).hexdigest()

# Print shell export commands
print(f"export TIMESTAMP={payload['time']}")
print(f"export SIGNATURE={signature}")
print(f"export REQUEST_BODY='{json_str}'")