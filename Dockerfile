# Dockerfile
FROM python:3.11-bullseye

# Set working directory
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./.env /code/.env

COPY ./$GOOGLE_APPLICATION_CREDENTIALS /code/$GOOGLE_APPLICATION_CREDENTIALS

COPY ./server.py /code/server.py

COPY ./utils /code/utils

RUN pip install --no-cache-dir --upgrade --progress-bar off -r requirements.txt

# Command to run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9696"]
