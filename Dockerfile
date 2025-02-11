# Dockerfile
FROM python:3.11-bullseye

# Set working directory
WORKDIR /code

ARG MIDEND_PORT, GOOGLE_APPLICATION_CREDENTIALS, MIDEND_BASE_URL

EXPOSE $MIDEND_PORT $MIDEND_PORT

COPY ./requirements.txt /code/requirements.txt
COPY ./.env /code/.env

COPY ./$GOOGLE_APPLICATION_CREDENTIALS /code/$GOOGLE_APPLICATION_CREDENTIALS

COPY ./server.py /code/server.py

RUN pip install --no-cache-dir --upgrade --progress-bar off -r requirements.txt

# Command to run the application
CMD uvicorn server:app --host $MIDEND_BASE_URL --port $MIDEND_PORT
