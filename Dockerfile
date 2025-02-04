# Dockerfile
FROM python:3.11-bullseye

# Set working directory
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./.env /code/.env

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

RUN pip install --progress-bar off -r requirements.txt

# Command to run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9696"]
