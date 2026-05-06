FROM python:3.11-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render ignores EXPOSE, but this is good practice
EXPOSE 8000 

# Use shell form (no brackets) to allow $PORT substitution
CMD uvicorn main:app --host 0.0.0.0 --port $PORT