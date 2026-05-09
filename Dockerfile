FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install opencv-python-headless

COPY . .

# Render ignores EXPOSE, but this is good practice
EXPOSE 8000 

# Use shell form (no brackets) to allow $PORT substitution
CMD uvicorn main:app --host 0.0.0.0 --port $PORT