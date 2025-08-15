# Use slim Python image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for YOLO, OpenCV, and PIL
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a writable Ultralytics config/cache directory
RUN mkdir -p /app/.ultralytics && chmod -R 777 /app/.ultralytics

# Ensure Ultralytics always uses this directory (bypasses /tmp)
ENV YOLO_CONFIG_DIR=/app/.ultralytics

# Copy and install Python dependencies first (better build caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# (Optional) Pre-download YOLO weights for faster startup in Hugging Face Spaces
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Run FastAPI app (single worker for memory efficiency)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
