# Pic2Info Object Detection API

A FastAPI-based object detection service using YOLOv8 (ultralytics), ready for Docker and Hugging Face Spaces.

## Features

- `/` health check route
- `/upload-image` for object detection (upload an image, get detected objects)
- YOLOv8 model (COCO pretrained)
- Docker compatible
- Hugging Face Spaces ready
- MetaUI placeholder (add your UI integration)

## Usage

### Build & Run with Docker

```sh
docker build -t pic2info .
docker run -p 7860:7860 pic2info
```

### API Endpoints

- `GET /` — Health check
- `POST /upload-image` — Upload image for object detection

### Hugging Face Spaces

- Compatible with Spaces (just push this repo)

## Requirements

See `requirements.txt`.

## MetaUI

Add your MetaUI integration in `main.py` where indicated.
