from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from PIL.ExifTags import TAGS
import io
from datetime import datetime
import hashlib

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model once at startup
model = YOLO("yolov8n.pt")  # nano version for speed

def extract_metadata(image: Image.Image, filename: str):
    """Extract EXIF metadata for OSINT purposes"""
    metadata = {}
    
    try:
        # Basic image info
        metadata["dimensions"] = f"{image.width}x{image.height}"
        metadata["format"] = image.format or "Unknown"
        metadata["size"] = f"{len(image.tobytes()) // 1024} KB"
        
        # Extract EXIF data
        exifdata = image.getexif()
        if exifdata:
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ["DateTime", "DateTimeOriginal"]:
                    metadata["timestamp"] = str(value)
                elif tag == "Make":
                    metadata["device_make"] = str(value)
                elif tag == "Model":
                    metadata["device_model"] = str(value)
                elif tag == "Software":
                    metadata["software"] = str(value)
    except Exception as e:
        print(f"Metadata extraction error: {e}")
    
    # File hash for forensic tracking
    metadata["file_hash"] = hashlib.md5(filename.encode()).hexdigest()[:8]
    
    return metadata

def calculate_threat_level(detections):
    """Calculate threat level based on detected objects"""
    high_threat_objects = ["gun", "knife", "weapon", "fire", "explosion"]
    medium_threat_objects = ["person", "car", "truck", "motorcycle"]
    
    max_confidence = max([d["confidence"] for d in detections], default=0)
    detected_labels = [d["label"].lower() for d in detections]
    
    # Check for high-threat objects
    for threat_obj in high_threat_objects:
        if any(threat_obj in label for label in detected_labels):
            return "HIGH" if max_confidence > 0.7 else "MEDIUM"
    
    # Check for medium-threat objects with high confidence
    for med_obj in medium_threat_objects:
        if any(med_obj in label for label in detected_labels) and max_confidence > 0.8:
            return "MEDIUM"
    
    # Default to LOW
    return "LOW"

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    # Read the uploaded file into memory
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Run YOLO detection
    results = model(image)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        confidence = float(box.conf[0])
        detections.append({
            "label": label,
            "confidence": round(confidence, 3)
        })
        
    if file.filename is not None:
        metadata = extract_metadata(image, file.filename)
    else:
        # Handle the case where no filename is provided
        metadata = extract_metadata(image, "unknown_filename")
    
    # Calculate threat level
    threat_level = calculate_threat_level(detections)

    return {
        "filename": file.filename,
        "detections": detections,
        "metadata": metadata,
        "threat_level": threat_level,
        "analysis_timestamp": datetime.now().isoformat(),
        "total_objects": len(detections)
    }