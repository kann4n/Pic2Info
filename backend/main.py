"""
FastAPI backend for Pic2Info — works locally and on Hugging Face Spaces.

Endpoints
---------
GET  /               -> Health check & metadata
GET  /labels         -> Class names available in the YOLO model
POST /predict        -> Detect objects in an image (multipart upload or JSON URL)

Usage (local)
-------------
uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1

Notes
-----
• Safe model loading for PyTorch >= 2.6 is implemented; falls back gracefully.
• Returns both raw detections and an annotated image (base64 PNG).
• CORS enabled for browser clients.
"""
from __future__ import annotations

import base64
import io
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from PIL import Image

# Optional: requests is used only when the client sends an image URL in JSON.
try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and class names
_MODEL = None
_CLASS_NAMES: Dict[int, str] = {}

# Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB default
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# -------------------------
#   Model loading helpers
# -------------------------

def _allowlist_ultralytics_detectionmodel_safely() -> None:
    """Allow-list the Ultralytics DetectionModel for PyTorch >= 2.6 safe loading.

    This avoids UnpicklingError when torch.load uses weights_only=True by default.
    We only do this if the ultralytics class is available and torch exposes the helper.
    """
    try:
        import torch  # type: ignore
        from ultralytics.nn.tasks import DetectionModel  # type: ignore

        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if callable(add_safe_globals):
            add_safe_globals([DetectionModel])
            logger.info("Successfully added DetectionModel to safe globals")
    except Exception as e:
        # If this fails, we simply proceed; a later load attempt may still succeed
        logger.warning(f"Failed to add safe globals: {e}")


def _load_model(weights: str = "yolov8n.pt"):
    """Load YOLO model with robust handling across Torch/Ultralytics versions."""
    global _CLASS_NAMES
    
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        raise RuntimeError("ultralytics package is required but not installed")

    logger.info(f"Loading YOLO model: {weights}")

    # First attempt: standard load (Ultralytics will auto-download if missing)
    try:
        model = YOLO(weights)
        _CLASS_NAMES = model.model.names if hasattr(model, "model") and hasattr(model.model, "names") else {}
        logger.info(f"Model loaded successfully with {len(_CLASS_NAMES)} classes")
        return model
    except Exception as e1:
        logger.warning(f"First load attempt failed: {e1}")
        
        # Second attempt: allow-list DetectionModel, then retry
        _allowlist_ultralytics_detectionmodel_safely()
        try:
            model = YOLO(weights)
            _CLASS_NAMES = model.model.names if hasattr(model, "model") and hasattr(model.model, "names") else {}
            logger.info(f"Model loaded on second attempt with {len(_CLASS_NAMES)} classes")
            return model
        except Exception as e2:
            logger.warning(f"Second load attempt failed: {e2}")
            
            # Third attempt: explicitly force weights_only=False as a last resort
            try:
                import torch  # type: ignore
                # Ensure the file exists; if not, YOLO download it once so we can torch.load
                if not os.path.exists(weights):
                    _ = YOLO(weights)  # triggers download to local cache
                _allowlist_ultralytics_detectionmodel_safely()
                _ = torch.load(weights, map_location="cpu", weights_only=False)
                # After a successful unsafe load, retry YOLO (it now can unpickle)
                model = YOLO(weights)
                _CLASS_NAMES = model.model.names if hasattr(model, "model") and hasattr(model.model, "names") else {}
                logger.info(f"Model loaded on third attempt with {len(_CLASS_NAMES)} classes")
                return model
            except Exception as e3:
                logger.error("All model loading attempts failed")
                raise RuntimeError(
                    f"Failed to load YOLO weights '{weights}'.\n"
                    f"Attempt 1: {type(e1).__name__}: {e1}\n"
                    f"Attempt 2: {type(e2).__name__}: {e2}\n"
                    f"Attempt 3: {type(e3).__name__}: {e3}\n"
                    "If on PyTorch >= 2.6, try torch==2.5.1 or ensure Ultralytics is up to date."
                )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    global _MODEL
    try:
        weights = os.getenv("PIC2INFO_WEIGHTS", "yolov8n.pt")
        _MODEL = _load_model(weights)
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Application shutdown")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Pic2Info API",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for easy frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
#     Pydantic schemas
# -------------------------
class URLRequest(BaseModel):
    image_url: HttpUrl = Field(..., description="Publicly accessible image URL")
    conf: Optional[float] = Field(0.25, ge=0.0, le=1.0, description="Confidence threshold")
    iou: Optional[float] = Field(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
    max_det: Optional[int] = Field(300, ge=1, le=3000, description="Max detections per image")


class Detection(BaseModel):
    cls: int
    label: str
    conf: float
    box_xyxy: List[float]  # [x1, y1, x2, y2]


class PredictResponse(BaseModel):
    duration_ms: int
    width: int
    height: int
    detections: List[Detection]
    annotated_image_png_base64: str


# -------------------------
#   Response for /upload-image (Next.js client)
# -------------------------
class UploadDetection(BaseModel):
    label: str
    confidence: float  # 0..1


class UploadMetadata(BaseModel):
    location: Optional[str] = None
    timestamp: Optional[str] = None
    device: Optional[str] = None
    dimensions: Optional[str] = None


class UploadResponse(BaseModel):
    detections: List[UploadDetection]
    metadata: Optional[UploadMetadata] = None
    threat_level: Optional[str] = None  # LOW | MEDIUM | HIGH | CRITICAL


# -------------------------
#        Utilities
# -------------------------

def _validate_file_size(file: UploadFile) -> None:
    """Validate file size to prevent memory issues."""
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE} bytes."
        )


def _validate_file_extension(filename: str) -> None:
    """Validate file extension."""
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")
    
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def _read_image_from_upload(file: UploadFile) -> Image.Image:
    """Read and validate image from file upload."""
    _validate_file_size(file)
    _validate_file_extension(file.filename or "")
    
    try:
        data = file.file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file upload.")
        
        image = Image.open(io.BytesIO(data)).convert("RGB")
        logger.info(f"Loaded image: {image.size[0]}x{image.size[1]}")
        return image
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    finally:
        try:
            file.file.close()
        except Exception:
            pass


def _read_image_from_url(url: str) -> Image.Image:
    """Read and validate image from URL."""
    if requests is None:
        raise HTTPException(
            status_code=500, 
            detail="URL image loading is not available on this runtime."
        )
    
    try:
        logger.info(f"Fetching image from URL: {url}")
        response = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        
        # Check content length if available
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"Image too large. Maximum size is {MAX_FILE_SIZE} bytes."
            )
        
        image_data = response.content
        if len(image_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"Image too large. Maximum size is {MAX_FILE_SIZE} bytes."
            )
        
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        logger.info(f"Loaded image from URL: {image.size[0]}x{image.size[1]}")
        return image
        
    except HTTPException:
        raise
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image from URL: {e}")


def _encode_png(img: np.ndarray | Image.Image) -> str:
    """Encode image as base64 PNG."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _safe_version(pkg: str) -> Optional[str]:
    """Get package version safely."""
    try:
        mod = __import__(pkg)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


# -------------------------
#         Endpoints
# -------------------------

@app.get("/")
def root() -> Dict[str, Any]:
    """Health check and API information."""
    labels_list = list(_CLASS_NAMES.values()) if _CLASS_NAMES else []
    
    return {
        "name": "Pic2Info API",
        "status": "ok",
        "model_loaded": _MODEL is not None,
        "labels_count": len(_CLASS_NAMES),
        "labels_preview": labels_list[:10],
        "config": {
            "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
            "request_timeout_sec": REQUEST_TIMEOUT,
            "allowed_extensions": list(ALLOWED_EXTENSIONS),
        },
        "env": {
            "torch": _safe_version("torch"),
            "ultralytics": _safe_version("ultralytics"),
        },
    }


@app.get("/labels")
def labels() -> Dict[str, Any]:
    """Get available class labels from the model."""
    labels_list = list(_CLASS_NAMES.values()) if _CLASS_NAMES else []
    return {
        "count": len(_CLASS_NAMES), 
        "labels": labels_list,
        "labels_dict": _CLASS_NAMES
    }


@app.post("/predict", response_model=PredictResponse)
def predict(
    file: Optional[UploadFile] = File(None, description="Image file to analyze"),
    payload: Optional[URLRequest] = Body(None, description="JSON body with image_url and options"),
):
    """Detect objects in an image and return detections with annotated image."""
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Read image either from upload or URL
    if file is not None:
        image = _read_image_from_upload(file)
        conf = 0.25
        iou = 0.45
        max_det = 300
    elif payload is not None:
        image = _read_image_from_url(str(payload.image_url))
        conf = payload.conf or 0.25
        iou = payload.iou or 0.45
        max_det = payload.max_det or 300
    else:
        raise HTTPException(
            status_code=400, 
            detail="Provide an image file or a JSON body with image_url."
        )

    # Run inference
    t0 = time.time()
    try:
        logger.info(f"Running inference with conf={conf}, iou={iou}, max_det={max_det}")
        results = _MODEL.predict(
            source=image,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    if not results:
        raise HTTPException(status_code=500, detail="Model returned no results.")

    r = results[0]

    # Extract detections
    detections: List[Detection] = []
    try:
        if hasattr(r, "boxes") and r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(clss)):
                cls_id = int(clss[i])
                label = _CLASS_NAMES.get(cls_id, str(cls_id))
                detections.append(
                    Detection(
                        cls=cls_id,
                        label=label,
                        conf=float(confs[i]),
                        box_xyxy=[float(v) for v in xyxy[i].tolist()],
                    )
                )
        
        logger.info(f"Found {len(detections)} detections")
    except Exception as e:
        logger.warning(f"Failed to extract detections: {e}")
        detections = []

    # Render annotated image
    try:
        annotated = r.plot()
        if isinstance(annotated, np.ndarray):
            # Convert BGR to RGB if needed
            if annotated.shape[-1] == 3:
                annotated = annotated[..., ::-1]
            annotated_img = Image.fromarray(annotated)
        else:
            annotated_img = annotated  # type: ignore
    except Exception as e:
        logger.warning(f"Failed to create annotated image: {e}")
        annotated_img = image

    b64_png = _encode_png(annotated_img)
    dt_ms = int((time.time() - t0) * 1000)

    w, h = image.size
    logger.info(f"Prediction completed in {dt_ms}ms")
    
    return PredictResponse(
        duration_ms=dt_ms,
        width=w,
        height=h,
        detections=detections,
        annotated_image_png_base64=b64_png,
    )


# -------------------------
#   Next.js compatible endpoint: /upload-image
# -------------------------
@app.post("/upload-image", response_model=UploadResponse)
def upload_image(file: UploadFile = File(...)):
    """Upload image endpoint compatible with Next.js frontend."""
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    image = _read_image_from_upload(file)

    # Run inference with sensible defaults
    try:
        logger.info("Running inference for upload-image endpoint")
        results = _MODEL.predict(
            source=image, 
            conf=0.25, 
            iou=0.45, 
            max_det=300, 
            verbose=False
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    if not results:
        raise HTTPException(status_code=500, detail="Model returned no results.")

    r = results[0]

    # Convert to UploadDetection list
    detections: List[UploadDetection] = []
    person_count = 0
    max_conf = 0.0

    try:
        if hasattr(r, "boxes") and r.boxes is not None:
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(clss)):
                cls_id = int(clss[i])
                label = _CLASS_NAMES.get(cls_id, str(cls_id))
                conf = float(confs[i])
                detections.append(UploadDetection(label=label, confidence=conf))
                max_conf = max(max_conf, conf)
                if label == "person":
                    person_count += 1
        
        logger.info(f"Upload endpoint found {len(detections)} detections")
    except Exception as e:
        logger.warning(f"Failed to extract detections for upload endpoint: {e}")
        detections = []

    # Simple, conservative threat heuristic (customize as needed)
    labels_set = {d.label for d in detections}
    if any(lbl in labels_set for lbl in ["knife", "scissors"]) and max_conf > 0.60:
        threat_level = "HIGH"
    elif person_count >= 1 and max_conf > 0.75:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"

    w, h = image.size
    meta = UploadMetadata(
        location="Unknown",
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        device="Unknown Camera",
        dimensions=f"{w}x{h}",
    )

    return UploadResponse(
        detections=detections, 
        metadata=meta, 
        threat_level=threat_level
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)