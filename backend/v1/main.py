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
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl
from PIL import Image

# ---------------------------------------------------------------------
# Ultralytics config dir: avoid /tmp permission issues on Spaces
# ---------------------------------------------------------------------
ULTRA_DIR = os.getenv("YOLO_CONFIG_DIR", "/app/.ultralytics")
os.environ["YOLO_CONFIG_DIR"] = ULTRA_DIR
os.makedirs(ULTRA_DIR, exist_ok=True)

# Optional: requests is used only when the client sends an image URL in JSON.
try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

app = FastAPI(title="Pic2Info API", version="1.0.0")

# Enable CORS for easy frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
#   Model loading helpers
# -------------------------
_MODEL = None
_CLASS_NAMES: List[str] = []


def _normalize_names(names_obj: Any) -> List[str]:
    """Ultralytics may return names as list, dict, tuple, etc. Always return a list[str]."""
    if isinstance(names_obj, dict):
        return [str(v) for v in names_obj.values()]
    if isinstance(names_obj, list):
        return [str(v) for v in names_obj]
    try:
        return [str(v) for v in list(names_obj)]
    except Exception:
        return []


def _allowlist_ultralytics_detectionmodel_safely() -> None:
    """Allow-list Ultralytics DetectionModel for PyTorch >= 2.6 safe loading."""
    try:
        import torch  # type: ignore
        from ultralytics.nn.tasks import DetectionModel  # type: ignore

        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if callable(add_safe_globals):
            add_safe_globals([DetectionModel])
    except Exception:
        # If this fails, proceed; later load attempt may still succeed.
        pass


def _load_model(weights: str = "yolov8n.pt"):
    """Load YOLO model with robust handling across Torch/Ultralytics versions."""
    global _CLASS_NAMES
    from ultralytics import YOLO  # type: ignore

    # Attempt 1: standard load
    try:
        model = YOLO(weights)
        names_obj = getattr(getattr(model, "model", None), "names", [])
        _CLASS_NAMES = _normalize_names(names_obj)
        return model
    except Exception as e1:
        # Attempt 2: allow-list DetectionModel, then retry
        _allowlist_ultralytics_detectionmodel_safely()
        try:
            model = YOLO(weights)
            names_obj = getattr(getattr(model, "model", None), "names", [])
            _CLASS_NAMES = _normalize_names(names_obj)
            return model
        except Exception as e2:
            # Attempt 3: explicitly force weights_only=False as last resort
            try:
                import torch  # type: ignore
                # Ensure the file exists; if not, YOLO will download it once so we can torch.load
                if not os.path.exists(weights):
                    _ = YOLO(weights)  # triggers download to local cache
                _allowlist_ultralytics_detectionmodel_safely()
                _ = torch.load(weights, map_location="cpu", weights_only=False)
                # After a successful unsafe load, retry YOLO (it now can unpickle)
                model = YOLO(weights)
                names_obj = getattr(getattr(model, "model", None), "names", [])
                _CLASS_NAMES = _normalize_names(names_obj)
                return model
            except Exception as e3:
                raise RuntimeError(
                    f"Failed to load YOLO weights '{weights}'.\n"
                    f"Attempt 1: {type(e1).__name__}: {e1}\n"
                    f"Attempt 2: {type(e2).__name__}: {e2}\n"
                    f"Attempt 3: {type(e3).__name__}: {e3}\n"
                    "If on PyTorch >= 2.6, try torch==2.5.1 or ensure Ultralytics is up to date."
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
def _read_image_from_upload(file: UploadFile) -> Image.Image:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def _read_image_from_url(url: str) -> Image.Image:
    if requests is None:
        raise HTTPException(status_code=500, detail="'requests' is not available on this runtime.")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")


def _encode_png(img: np.ndarray | Image.Image) -> str:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _safe_version(pkg: str) -> Optional[str]:
    try:
        mod = __import__(pkg)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


# -------------------------
#         Endpoints
# -------------------------
@app.on_event("startup")
def _startup_load_model():
    global _MODEL
    weights = os.getenv("PIC2INFO_WEIGHTS", "yolov8n.pt")
    _MODEL = _load_model(weights)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "Pic2Info API",
        "status": "ok",
        "model_loaded": _MODEL is not None,
        "labels_count": len(_CLASS_NAMES),
        "labels_preview": list(_CLASS_NAMES)[:10],  # always a list, but stay defensive
        "env": {
            "torch": _safe_version("torch"),
            "ultralytics": _safe_version("ultralytics"),
        },
    }


@app.get("/labels")
def labels() -> Dict[str, Any]:
    return {"count": len(_CLASS_NAMES), "labels": _CLASS_NAMES}


@app.post("/predict", response_model=PredictResponse)
def predict(
    file: Optional[UploadFile] = File(None, description="Image file to analyze"),
    payload: Optional[URLRequest] = Body(None, description="JSON body with image_url and options"),
):
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
        raise HTTPException(status_code=400, detail="Provide an image file or a JSON body with image_url.")

    # Run inference
    t0 = time.time()
    try:
        results = _MODEL.predict(
            source=image,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    if not results:
        raise HTTPException(status_code=500, detail="Model returned no results.")

    r = results[0]

    # Extract detections
    detections: List[Detection] = []
    try:
        xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r, "boxes") and r.boxes is not None else np.zeros((0, 4))
        confs = r.boxes.conf.cpu().numpy() if hasattr(r, "boxes") and r.boxes is not None else np.zeros((0,))
        clss = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r, "boxes") and r.boxes is not None else np.zeros((0,), dtype=int)

        for i in range(len(clss)):
            cls_id = int(clss[i])
            label = _CLASS_NAMES[cls_id] if 0 <= cls_id < len(_CLASS_NAMES) else str(cls_id)
            detections.append(
                Detection(
                    cls=cls_id,
                    label=label,
                    conf=float(confs[i]),
                    box_xyxy=[float(v) for v in xyxy[i].tolist()],
                )
            )
    except Exception:
        detections = []

    # Render annotated image
    try:
        annotated = r.plot()
        if isinstance(annotated, np.ndarray):
            # r.plot() often returns BGR; convert to RGB if needed
            if annotated.shape[-1] == 3:
                annotated = annotated[..., ::-1]
            annotated_img = Image.fromarray(annotated)
        else:
            annotated_img = annotated  # type: ignore
    except Exception:
        annotated_img = image

    b64_png = _encode_png(annotated_img)
    dt_ms = int((time.time() - t0) * 1000)

    w, h = image.size
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
async def upload_image(file: UploadFile = File(...)):
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    image = _read_image_from_upload(file)

    # Run inference with sensible defaults
    try:
        results = _MODEL.predict(source=image, conf=0.25, iou=0.45, max_det=300, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    if not results:
        raise HTTPException(status_code=500, detail="Model returned no results.")

    r = results[0]

    # Convert to UploadDetection list
    detections: List[UploadDetection] = []
    person_count = 0
    max_conf = 0.0

    try:
        confs = r.boxes.conf.cpu().numpy() if hasattr(r, "boxes") and r.boxes is not None else np.zeros((0,))
        clss = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r, "boxes") and r.boxes is not None else np.zeros((0,), dtype=int)

        for i in range(len(clss)):
            cls_id = int(clss[i])
            label = _CLASS_NAMES[cls_id] if 0 <= cls_id < len(_CLASS_NAMES) else str(cls_id)
            conf = float(confs[i])
            detections.append(UploadDetection(label=label, confidence=conf))
            max_conf = max(max_conf, conf)
            if label == "person":
                person_count += 1
    except Exception:
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

    return UploadResponse(detections=detections, metadata=meta, threat_level=threat_level)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)
