from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from typing import Dict, Any, Optional
import logging
import requests
from dotenv import load_dotenv  # <-- add this

# Load .env next to this file (adjust path if your .env is elsewhere)
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pneumonia Detection API",
    description="A FastAPI backend for pneumonia detection using ResNet18",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to known origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Config: Object storage (MinIO/S3) ----
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "uploads")
S3_USE_SSL = os.getenv("S3_USE_SSL", "false").lower() == "true"

# ---- Config: Model path (NEW) ----
MODEL_PATH = os.getenv("MODEL_PATH", "resnet18_pneumonia_best.pth")

# Global S3 client
s3_client = None

def init_s3_client():
    global s3_client
    s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
        use_ssl=S3_USE_SSL,
        config=Config(s3={"addressing_style": "path"})
    )
    logger.info("Initialized S3 client for endpoint=%s bucket=%s", S3_ENDPOINT_URL, S3_BUCKET)

def s3_download_bytes(bucket: str, key: str) -> bytes:
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except ClientError as e:
        logger.exception("S3 download failed for %s/%s", bucket, key)
        raise HTTPException(status_code=404, detail=f"Object not found: {key}")

# ---- DTO for upload notification ----
class UploadNotifyRequest(BaseModel):
    filename: str = Field(..., description="Object key or filename")
    userId: str = Field(..., description="User identifier who uploaded the file")
    contentType: str = Field(..., description="MIME type of the uploaded file")
    size: int = Field(..., ge=0, description="Size in bytes")
    callbackUrl: Optional[HttpUrl] = Field(
        None,
        description="Optional URL to receive the prediction result via POST"
    )

# In-memory store for results (for demo/testing)
RESULTS: Dict[str, Dict[str, Any]] = {}

def result_key(user_id: str, filename: str) -> str:
    return f"{user_id}:{filename}"

# Define the ResNet18 model architecture
class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        from torchvision.models import resnet18
        # Use weights=None instead of deprecated pretrained=False
        self.model = resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Global variables for model and device
model = None
device = None
transform = None
model_load_error: Optional[str] = None  # NEW: track last load error message

def load_model():
    """Load the trained ResNet18 model"""
    global model, device, transform, model_load_error
    model_load_error = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        from torchvision.models import resnet18
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = base.fc.in_features
        base.fc = nn.Linear(num_ftrs, 2)

        # Resolve model path and check existence
        resolved_path = os.path.abspath(MODEL_PATH)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"MODEL_PATH does not exist: {resolved_path}")

        checkpoint = torch.load(resolved_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                base.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                base.load_state_dict(checkpoint['state_dict'])
            else:
                base.load_state_dict(checkpoint)
        else:
            base.load_state_dict(checkpoint)

        base.to(device)
        base.eval()

        model = base
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        logger.info("Model loaded successfully from %s", resolved_path)
    except Exception as e:
        # Don't crash app; record error and continue
        model = None
        transform = None
        model_load_error = str(e)
        logger.error("Error loading model: %s", model_load_error)

@app.on_event("startup")
async def startup_event():
    """Load the model and init S3 when the application starts"""
    load_model()
    init_s3_client()

def ensure_model_loaded():
    """Helper to ensure model is loaded before inference"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Error: {model_load_error or 'unknown'}. "
                   f"Set MODEL_PATH correctly and call /admin/reload-model."
        )

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess the uploaded image"""
    try:
        ensure_model_loaded()
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'L':
            image = image.convert('L')
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor.to(device)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def predict_pneumonia(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Make prediction on the preprocessed image"""
    try:
        ensure_model_loaded()
        with torch.no_grad():
            outputs = model(image_tensor)
            max_output = torch.max(torch.abs(outputs)).item()
            if max_output > 10:
                temperature = max_output / 5.0
                outputs = outputs / temperature

            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            labels = ["Normal", "Pneumonia"]

            normal_prob = probabilities[0][0].item()
            pneumonia_prob = probabilities[0][1].item()

            # Clamp and renormalize
            min_prob = 0.05
            max_prob = 0.95
            normal_prob = max(min_prob, min(max_prob, normal_prob))
            pneumonia_prob = max(min_prob, min(max_prob, pneumonia_prob))
            total_prob = normal_prob + pneumonia_prob
            normal_prob /= total_prob
            pneumonia_prob /= total_prob
            confidence = max(normal_prob, pneumonia_prob)

            return {
                "prediction": labels[prediction],
                "confidence": round(confidence * 100, 2),
                "probabilities": {
                    "normal": round(normal_prob * 100, 2),
                    "pneumonia": round(pneumonia_prob * 100, 2)
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Pneumonia Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_error": model_load_error,
        "model_path": os.path.abspath(MODEL_PATH),
        "device": str(device) if device else None
    }

# Admin endpoint to reload the model without restarting (NEW)
@app.post("/admin/reload-model")
async def reload_model():
    load_model()
    if model is None:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {model_load_error}")
    return {"status": "ok", "message": "Model reloaded"}

# ---- Upload notification flow ----
def process_upload_notification(req: UploadNotifyRequest) -> None:
    try:
        logger.info(
            "Processing upload notification: filename=%s userId=%s contentType=%s size=%d",
            req.filename, req.userId, req.contentType, req.size
        )
        image_bytes = s3_download_bytes(S3_BUCKET, req.filename)
        image_tensor = preprocess_image(image_bytes)
        result = predict_pneumonia(image_tensor)

        final_result = {
            "filename": req.filename,
            "userId": req.userId,
            "contentType": req.contentType,
            "size": req.size,
            "result": result
        }
        RESULTS[result_key(req.userId, req.filename)] = final_result
        logger.info("Prediction completed for %s (userId=%s)", req.filename, req.userId)

        if req.callbackUrl:
            try:
                resp = requests.post(req.callbackUrl, json=final_result, timeout=5)
                logger.info("Posted result to callbackUrl=%s status=%s", req.callbackUrl, resp.status_code)
            except Exception as cb_err:
                logger.warning("Failed posting to callbackUrl=%s: %s", req.callbackUrl, cb_err)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed processing upload notification: %s", e)

@app.post("/ml/upload-notify")
async def upload_notify(payload: UploadNotifyRequest, background_tasks: BackgroundTasks):
    try:
        logger.info(f"Received upload notification: {payload}")
        if not payload.filename:
            raise HTTPException(status_code=400, detail="filename is required")
        if payload.size is not None and payload.size < 0:
            raise HTTPException(status_code=400, detail="size must be >= 0")
        
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded, cannot process request")
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        background_tasks.add_task(process_upload_notification, payload)
        logger.info(f"Background task added for filename: {payload.filename}")
        return {"status": "accepted", "filename": payload.filename, "userId": payload.userId}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in upload_notify: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/ml/result")
async def get_result(userId: str = Query(...), filename: str = Query(...)):
    key = result_key(userId, filename)
    data = RESULTS.get(key)
    if not data:
        raise HTTPException(status_code=404, detail="Result not found (not processed yet or wrong keys)")
    return data

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        result = predict_pneumonia(image_tensor)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            try:
                if not file.content_type.startswith('image/'):
                    results.append({"filename": file.filename, "error": "File must be an image"})
                    continue
                image_bytes = await file.read()
                image_tensor = preprocess_image(image_bytes)
                result = predict_pneumonia(image_tensor)
                result["filename"] = file.filename
                results.append(result)
            except Exception as e:
                results.append({"filename": file.filename, "error": str(e)})
        return JSONResponse(content={"predictions": results})
    except Exception as e:
        logger.error(f"Unexpected error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("APP_HOST", "0.0.0.0"), port=int(os.getenv("APP_PORT", "8000")))
