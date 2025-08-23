from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
from typing import Dict, Any
import logging

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
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the ResNet18 model architecture
class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        # Import resnet18 from torchvision
        from torchvision.models import resnet18
        self.model = resnet18(pretrained=False)
        # Modify the final layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Global variables for model and device
model = None
device = None
transform = None

def load_model():
    """Load the trained ResNet18 model"""
    global model, device, transform
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize model - use direct ResNet18 structure to match the saved model
        from torchvision.models import resnet18
        model = resnet18(pretrained=False)
        
        # Modify the first layer to accept grayscale input (1 channel instead of 3)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final layer for binary classification
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
        # Load the trained weights
        model_path = "../ml-service/resnet18_pneumonia_best.pth"
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        # Define image transformations for grayscale
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale
        ])
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts"""
    load_model()

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess the uploaded image"""
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale if necessary
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply transformations
        image_tensor = transform(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(device)
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def predict_pneumonia(image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Make prediction on the preprocessed image"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            
            # Check if outputs are extremely large (indicating uncalibrated model)
            max_output = torch.max(torch.abs(outputs)).item()
            
            # If outputs are too extreme, apply temperature scaling
            if max_output > 10:
                temperature = max_output / 5.0  # Scale down extreme outputs
                outputs = outputs / temperature
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction and confidence
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            # Map prediction to label
            labels = ["Normal", "Pneumonia"]
            predicted_label = labels[prediction]
            
            # Get individual probabilities
            normal_prob = probabilities[0][0].item()
            pneumonia_prob = probabilities[0][1].item()
            
            # Apply a minimum probability threshold to avoid 0% or 100%
            min_prob = 0.05  # 5% minimum probability
            max_prob = 0.95  # 95% maximum probability
            
            # Clamp and renormalize probabilities
            normal_prob = max(min_prob, min(max_prob, normal_prob))
            pneumonia_prob = max(min_prob, min(max_prob, pneumonia_prob))
            
            # Renormalize to ensure they sum to 1
            total_prob = normal_prob + pneumonia_prob
            normal_prob = normal_prob / total_prob
            pneumonia_prob = pneumonia_prob / total_prob
            
            # Recalculate confidence based on the clamped probabilities
            confidence = max(normal_prob, pneumonia_prob)
            
            return {
                "prediction": predicted_label,
                "confidence": round(confidence * 100, 2),
                "probabilities": {
                    "normal": round(normal_prob * 100, 2),
                    "pneumonia": round(pneumonia_prob * 100, 2)
                }
            }
    
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pneumonia Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image and get pneumonia detection prediction
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with prediction results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        image_tensor = preprocess_image(image_bytes)
        
        # Make prediction
        result = predict_pneumonia(image_tensor)
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Upload multiple images and get predictions for each
    
    Args:
        files: List of image files
    
    Returns:
        JSON with predictions for all images
    """
    try:
        results = []
        
        for i, file in enumerate(files):
            try:
                # Validate file type
                if not file.content_type.startswith('image/'):
                    results.append({
                        "filename": file.filename,
                        "error": "File must be an image"
                    })
                    continue
                
                # Read image bytes
                image_bytes = await file.read()
                
                # Preprocess image
                image_tensor = preprocess_image(image_bytes)
                
                # Make prediction
                result = predict_pneumonia(image_tensor)
                result["filename"] = file.filename
                results.append(result)
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return JSONResponse(content={"predictions": results})
    
    except Exception as e:
        logger.error(f"Unexpected error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 