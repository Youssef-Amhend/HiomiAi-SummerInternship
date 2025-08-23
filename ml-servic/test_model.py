import torch
import torch.nn as nn
from torchvision.models import resnet18
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if the model can be loaded correctly"""
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize model - use direct ResNet18 structure to match the saved model
        model = resnet18(pretrained=False)
        
        # Modify the first layer to accept grayscale input (1 channel instead of 3)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the final layer for binary classification
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        logger.info("Model initialized successfully")
        
        # Load the trained weights
        model_path = "../ml-service/resnet18_pneumonia_best.pth"
        logger.info(f"Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        logger.info(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded using 'model_state_dict' key")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                logger.info("Loaded using 'state_dict' key")
            else:
                model.load_state_dict(checkpoint)
                logger.info("Loaded using direct state_dict")
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded using direct state_dict")
        
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully!")
        
        # Test with a dummy input (grayscale - 1 channel)
        dummy_input = torch.randn(1, 1, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            logger.info(f"Model output shape: {output.shape}")
            logger.info(f"Model output: {output}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("✅ Model loading test passed!")
    else:
        print("❌ Model loading test failed!") 