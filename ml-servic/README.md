# Pneumonia Detection API

A FastAPI backend for pneumonia detection using a trained ResNet18 model. This API allows you to upload chest X-ray images and get predictions for pneumonia detection.

## Features

- 🏥 Pneumonia detection using ResNet18
- 📤 Image upload support (JPEG, PNG, etc.) - automatically converts to grayscale
- 🔄 Batch prediction for multiple images
- 🚀 FastAPI with automatic API documentation
- 🛡️ Error handling and validation
- 📊 Confidence scores and probabilities

## Prerequisites

- Python 3.8 or higher
- PyTorch
- FastAPI
- PIL (Pillow)

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model loading:**
   ```bash
   python test_model.py
   ```

## Usage

### Starting the Server

```bash
# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or run directly:

```bash
python main.py
```

The server will start on `http://localhost:8000`

### API Endpoints

#### 1. Health Check

- **GET** `/health`
- Returns the health status of the API and model

#### 2. Single Image Prediction

- **POST** `/predict`
- Upload a single image and get prediction results

#### 3. Batch Prediction

- **POST** `/predict-batch`
- Upload multiple images and get predictions for all

#### 4. API Documentation

- **GET** `/docs`
- Interactive API documentation (Swagger UI)

### Example Usage

#### Using curl for single image prediction:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

#### Using Python requests:

```python
import requests

# Single image prediction
url = "http://localhost:8000/predict"
files = {"file": open("chest_xray.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()
print(result)
```

#### Example Response:

```json
{
  "prediction": "Pneumonia",
  "confidence": 85.67,
  "probabilities": {
    "normal": 14.33,
    "pneumonia": 85.67
  }
}
```

### API Documentation

Once the server is running, you can access:

- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc

## Model Information

- **Architecture**: ResNet18
- **Classes**: 2 (Normal, Pneumonia)
- **Input size**: 224x224 pixels
- **Input channels**: 1 (Grayscale)
- **Output**: Binary classification with confidence scores
- **Preprocessing**: Images are automatically converted to grayscale and normalized

## Error Handling

The API includes comprehensive error handling for:

- Invalid file types
- Corrupted images
- Model loading errors
- Processing errors

## Development

### Testing

Run the model loading test:

```bash
python test_model.py
```

### Project Structure

```
ml-service/
├── main.py                 # FastAPI application
├── test_model.py          # Model loading test
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── resnet18_pneumonia_best.pth  # Trained model
```

## Troubleshooting

### Common Issues

1. **Model loading fails**: Ensure the model file `resnet18_pneumonia_best.pth` is in the same directory as `main.py`

2. **CUDA out of memory**: The model will automatically fall back to CPU if CUDA is not available or runs out of memory

3. **Image processing errors**: Make sure uploaded files are valid image formats (JPEG, PNG, etc.)

4. **Grayscale conversion**: The model expects grayscale images, but the API automatically converts RGB images to grayscale

### Logs

The application logs important information including:

- Model loading status
- Device being used (CPU/GPU)
- Processing errors
- Prediction results

## License

This project is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!
