import requests
import json
import os
from pathlib import Path

def test_health_check(base_url="http://localhost:8000"):
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")

def test_single_prediction(image_path, base_url="http://localhost:8000"):
    """Test single image prediction"""
    try:
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return
        
        print(f"📤 Uploading image: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Probabilities: {result['probabilities']}")
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")

def test_batch_prediction(image_paths, base_url="http://localhost:8000"):
    """Test batch prediction with multiple images"""
    try:
        # Filter out non-existent files
        existing_paths = [path for path in image_paths if os.path.exists(path)]
        
        if not existing_paths:
            print("❌ No valid image files found")
            return
        
        print(f"📤 Uploading {len(existing_paths)} images for batch prediction")
        
        files = []
        for path in existing_paths:
            files.append(('files', open(path, 'rb')))
        
        response = requests.post(f"{base_url}/predict-batch", files=files)
        
        # Close all opened files
        for _, file in files:
            file.close()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch prediction successful!")
            for prediction in result['predictions']:
                if 'error' in prediction:
                    print(f"❌ {prediction['filename']}: {prediction['error']}")
                else:
                    print(f"✅ {prediction['filename']}: {prediction['prediction']} ({prediction['confidence']}%)")
        else:
            print(f"❌ Batch prediction failed with status {response.status_code}")
            print(f"Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Batch prediction failed: {str(e)}")

def main():
    """Main function to run all tests"""
    base_url = "http://localhost:8000"
    
    print("🚀 Testing Pneumonia Detection API")
    print("=" * 40)
    
    # Test health check
    print("\n1. Testing health check...")
    test_health_check(base_url)
    
    # Test single prediction (if you have a test image)
    print("\n2. Testing single prediction...")
    # You can replace this with the path to your test image
    test_image_path = "test_image.jpg"  # Change this to your actual test image path
    if os.path.exists(test_image_path):
        test_single_prediction(test_image_path, base_url)
    else:
        print(f"⚠️  Test image not found: {test_image_path}")
        print("   Skipping single prediction test")
    
    # Test batch prediction (if you have multiple test images)
    print("\n3. Testing batch prediction...")
    test_image_paths = ["test1.jpg", "test2.jpg", "test3.jpg"]  # Change these to your actual test image paths
    existing_paths = [path for path in test_image_paths if os.path.exists(path)]
    
    if existing_paths:
        test_batch_prediction(existing_paths, base_url)
    else:
        print("⚠️  No test images found for batch prediction")
        print("   Skipping batch prediction test")
    
    print("\n" + "=" * 40)
    print("🎉 Testing completed!")

if __name__ == "__main__":
    main() 