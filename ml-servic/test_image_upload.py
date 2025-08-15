#!/usr/bin/env python3
"""
Test script to upload the test image and get prediction
"""

import requests
import json
import os

def test_image_upload():
    """Test uploading the test image and getting prediction"""
    
    # Check if test image exists
    test_image_path = "NORMAL2-IM-1440-0001.jpeg"
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    print("🚀 Testing Image Upload and Prediction")
    print("=" * 50)
    print(f"📁 Using test image: {test_image_path}")
    
    # API endpoint
    url = "http://localhost:8000/predict"
    
    try:
        # Open and upload the image
        with open(test_image_path, 'rb') as f:
            files = {'file': (test_image_path, f, 'image/jpeg')}
            
            print("📤 Uploading image...")
            response = requests.post(url, files=files)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print("\n📊 Results:")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']}%")
            print(f"   Probabilities:")
            print(f"     - Normal: {result['probabilities']['normal']}%")
            print(f"     - Pneumonia: {result['probabilities']['pneumonia']}%")
            
            # Add some visual indicators
            if result['prediction'] == 'Pneumonia':
                print("\n⚠️  ALERT: Pneumonia detected!")
                print("   Please consult a healthcare professional immediately.")
            else:
                print("\n✅ Normal chest X-ray detected.")
                print("   No immediate concerns, but always consult with healthcare professionals.")
                
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the server is running on http://localhost:8000")
        print("   Run: python start_server.py")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    test_image_upload() 