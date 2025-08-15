#!/usr/bin/env python3
"""
Test script to verify frontend API integration
"""

import requests
import json
import os

def test_frontend_api():
    """Test the API endpoints that the frontend uses"""
    
    base_url = "http://localhost:8000"
    
    print("🚀 Testing Frontend API Integration")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed: {health_data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            root_data = response.json()
            print(f"✅ Root endpoint working: {root_data}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False
    
    # Test 3: Image prediction (if test image exists)
    print("\n3. Testing image prediction...")
    test_image_path = "IM-0001-0001.jpeg"
    if os.path.exists(test_image_path):
        try:
            with open(test_image_path, 'rb') as f:
                files = {'file': (test_image_path, f, 'image/jpeg')}
                response = requests.post(f"{base_url}/predict", files=files)
                
                if response.status_code == 200:
                    prediction_data = response.json()
                    print(f"✅ Prediction successful!")
                    print(f"   Prediction: {prediction_data['prediction']}")
                    print(f"   Confidence: {prediction_data['confidence']}%")
                    print(f"   Probabilities: {prediction_data['probabilities']}")
                    
                    # Check if the probabilities are reasonable (not 0% or 100%)
                    normal_prob = prediction_data['probabilities']['normal']
                    pneumonia_prob = prediction_data['probabilities']['pneumonia']
                    
                    if normal_prob == 0.0 or pneumonia_prob == 0.0:
                        print("⚠️  WARNING: One probability is 0% - this might indicate an issue")
                    elif normal_prob == 100.0 or pneumonia_prob == 100.0:
                        print("⚠️  WARNING: One probability is 100% - this might indicate an issue")
                    else:
                        print("✅ Probabilities look reasonable!")
                        
                else:
                    print(f"❌ Prediction failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return False
    else:
        print(f"⚠️  Test image not found: {test_image_path}")
    
    print("\n" + "=" * 50)
    print("🎉 Frontend API integration test completed!")
    return True

if __name__ == "__main__":
    test_frontend_api() 