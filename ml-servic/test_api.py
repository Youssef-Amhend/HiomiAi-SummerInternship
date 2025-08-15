#!/usr/bin/env python3
"""
Test script for the Pneumonia Detection API
"""

import requests
import json
import time

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🚀 Testing Pneumonia Detection API")
    print("=" * 50)
    
    # Test health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {result['status']}")
            print(f"   Model loaded: {result['model_loaded']}")
            print(f"   Device: {result['device']}")
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return
    
    # Test root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            result = response.json()
            print("✅ Root endpoint working!")
            print(f"   Message: {result['message']}")
            print(f"   Version: {result['version']}")
        else:
            print(f"❌ Root endpoint failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎉 API is ready for use!")
    print("\n📚 You can now:")
    print("   - Visit http://localhost:8000/docs for interactive API documentation")
    print("   - Upload images to http://localhost:8000/predict for predictions")
    print("   - Use the example_client.py script to test with actual images")
    
    print("\n💡 Example usage:")
    print("   curl -X POST \"http://localhost:8000/predict\" \\")
    print("        -H \"accept: application/json\" \\")
    print("        -H \"Content-Type: multipart/form-data\" \\")
    print("        -F \"file=@your_chest_xray.jpg\"")

if __name__ == "__main__":
    test_api() 