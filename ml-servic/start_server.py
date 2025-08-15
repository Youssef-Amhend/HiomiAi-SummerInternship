#!/usr/bin/env python3
"""
Startup script for the Pneumonia Detection API
"""

import uvicorn
import sys
import os

def main():
    """Main function to start the server"""
    try:
        # Check if the model file exists
        model_path = "resnet18_pneumonia_best.pth"
        if not os.path.exists(model_path):
            print(f"❌ Error: Model file '{model_path}' not found!")
            print("   Please ensure the model file is in the same directory as this script.")
            sys.exit(1)
        
        print("🚀 Starting Pneumonia Detection API...")
        print("📁 Model file found:", model_path)
        print("🌐 Server will be available at: http://localhost:8000")
        print("📚 API documentation will be available at: http://localhost:8000/docs")
        print("=" * 50)
        
        # Start the server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 