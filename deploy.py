"""
Deployment script for the Audio Language Model (ALM) web application
"""
import os
import subprocess
import sys
import argparse

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import flask
        import torch
        import librosa
        import yaml
        print("✓ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def setup_directories():
    """Create necessary directories for deployment"""
    directories = ['uploads', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def test_web_app():
    """Test if the web application can start"""
    try:
        # Test import
        from web_app import app
        print("✓ Web application imports successfully")
        return True
    except Exception as e:
        print(f"✗ Error importing web application: {e}")
        return False

def start_development_server():
    """Start the development server"""
    print("Starting ALM web application in development mode...")
    print("Access the application at: http://localhost:5000")
    
    try:
        subprocess.run([sys.executable, "web_app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting the web application: {e}")

def create_production_script():
    """Create a production deployment script"""
    script_content = """#!/bin/bash
# Production deployment script for ALM

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads logs

# Start the application with Gunicorn
gunicorn --bind 0.0.0.0:8000 --workers 4 web_app:app
"""
    
    with open("deploy.sh", "w") as f:
        f.write(script_content)
    
    # Also create a Windows batch file
    batch_content = """@echo off
REM Production deployment script for ALM

REM Install dependencies
pip install -r requirements.txt

REM Create necessary directories
if not exist uploads mkdir uploads
if not exist logs mkdir logs

REM Start the application with Gunicorn
gunicorn --bind 0.0.0.0:8000 --workers 4 web_app:app
"""
    
    with open("deploy.bat", "w") as f:
        f.write(batch_content)
    
    print("✓ Created production deployment scripts (deploy.sh and deploy.bat)")

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy the ALM web application')
    parser.add_argument('--mode', choices=['dev', 'prod'], default='dev',
                        help='Deployment mode: dev (development) or prod (production)')
    parser.add_argument('--test', action='store_true',
                        help='Test the application without starting it')
    
    args = parser.parse_args()
    
    print("Audio Language Model (ALM) Deployment Script")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("Please install the required dependencies first:")
        print("pip install -r requirements.txt")
        return
    
    # Setup directories
    setup_directories()
    
    # Test the application
    if not test_web_app():
        print("Application test failed. Please check the errors above.")
        return
    
    if args.test:
        print("✓ Application test completed successfully")
        return
    
    # Create production scripts if needed
    if args.mode == 'prod':
        create_production_script()
        print("Production deployment scripts created.")
        print("To deploy in production, run: ./deploy.sh (Linux/Mac) or deploy.bat (Windows)")
    else:
        # Start development server
        start_development_server()

if __name__ == "__main__":
    main()