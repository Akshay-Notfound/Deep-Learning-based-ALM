"""
Project initialization script for the Audio Language Model (ALM)
Sets up the environment and generates sample data
"""

import os
import sys
import subprocess
import argparse
from utils.dataset_generator import generate_synthetic_audio_dataset, create_dataset_statistics


def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    return True


def setup_directories():
    """Create necessary directories"""
    print("Setting up directories...")
    directories = [
        "data/raw",
        "data/processed",
        "checkpoints",
        "logs",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def generate_sample_data(num_samples=100):
    """Generate sample synthetic dataset"""
    print(f"Generating sample dataset with {num_samples} samples...")
    
    # Sample configuration
    config = {
        'data': {
            'sample_rate': 16000,
            'max_audio_length': 30,
            'languages': ['english', 'mandarin', 'urdu', 'hindi', 'telugu', 'tamil', 'bangla'],
            'n_fft': 400,
            'hop_length': 160,
            'n_mels': 80
        },
        'model': {
            'output_dims': {
                'audio_events': 100,
                'speaker_diarization': 10,
                'paralinguistics': 20,
                'speech_recognition': 50000
            }
        }
    }
    
    try:
        generate_synthetic_audio_dataset('data', config, num_samples)
        stats = create_dataset_statistics('data')
        print("Dataset statistics:", stats)
        print("Sample data generated successfully!")
        return True
    except Exception as e:
        print(f"Error generating sample data: {e}")
        return False


def main():
    """Main initialization function"""
    parser = argparse.ArgumentParser(description='Initialize Audio Language Model project')
    parser.add_argument('--skip-deps', action='store_true', 
                        help='Skip dependency installation')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of sample data points to generate')
    args = parser.parse_args()
    
    print("Initializing Audio Language Model (ALM) project...")
    print("="*50)
    
    # Setup directories
    setup_directories()
    
    # Install dependencies (unless skipped)
    if not args.skip_deps:
        if not install_dependencies():
            print("Failed to install dependencies. Continuing anyway...")
    
    # Generate sample data
    if not generate_sample_data(args.samples):
        print("Failed to generate sample data.")
    
    print("="*50)
    print("Project initialization completed!")
    print("\nNext steps:")
    print("1. Review the configuration in config/config.yaml")
    print("2. Prepare your actual dataset or use the generated sample data")
    print("3. Train the model using: python training/train.py")
    print("4. Run inference using: python main.py --checkpoint checkpoints/best_model.pt --audio path/to/audio.wav")


if __name__ == '__main__':
    main()