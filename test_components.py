"""
Test script to verify ALM components
"""

def test_imports():
    """Test that all modules can be imported"""
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
    
    try:
        from models.speech_encoder import SpeechEncoder, SpeechRecognitionHead
        print("✓ Speech encoder imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import speech encoder: {e}")
        return False
    
    try:
        from models.audio_encoder import AudioEncoder, AudioEventHead, SpeakerDiarizationHead, ParalinguisticHead
        print("✓ Audio encoder imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import audio encoder: {e}")
        return False
    
    try:
        from models.fusion_module import FusionModule
        print("✓ Fusion module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import fusion module: {e}")
        return False
    
    try:
        from models.alm_model import ALMModel
        print("✓ ALM model imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ALM model: {e}")
        return False
    
    try:
        from data.datasets import ALMDataset
        print("✓ Dataset module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import dataset module: {e}")
        return False
    
    try:
        from utils.preprocessing import load_and_preprocess_audio, extract_mel_spectrogram
        print("✓ Preprocessing utilities imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import preprocessing utilities: {e}")
        return False
    
    try:
        from utils.evaluation import compute_classification_metrics
        print("✓ Evaluation utilities imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluation utilities: {e}")
        return False
    
    return True


def test_model_creation():
    """Test that models can be created"""
    try:
        import torch
        from models.alm_model import ALMModel
        
        # Sample configuration
        config = {
            'data': {
                'languages': ['english', 'mandarin', 'urdu', 'hindi', 'telugu', 'tamil', 'bangla']
            },
            'model': {
                'speech_encoder': {
                    'hidden_size': 768,
                    'num_layers': 12,
                    'dropout': 0.1
                },
                'audio_encoder': {
                    'hidden_size': 768,
                    'num_layers': 6,
                    'dropout': 0.1
                },
                'fusion_module': {
                    'hidden_size': 1024,
                    'num_heads': 8,
                    'num_layers': 6,
                    'dropout': 0.1
                },
                'output_dims': {
                    'speech_recognition': 50000,
                    'audio_events': 100,
                    'speaker_diarization': 10,
                    'paralinguistics': 20
                }
            }
        }
        
        # Create model
        model = ALMModel(config)
        print("✓ ALM model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create ALM model: {e}")
        return False


def test_dataset_creation():
    """Test that datasets can be created"""
    try:
        from data.datasets import ALMDataset
        import os
        
        # Create dummy config
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
                    'speech_recognition': 50000,
                    'audio_events': 100,
                    'speaker_diarization': 10,
                    'paralinguistics': 20
                }
            },
            'training': {
                'batch_size': 4
            }
        }
        
        # Create dataset (this will use dummy data since we don't have real data yet)
        dataset = ALMDataset('.', config, 'train')
        print("✓ ALM dataset created successfully")
        print(f"  Dataset size: {len(dataset)}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create ALM dataset: {e}")
        return False


def main():
    """Run all tests"""
    print("Testing ALM Components")
    print("=" * 30)
    
    # Test imports
    print("\n1. Testing module imports...")
    if not test_imports():
        print("Import tests failed!")
        return
    
    # Test model creation
    print("\n2. Testing model creation...")
    if not test_model_creation():
        print("Model creation tests failed!")
        return
    
    # Test dataset creation
    print("\n3. Testing dataset creation...")
    if not test_dataset_creation():
        print("Dataset creation tests failed!")
        return
    
    print("\n" + "=" * 30)
    print("All tests passed! ✓")
    print("\nNext steps:")
    print("1. Run init_project.bat to install dependencies and generate sample data")
    print("2. Train the model using: python training/train.py")
    print("3. Run inference using: python main.py")


if __name__ == '__main__':
    main()