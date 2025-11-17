"""
Test script to check if our ALM modules can be imported
"""

def test_imports():
    """Test importing our ALM modules"""
    print("Testing ALM Module Imports")
    print("=" * 30)
    
    # Test config import
    try:
        import config
        print("✓ Config module imported successfully")
    except Exception as e:
        print(f"✗ Config module import failed: {e}")
    
    # Test data module import
    try:
        from data import datasets
        print("✓ Data module imported successfully")
    except Exception as e:
        print(f"✗ Data module import failed: {e}")
    
    # Test models import
    try:
        from models import alm_model
        print("✓ Models module imported successfully")
    except Exception as e:
        print(f"✗ Models module import failed: {e}")
    
    # Test training module import
    try:
        from training import train
        print("✓ Training module imported successfully")
    except Exception as e:
        print(f"✗ Training module import failed: {e}")
    
    # Test utils module import
    try:
        from utils import preprocessing
        print("✓ Utils module imported successfully")
    except Exception as e:
        print(f"✗ Utils module import failed: {e}")

if __name__ == "__main__":
    test_imports()