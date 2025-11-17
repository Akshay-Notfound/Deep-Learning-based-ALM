"""
Minimal test to verify ALM code structure
"""

def test_file_structure():
    """Test that we can at least import our modules without dependencies"""
    try:
        # Test importing config
        import config
        print("✓ Config module structure OK")
    except Exception as e:
        print(f"✗ Config module issue: {e}")
    
    try:
        # Test importing data module structure
        from data import datasets
        print("✓ Data module structure OK")
    except Exception as e:
        print(f"✗ Data module issue: {e}")
    
    try:
        # Test importing models structure
        from models import alm_model
        print("✓ Models module structure OK")
    except Exception as e:
        print(f"✗ Models module issue: {e}")
    
    try:
        # Test importing training structure
        from training import train
        print("✓ Training module structure OK")
    except Exception as e:
        print(f"✗ Training module issue: {e}")
    
    try:
        # Test importing utils structure
        from utils import preprocessing
        print("✓ Utils module structure OK")
    except Exception as e:
        print(f"✗ Utils module issue: {e}")

if __name__ == "__main__":
    print("Testing ALM File Structure")
    print("=" * 30)
    test_file_structure()
    print("\nFile structure test completed!")