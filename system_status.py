"""
System Status Checker for ALM
Checks what components are available and what's missing
"""

def check_system_status():
    """Check what components of the ALM system are available"""
    print("Audio Language Model (ALM) System Status")
    print("=" * 45)
    
    # Check Python version
    import sys
    print(f"Python Version: {sys.version}")
    
    # Check core dependencies
    dependencies = {
        "PyTorch": "torch",
        "Librosa": "librosa",
        "PyYAML": "yaml",
        "NumPy": "numpy",
        "SoundFile": "soundfile"
    }
    
    available_deps = []
    missing_deps = []
    
    for name, module in dependencies.items():
        try:
            __import__(module)
            available_deps.append(name)
            print(f"✓ {name}: Available")
        except ImportError:
            missing_deps.append(name)
            print(f"✗ {name}: Missing")
    
    print("\nSystem Components:")
    print("-" * 20)
    
    # Check ALM modules
    alm_modules = {
        "Config Module": "config",
        "Data Module": "data.datasets",
        "Models Module": "models.alm_model",
        "Training Module": "training.train",
        "Utils Module": "utils.preprocessing"
    }
    
    available_modules = []
    missing_modules = []
    
    for name, module_path in alm_modules.items():
        try:
            # Split the module path and import step by step
            parts = module_path.split('.')
            if len(parts) == 1:
                __import__(parts[0])
            else:
                __import__(parts[0])
                # Try to access the submodule
                getattr(__import__(parts[0]), parts[1])
            available_modules.append(name)
            print(f"✓ {name}: Available")
        except (ImportError, AttributeError):
            missing_modules.append(name)
            print(f"✗ {name}: Missing dependencies")
    
    print("\n" + "=" * 45)
    print("SUMMARY")
    print("=" * 45)
    print(f"Available Dependencies: {len(available_deps)}/{len(dependencies)}")
    print(f"Missing Dependencies: {len(missing_deps)}")
    print(f"Available Modules: {len(available_modules)}/{len(alm_modules)}")
    print(f"Missing Modules: {len(missing_modules)}")
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps).lower())
    
    if missing_modules:
        print(f"Modules with missing dependencies: {', '.join(missing_modules)}")
    
    if len(available_deps) == len(dependencies):
        print("\n✓ All dependencies installed! You can now run the full ALM system.")
    else:
        print("\n⚠ Some dependencies missing. The system will run in limited mode.")
        print("Run the demo to see core functionality.")

if __name__ == "__main__":
    check_system_status()