"""
Installation Summary for ALM System
Shows what has been installed and what's still needed
"""

def show_installation_summary():
    """Show the installation status of the ALM system"""
    print("Audio Language Model (ALM) Installation Summary")
    print("=" * 50)
    
    # Required packages
    required_packages = [
        "torch",
        "torchaudio", 
        "librosa",
        "soundfile",
        "numpy",
        "pandas",
        "scikit-learn",
        "pyyaml",
        "transformers"
    ]
    
    # Check what's installed
    installed = []
    not_installed = []
    
    for package in required_packages:
        try:
            if package == "torch":
                import torch
                installed.append(f"✓ {package} ({torch.__version__})")
            elif package == "librosa":
                import librosa
                installed.append(f"✓ {package} ({librosa.__version__})")
            elif package == "numpy":
                import numpy
                installed.append(f"✓ {package} ({numpy.__version__})")
            elif package == "pandas":
                import pandas
                installed.append(f"✓ {package} ({pandas.__version__})")
            elif package == "pyyaml":
                import yaml
                installed.append(f"✓ {package}")
            elif package == "transformers":
                import transformers
                installed.append(f"✓ {package} ({transformers.__version__})")
            else:
                __import__(package)
                installed.append(f"✓ {package}")
        except ImportError:
            not_installed.append(f"✗ {package}")
    
    print("INSTALLED PACKAGES:")
    print("-" * 20)
    if installed:
        for item in installed:
            print(item)
    else:
        print("None")
    
    print("\nNOT INSTALLED:")
    print("-" * 15)
    if not_installed:
        for item in not_installed:
            print(item)
    else:
        print("None")
    
    print("\n" + "=" * 50)
    print("SYSTEM STATUS:")
    print("=" * 50)
    
    if len(installed) == len(required_packages):
        print("✓ ALL PACKAGES INSTALLED!")
        print("✓ The full ALM system is ready to run!")
        print("\nYou can now run:")
        print("  python main.py --help")
        print("  python training/train.py --help")
    else:
        print(f"⚠ {len(not_installed)} packages still needed")
        print("The system will run in limited mode.")
        print("\nTo install missing packages, run:")
        missing_names = [pkg.split()[1] for pkg in not_installed]
        print(f"  pip install {' '.join(missing_names)}")
        print("\nDemo mode is available:")
        print("  python alm_demo.py")

if __name__ == "__main__":
    show_installation_summary()