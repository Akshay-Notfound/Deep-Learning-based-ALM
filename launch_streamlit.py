"""
Script to launch the Streamlit app for the Audio Language Model (ALM) system
"""
import subprocess
import sys
import os

def launch_streamlit():
    """Launch the Streamlit application"""
    try:
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the Streamlit app
        app_path = os.path.join(script_dir, "streamlit_app.py")
        
        # Try to launch Streamlit
        print("Launching Streamlit app...")
        print("If this doesn't work, make sure you have installed Streamlit:")
        print("pip install streamlit")
        print()
        print("App will be available at: http://localhost:8501")
        print()
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.headless", "true"
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error launching Streamlit: {e}")
        print("Please make sure Streamlit is installed:")
        print("pip install streamlit")
    except FileNotFoundError:
        print("Streamlit not found. Please install it with:")
        print("pip install streamlit")
    except KeyboardInterrupt:
        print("Streamlit app stopped.")

if __name__ == "__main__":
    launch_streamlit()