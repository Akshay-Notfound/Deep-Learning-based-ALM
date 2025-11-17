# Deep Learning based Audio Language Model (ALM)

## Overview
This project implements an Audio Language Model (ALM) capable of simultaneously recognizing and jointly understanding speech and non-speech audio elements with reasoning capabilities.

## Features
- Speech Recognition for Asian languages (Mandarin, Urdu, Hindi, Telugu, Tamil, Bangla) and English
- Non-Speech Audio Understanding (music, alarms, environmental noises)
- Speaker Diarization (differentiating between speakers)
- Paralinguistic Analysis (emotion, tone, hesitation)
- Audio Event Detection (car honking, dog barking, aircraft sounds, etc.)
- Joint understanding of speech and non-speech elements for complex reasoning
- Web interface for easy access and deployment

## Project Structure
```
├── data/
│   ├── raw/
│   ├── processed/
│   └── datasets.py
├── models/
│   ├── alm_model.py
│   ├── speech_encoder.py
│   ├── audio_encoder.py
│   └── fusion_module.py
├── training/
│   ├── train.py
│   └── trainer.py
├── utils/
│   ├── preprocessing.py
│   └── evaluation.py
├── config/
│   └── config.yaml
├── templates/
│   └── index.html
├── main.py
├── web_app.py
├── deploy.py
└── requirements.txt
```

## Requirements
- Python 3.8+
- PyTorch 1.9+
- Transformers
- Librosa
- SoundFile
- NumPy
- Pandas
- Flask
- Gunicorn

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Akshay-Notfound/Deep-Learning-based-ALM.git
   cd Deep-Learning-based-ALM
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Initialize the project with sample data:
   ```
   python init_project.py
   ```

## Web Interface

The project includes a web interface built with Flask for easy access to the ALM functionality.

To run the web interface locally:

1. Start the web application:
   ```
   python web_app.py
   ```

2. Open your browser and navigate to `http://localhost:5000`

## Deployment

### Development Mode

To run the application in development mode:

```
python deploy.py --mode dev
```

### Production Mode

To create production deployment scripts:

```
python deploy.py --mode prod
```

This will generate `deploy.sh` (for Linux/Mac) and `deploy.bat` (for Windows) files that can be used to deploy the application in a production environment.

### Manual Production Deployment

For manual deployment, you can use Gunicorn:

```
gunicorn --bind 0.0.0.0:8000 --workers 4 web_app:app
```

## Usage

### Command Line Interface

Run the ALM system from the command line:

```
python main.py --config config/config.yaml --checkpoint path/to/checkpoint.pt --audio path/to/audio.wav
```

### Web Interface

1. Upload an audio file using the web interface
2. Optionally, ask a question about the audio content
3. View the analysis results including speech recognition, audio events, speaker diarization, and paralinguistic analysis

## API Endpoints

- `GET /` - Main web interface
- `POST /analyze` - Analyze an uploaded audio file
- `GET /api/status` - Check API status

## Demo

Run the demo to see a simulation of the ALM capabilities without requiring heavy dependencies:

```
python alm_demo.py
```

## Streamlit Web Interface

The project also includes a Streamlit web interface for easy access to the ALM functionality.

To run the Streamlit interface:

1. Install Streamlit:
   ```
   pip install streamlit
   ```

2. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

3. The app will open in your default browser at `http://localhost:8501`

If the `streamlit` command is not found, try:
```
python -m streamlit run streamlit_app.py
```

The Streamlit interface provides:
- File upload for audio analysis
- Question answering about audio content
- Visual results display
- Demo mode to see sample outputs
```