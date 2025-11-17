"""
Web interface for the Audio Language Model (ALM) system
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import sys
import json
import tempfile
from werkzeug.utils import secure_filename
from main import ALMModel, load_model, process_audio_file, format_alm_response
import torch
import yaml

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and config
model = None
config = None

def initialize_model():
    """Initialize the ALM model"""
    global model, config
    try:
        # Load configuration
        config_path = 'config/config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        else:
            # Use default config if file doesn't exist
            config = {
                'data': {
                    'sample_rate': 16000,
                    'n_fft': 400,
                    'hop_length': 160,
                    'n_mels': 80,
                    'max_audio_length': 30
                },
                'model': {
                    'output_dims': {
                        'audio_events': 100,
                        'speaker_diarization': 10,
                        'paralinguistics': 20
                    }
                }
            }
        
        # For web demo, we'll use a simplified approach
        print("Model initialized for web interface")
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Analyze uploaded audio file"""
    global model, config
    
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    
    question = request.form.get('question', '')
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # For web demo, we'll return simulated results
        # In a full implementation, you would process the audio file with the model
        results = {
            "speech_recognition": "This is a sample transcription of the audio content.",
            "audio_events": ["background_noise", "speech", "keyboard_typing"],
            "speakers": ["speaker_1", "speaker_2"],
            "paralinguistics": "neutral_tone",
            "scene_interpretation": "The audio appears to contain a conversation between two speakers with some background noise and typing sounds."
        }
        
        if question:
            results["question_answer"] = f"This is a sample answer to your question: '{question}'"
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'supported_languages': ['english', 'mandarin', 'urdu', 'hindi', 'telugu', 'tamil', 'bangla']
    })

if __name__ == '__main__':
    # Initialize model
    initialize_model()
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)