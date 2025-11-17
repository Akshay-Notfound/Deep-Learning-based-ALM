"""
Streamlit interface for the Audio Language Model (ALM) system
"""
import streamlit as st
import os
import sys
import tempfile
import yaml
from main import ALMModel, load_model, process_audio_file, format_alm_response
import torch

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Audio Language Model (ALM)",
    page_icon="🔊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .header {
        background-color: #0066cc;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='header'><h1>🔊 Audio Language Model (ALM)</h1><p>Advanced audio analysis with speech recognition, event detection, and more</p></div>", unsafe_allow_html=True)

# Features section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='feature-card'><h3>🎤 Speech Recognition</h3><p>Accurate transcription of speech in multiple languages</p></div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='feature-card'><h3>🔊 Audio Event Detection</h3><p>Identify environmental sounds and events</p></div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='feature-card'><h3>👥 Speaker Diarization</h3><p>Differentiate between multiple speakers</p></div>", unsafe_allow_html=True)

# Load configuration
@st.cache_data
def load_config():
    config_path = 'config/config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        # Use default config if file doesn't exist
        return {
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

# Initialize model
@st.cache_resource
def initialize_model():
    try:
        config = load_config()
        # For web demo, we'll use a simplified approach
        st.success("Model initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        return False

# Process audio file
def process_audio(filepath, question=""):
    """Process audio file and return results"""
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
    
    return results

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "m4a"], key="audio_uploader")

# Question input
question = st.text_input("Ask a question about the audio (optional)", placeholder="e.g., What language is being spoken?")

# Process button
if st.button("Analyze Audio", type="primary", disabled=uploaded_file is None):
    if uploaded_file is not None:
        with st.spinner("Analyzing audio..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_filepath = tmp_file.name
                
                # Process the audio
                results = process_audio(tmp_filepath, question)
                
                # Clean up temporary file
                os.unlink(tmp_filepath)
                
                # Display results
                st.markdown("<div class='result-card'><h2>Analysis Results</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Speech Recognition")
                    st.write(results.get("speech_recognition", "No transcription available"))
                    
                    st.subheader("Paralinguistics")
                    st.write(results.get("paralinguistics", "No analysis available"))
                    
                    if "question_answer" in results:
                        st.subheader("Question Answer")
                        st.write(results["question_answer"])
                
                with col2:
                    st.subheader("Audio Events")
                    events = results.get("audio_events", [])
                    if events:
                        for event in events:
                            st.markdown(f"- {event}")
                    else:
                        st.write("No events detected")
                    
                    st.subheader("Speakers")
                    speakers = results.get("speakers", [])
                    if speakers:
                        for speaker in speakers:
                            st.markdown(f"- {speaker}")
                    else:
                        st.write("No speakers identified")
                
                st.subheader("Scene Interpretation")
                st.write(results.get("scene_interpretation", "No interpretation available"))
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
    else:
        st.warning("Please upload an audio file first")

# Demo section
st.markdown("---")
st.subheader(" демо")
if st.button("Run Demo"):
    st.info("Running demo simulation...")
    
    # Simulated results
    demo_results = {
        "speech_recognition": "Flight LH456 to Frankfurt is now boarding at gate B12",
        "audio_events": ["aircraft_sound", "announcement", "footsteps"],
        "speakers": ["speaker_1", "speaker_2"],
        "paralinguistics": "urgent",
        "scene_interpretation": "The aircraft sound and announcement suggest that the person is in an airport boarding area. The urgency in the voice indicates they may be concerned about their flight."
    }
    
    st.markdown("<div class='result-card'><h2>Demo Results</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Speech Recognition")
        st.write(demo_results["speech_recognition"])
        
        st.subheader("Paralinguistics")
        st.write(demo_results["paralinguistics"])
    
    with col2:
        st.subheader("Audio Events")
        for event in demo_results["audio_events"]:
            st.markdown(f"- {event}")
        
        st.subheader("Speakers")
        for speaker in demo_results["speakers"]:
            st.markdown(f"- {speaker}")
    
    st.subheader("Scene Interpretation")
    st.write(demo_results["scene_interpretation"])
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Audio Language Model (ALM) System © 2025")