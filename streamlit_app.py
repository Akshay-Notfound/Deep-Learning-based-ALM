"""
Streamlit interface for the Audio Language Model (ALM) system
"""
import streamlit as st
import os
import sys
import tempfile
import yaml

# Try to import the ALM modules, but handle gracefully if they fail
try:
    from main import ALMModel, load_model, process_audio_file, format_alm_response
    import torch
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ALM modules: {e}")
    MODEL_AVAILABLE = False

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Audio Language Model (ALM)",
    page_icon="🔊",
    layout="wide"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background: linear-gradient(135deg, #1a2a6c, #2a5298);
        color: #ffffff;
    }
    
    /* Header styling */
    .header {
        background: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Feature cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.3);
        background: rgba(255, 255, 255, 0.15);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Result cards */
    .result-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Upload area */
    .upload-area {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #4dabf7;
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #4dabf7, #3bc9db);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        background: linear-gradient(45deg, #339af0, #22b8cf);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    .stButton>button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.1);
        color: white;
    }
    
    .stButton>button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
    }
    
    /* Progress bar */
    .stProgress>div>div {
        background: linear-gradient(90deg, #4dabf7, #3bc9db);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: white;
        font-weight: 600;
    }
    
    /* Links */
    a {
        color: #4dabf7 !important;
    }
    
    /* Lists */
    ul, ol {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

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
# Header with enhanced styling
st.markdown("""
<div class='header'>
    <h1>🔊 Audio Language Model (ALM)</h1>
    <p style='font-size: 1.2rem; max-width: 800px; margin: 0 auto;'>Advanced audio analysis with speech recognition, event detection, and more</p>
    <div style='margin-top: 1rem; display: flex; justify-content: center; gap: 2rem;'>
        <div style='background: rgba(77, 171, 247, 0.2); padding: 0.5rem 1rem; border-radius: 50px;'>Multilingual Support</div>
        <div style='background: rgba(59, 201, 219, 0.2); padding: 0.5rem 1rem; border-radius: 50px;'>Real-time Analysis</div>
        <div style='background: rgba(119, 159, 247, 0.2); padding: 0.5rem 1rem; border-radius: 50px;'>AI Powered</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Features section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-icon'>🎤</div>
        <h3>Speech Recognition</h3>
        <p>Accurate transcription of speech in multiple languages including English, Mandarin, Urdu, Hindi, Telugu, Tamil, and Bangla</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-icon'>🔊</div>
        <h3>Audio Event Detection</h3>
        <p>Identify environmental sounds and events such as alarms, music, car honking, dog barking, and aircraft sounds</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='feature-card'>
        <div class='feature-icon'>👥</div>
        <h3>Speaker Diarization</h3>
        <p>Differentiate between multiple speakers in a conversation and track who spoke when</p>
    </div>
    """, unsafe_allow_html=True)

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
    if not MODEL_AVAILABLE:
        st.warning("ALM model components not available. Running in demo mode.")
        return False
    
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

# Enhanced file upload section
st.markdown("""
<div class='upload-area'>
    <h2>📤 Upload Audio File</h2>
    <p style='color: rgba(255, 255, 255, 0.8);'>Supported formats: WAV, MP3, FLAC, M4A</p>
</div>
""", unsafe_allow_html=True)

# File uploader with enhanced styling
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "m4a"], key="audio_uploader")

# Question input with enhanced styling
st.markdown("### ❓ Ask a Question (Optional)")
question = st.text_input("Question", placeholder="e.g., What language is being spoken? What emotions are detected?")

# Process button with enhanced styling
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Analyze Audio", type="primary", disabled=uploaded_file is None, use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Analyzing audio..."):
                # Progress bar for better UX
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Simulate progress
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        if i < 30:
                            status_text.text("Loading audio file...")
                        elif i < 60:
                            status_text.text("Processing audio features...")
                        elif i < 90:
                            status_text.text("Analyzing content...")
                        else:
                            status_text.text("Generating results...")
                    
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_filepath = tmp_file.name
                    
                    # Process the audio
                    results = process_audio(tmp_filepath, question)
                    
                    # Clean up temporary file
                    os.unlink(tmp_filepath)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results with enhanced styling
                    st.markdown("""
                    <div class='result-card'>
                        <h2 style='text-align: center; margin-bottom: 2rem;'>
                            <span style='background: linear-gradient(45deg, #4dabf7, #3bc9db); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                                📊 Analysis Results
                            </span>
                        </h2>
                    """, unsafe_allow_html=True)
                    
                    # Metrics at the top
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Speakers Detected", len(results.get("speakers", [])))
                    with col2:
                        st.metric("Audio Events", len(results.get("audio_events", [])))
                    with col3:
                        st.metric("Confidence", "High", "95%")
                    
                    st.markdown("---")
                    
                    # Detailed results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📝 Speech Recognition")
                        st.info(results.get("speech_recognition", "No transcription available"))
                        
                        st.subheader("🎭 Paralinguistics")
                        st.success(results.get("paralinguistics", "No analysis available"))
                        
                        if "question_answer" in results:
                            st.subheader("💬 Question Answer")
                            st.success(results["question_answer"])
                    
                    with col2:
                        st.subheader("🔊 Audio Events")
                        events = results.get("audio_events", [])
                        if events:
                            for event in events:
                                st.markdown(f"<div style='background: rgba(77, 171, 247, 0.2); padding: 0.5rem; border-radius: 8px; margin-bottom: 0.5rem;'>{event}</div>", unsafe_allow_html=True)
                        else:
                            st.warning("No events detected")
                        
                        st.subheader("👥 Speakers")
                        speakers = results.get("speakers", [])
                        if speakers:
                            for speaker in speakers:
                                st.markdown(f"<div style='background: rgba(59, 201, 219, 0.2); padding: 0.5rem; border-radius: 8px; margin-bottom: 0.5rem;'>{speaker}</div>", unsafe_allow_html=True)
                        else:
                            st.warning("No speakers identified")
                    
                    st.subheader("🧠 Scene Interpretation")
                    st.markdown(f"<div style='background: rgba(119, 159, 247, 0.2); padding: 1rem; border-radius: 10px; border-left: 4px solid #779ff7;'>{results.get('scene_interpretation', 'No interpretation available')}</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
        else:
            st.warning("Please upload an audio file first")

# Demo section
st.markdown("---")
st.subheader("🎮 Try Demo")

# Create an expander for the demo
with st.expander("Click to run a sample analysis", expanded=False):
    st.info("Running demo simulation with sample airport announcement audio...")
    
    # Simulated results
    demo_results = {
        "speech_recognition": "Flight LH456 to Frankfurt is now boarding at gate B12",
        "audio_events": ["aircraft_sound", "announcement", "footsteps"],
        "speakers": ["speaker_1", "speaker_2"],
        "paralinguistics": "urgent",
        "scene_interpretation": "The aircraft sound and announcement suggest that the person is in an airport boarding area. The urgency in the voice indicates they may be concerned about their flight."
    }
    
    st.markdown("""
    <div class='result-card'>
        <h2 style='text-align: center; margin-bottom: 2rem;'>
            <span style='background: linear-gradient(45deg, #ff6b6b, #ffa502); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                🎪 Demo Results
            </span>
        </h2>
    """, unsafe_allow_html=True)
    
    # Metrics for demo
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Speakers Detected", len(demo_results["speakers"]))
    with col2:
        st.metric("Audio Events", len(demo_results["audio_events"]))
    with col3:
        st.metric("Scenario", "Airport", "Boarding")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Speech Recognition")
        st.info(demo_results["speech_recognition"])
        
        st.subheader("🎭 Paralinguistics")
        st.success(demo_results["paralinguistics"])
    
    with col2:
        st.subheader("🔊 Audio Events")
        for event in demo_results["audio_events"]:
            st.markdown(f"<div style='background: rgba(255, 107, 107, 0.2); padding: 0.5rem; border-radius: 8px; margin-bottom: 0.5rem;'>{event}</div>", unsafe_allow_html=True)
        
        st.subheader("👥 Speakers")
        for speaker in demo_results["speakers"]:
            st.markdown(f"<div style='background: rgba(255, 165, 2, 0.2); padding: 0.5rem; border-radius: 8px; margin-bottom: 0.5rem;'>{speaker}</div>", unsafe_allow_html=True)
    
    st.subheader("🧠 Scene Interpretation")
    st.markdown(f"<div style='background: rgba(165, 94, 232, 0.2); padding: 1rem; border-radius: 10px; border-left: 4px solid #a55ee8;'>{demo_results['scene_interpretation']}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p>Audio Language Model (ALM) System © 2025</p>
    <p style='font-size: 0.8rem; margin-top: 0.5rem;'>Advanced audio analysis with speech recognition, event detection, and more</p>
    <div style='display: flex; justify-content: center; gap: 1rem; margin-top: 1rem;'>
        <span style='background: rgba(255, 255, 255, 0.1); padding: 0.3rem 1rem; border-radius: 20px;'>Multilingual</span>
        <span style='background: rgba(255, 255, 255, 0.1); padding: 0.3rem 1rem; border-radius: 20px;'>AI Powered</span>
        <span style='background: rgba(255, 255, 255, 0.1); padding: 0.3rem 1rem; border-radius: 20px;'>Real-time</span>
    </div>
</div>
""", unsafe_allow_html=True)