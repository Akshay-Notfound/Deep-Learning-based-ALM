"""
Main entry point for the Audio Language Model (ALM) application
Provides inference capabilities and example usage
"""

import torch
import yaml
import argparse
import os
import numpy as np
from data.datasets import ALMDataset
from models.alm_model import ALMModel
from utils.preprocessing import load_and_preprocess_audio, extract_mel_spectrogram
from utils.evaluation import print_evaluation_report


def load_model(config_path: str, checkpoint_path: str, device: torch.device):
    """
    Load trained ALM model from checkpoint
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (Loaded ALM model, config)
    """
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize model
    model = ALMModel(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    return model, config


def process_audio_file(model: ALMModel, config: dict, audio_path: str, 
                      question = None, device = None) -> dict:
    """
    Process a single audio file with the ALM model
    
    Args:
        model: Trained ALM model
        config: Configuration dictionary
        audio_path: Path to audio file
        question: Optional question for QA task
        device: Device to run inference on
        
    Returns:
        Dictionary of model outputs
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess audio
    sample_rate = config['data']['sample_rate']
    max_length = config['data']['max_audio_length'] * sample_rate
    
    audio, sr = load_and_preprocess_audio(audio_path, sample_rate, max_length)
    mel_spectrogram = extract_mel_spectrogram(
        audio, 
        sample_rate,
        n_mels=config['data']['n_mels'],
        n_fft=config['data']['n_fft'],
        hop_length=config['data']['hop_length']
    )
    
    # Convert to tensors
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)  # Add batch dimension
    mel_tensor = torch.FloatTensor(mel_spectrogram).unsqueeze(0).to(device)  # Add batch dimension
    
    # Prepare question tensor if provided
    question_tensor = None
    if question:
        # In a real implementation, you would tokenize the question
        # This is a placeholder
        question_tensor = torch.randn(1, 20, 1024).to(device)  # Batch=1, 20 tokens, 1024-dim
    
    # Run inference
    with torch.no_grad():
        outputs = model(
            audio=audio_tensor,
            mel_spectrogram=mel_tensor,
            question=question_tensor
        )
    
    return outputs


def format_alm_response(outputs: dict, config: dict) -> str:
    """
    Format model outputs into a human-readable response
    
    Args:
        outputs: Model outputs dictionary
        config: Configuration dictionary
        
    Returns:
        Formatted response string
    """
    response_parts = []
    
    # Speech recognition results
    if outputs.get('speech_logits') is not None:
        # In a real implementation, you would decode the logits to text
        response_parts.append("Speech Recognition: [Transcribed text would appear here]")
    
    # Audio event detection results
    if outputs.get('audio_event_logits') is not None:
        event_logits = outputs['audio_event_logits']
        # Get top events
        top_events = torch.topk(event_logits, k=3, dim=-1)
        event_names = [f"Event_{i}" for i in range(config['model']['output_dims']['audio_events'])]
        events = [event_names[idx] for idx in top_events.indices[0].cpu().numpy()]
        response_parts.append(f"Detected Audio Events: {', '.join(events)}")
    
    # Speaker diarization results
    if outputs.get('speaker_logits') is not None:
        speaker_logits = outputs['speaker_logits']
        num_speakers = torch.argmax(speaker_logits, dim=-1).unique().numel()
        response_parts.append(f"Identified Speakers: {num_speakers}")
    
    # Paralinguistic analysis results
    if outputs.get('paralinguistic_logits') is not None:
        paralinguistic_logits = outputs['paralinguistic_logits']
        # Get dominant emotion/tone
        dominant_trait = torch.argmax(paralinguistic_logits, dim=-1)
        trait_names = [f"Trait_{i}" for i in range(config['model']['output_dims']['paralinguistics'])]
        trait = trait_names[dominant_trait[0].item()]
        response_parts.append(f"Paralinguistic Analysis: {trait}")
    
    # Question answering results
    if outputs.get('qa_logits') is not None:
        response_parts.append("Answer to Question: [Generated answer would appear here]")
    
    return "\n".join(response_parts)


def main():
    """
    Main function for ALM inference
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Audio Language Model Inference')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to input audio file')
    parser.add_argument('--question', type=str, default=None,
                        help='Question to ask about the audio')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (cuda/cpu)')
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    try:
        model, config = load_model(args.config, args.checkpoint, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process audio file
    try:
        outputs = process_audio_file(model, config, args.audio, args.question, device)
    except Exception as e:
        print(f"Error processing audio: {e}")
        return
    
    # Format and print response
    response = format_alm_response(outputs, config)
    
    print("\n" + "="*60)
    print("ALM ANALYSIS RESULTS")
    print("="*60)
    print(f"Input Audio: {args.audio}")
    if args.question:
        print(f"Question: {args.question}")
    print("-"*60)
    print(response)
    print("="*60)


# Example usage and testing
def run_example():
    """
    Run example inference with dummy data
    """
    print("Running ALM example with dummy data...")
    
    # Dummy config
    config = {
        'data': {
            'sample_rate': 16000,
            'n_fft': 400,
            'hop_length': 160,
            'n_mels': 80,
            'max_audio_length': 30,
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
    
    # Create dummy model
    model = ALMModel(config)
    model.eval()
    
    # Create dummy audio data
    sample_rate = config['data']['sample_rate']
    duration = 5  # seconds
    dummy_audio = torch.randn(1, sample_rate * duration)  # Batch=1
    dummy_mel = torch.randn(1, config['data']['n_mels'], 300)  # Batch=1
    
    # Dummy question
    dummy_question = torch.randn(1, 20, 1024)  # Batch=1, 20 tokens
    
    # Run inference
    with torch.no_grad():
        outputs = model(
            audio=dummy_audio,
            mel_spectrogram=dummy_mel,
            question=dummy_question
        )
    
    # Format response
    response = format_alm_response(outputs, config)
    
    print("\n" + "="*60)
    print("ALM EXAMPLE RESULTS")
    print("="*60)
    print("Input: Dummy audio data")
    print("Question: Dummy question")
    print("-"*60)
    print(response)
    print("="*60)


if __name__ == '__main__':
    # If no arguments provided, run example
    import sys
    if len(sys.argv) == 1:
        run_example()
    else:
        main()