"""
Preprocessing utilities for the Audio Language Model (ALM)
Handles audio preprocessing, feature extraction, and data augmentation
"""

import numpy as np
import librosa
import soundfile as sf
import torch
from typing import Tuple, Dict, Any


def load_and_preprocess_audio(audio_path: str, sample_rate: int = 16000, 
                             max_length: int = None) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file
    
    Args:
        audio_path: Path to the audio file
        sample_rate: Target sample rate
        max_length: Maximum length in samples (None for no limit)
        
    Returns:
        Tuple of (audio_signal, sample_rate)
    """
    # Load audio file
    audio, sr = sf.read(audio_path)
    
    # Resample if necessary
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    
    # Pad or truncate to max length
    if max_length is not None:
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            padding = max_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
    
    return audio, sr


def extract_mel_spectrogram(audio: np.ndarray, sample_rate: int, 
                           n_mels: int = 80, n_fft: int = 400, 
                           hop_length: int = 160) -> np.ndarray:
    """
    Extract mel spectrogram from audio signal
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate of the audio
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        Mel spectrogram
    """
    # Convert to mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec


def augment_audio(audio: np.ndarray, sample_rate: int, 
                 augmentations: Dict[str, Any] = None) -> np.ndarray:
    """
    Apply data augmentation to audio signal
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate of the audio
        augmentations: Dictionary of augmentation parameters
        
    Returns:
        Augmented audio signal
    """
    if augmentations is None:
        augmentations = {}
    
    augmented = audio.copy()
    
    # Add noise
    if augmentations.get('add_noise', False):
        noise_factor = augmentations.get('noise_factor', 0.005)
        noise = np.random.randn(len(augmented))
        augmented = augmented + noise_factor * noise
    
    # Time shift
    if augmentations.get('time_shift', False):
        shift_max = augmentations.get('shift_max', 0.1)
        shift = int(sample_rate * shift_max * (2 * np.random.rand() - 1))
        if shift > 0:
            augmented[:-shift] = augmented[shift:]
            augmented[-shift:] = 0
        elif shift < 0:
            augmented[-shift:] = augmented[:shift]
            augmented[:-shift] = 0
    
    # Pitch shift (for speech)
    if augmentations.get('pitch_shift', False):
        n_steps = augmentations.get('pitch_steps', 2) * (2 * np.random.rand() - 1)
        augmented = librosa.effects.pitch_shift(augmented, sr=sample_rate, n_steps=n_steps)
    
    # Time stretching
    if augmentations.get('time_stretch', False):
        rate = 1.0 + 0.1 * (2 * np.random.rand() - 1)  # 10% variation
        augmented = librosa.effects.time_stretch(augmented, rate=rate)
    
    return augmented


def convert_text_to_tokens(text: str, tokenizer=None, max_length: int = 100) -> torch.Tensor:
    """
    Convert text to token IDs (placeholder implementation)
    
    Args:
        text: Input text
        tokenizer: Tokenizer object (if available)
        max_length: Maximum sequence length
        
    Returns:
        Token IDs tensor
    """
    # In a real implementation, you would use a proper tokenizer
    # This is a placeholder that converts characters to indices
    char_to_idx = {char: idx for idx, char in enumerate(set(text))}
    tokens = [char_to_idx.get(char, 0) for char in text[:max_length]]
    
    # Pad to max_length
    if len(tokens) < max_length:
        tokens.extend([0] * (max_length - len(tokens)))
    
    return torch.tensor(tokens)


def language_to_id(language: str, language_list: list) -> int:
    """
    Convert language string to ID
    
    Args:
        language: Language name
        language_list: List of supported languages
        
    Returns:
        Language ID
    """
    try:
        return language_list.index(language.lower())
    except ValueError:
        return 0  # Default to first language


# Example usage
if __name__ == '__main__':
    # Test audio preprocessing
    print("Testing audio preprocessing functions...")
    
    # Create dummy audio signal
    sample_rate = 16000
    duration = 3  # seconds
    dummy_audio = np.random.randn(sample_rate * duration)
    
    # Extract mel spectrogram
    mel_spec = extract_mel_spectrogram(dummy_audio, sample_rate)
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    
    # Apply augmentation
    augmented = augment_audio(dummy_audio, sample_rate, {
        'add_noise': True,
        'time_shift': True,
        'pitch_shift': True
    })
    print(f"Original audio shape: {dummy_audio.shape}")
    print(f"Augmented audio shape: {augmented.shape}")
    
    # Test text conversion
    dummy_text = "Hello, this is a test."
    tokens = convert_text_to_tokens(dummy_text)
    print(f"Text tokens shape: {tokens.shape}")
    
    # Test language conversion
    languages = ['english', 'mandarin', 'urdu', 'hindi', 'telugu', 'tamil', 'bangla']
    lang_id = language_to_id('hindi', languages)
    print(f"Hindi language ID: {lang_id}")