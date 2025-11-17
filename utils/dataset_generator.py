"""
Dataset generation utilities for the Audio Language Model (ALM)
Handles creation of synthetic datasets for training the ALM model
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from typing import Dict, List, Tuple
import json
from tqdm import tqdm


def generate_synthetic_audio_dataset(output_dir: str, config: Dict, 
                                   num_samples: int = 1000) -> None:
    """
    Generate synthetic audio dataset for ALM training
    
    Args:
        output_dir: Directory to save the dataset
        config: Configuration dictionary
        num_samples: Number of samples to generate
    """
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
    
    # Sample rate and duration from config
    sample_rate = config['data']['sample_rate']
    max_duration = config['data']['max_audio_length']
    
    # Supported languages
    languages = config['data']['languages']
    
    # Audio event types
    audio_events = [f'event_{i}' for i in range(config['model']['output_dims']['audio_events'])]
    
    # Speaker IDs
    speaker_ids = [f'speaker_{i}' for i in range(config['model']['output_dims']['speaker_diarization'])]
    
    # Paralinguistic traits
    paralinguistic_traits = [f'trait_{i}' for i in range(config['model']['output_dims']['paralinguistics'])]
    
    # Generate dataset metadata
    metadata = []
    
    print(f"Generating {num_samples} synthetic audio samples...")
    
    for i in tqdm(range(num_samples)):
        # Generate synthetic audio
        duration = np.random.uniform(1.0, max_duration)
        audio_length = int(duration * sample_rate)
        
        # Create composite audio signal
        audio = generate_composite_audio(sample_rate, duration)
        
        # Save audio file
        audio_filename = f'sample_{i:05d}.wav'
        audio_path = os.path.join(output_dir, 'raw', audio_filename)
        sf.write(audio_path, audio, sample_rate)
        
        # Generate metadata
        language = np.random.choice(languages)
        speaker = np.random.choice(speaker_ids)
        
        # Generate multiple speakers for diarization
        num_speakers = np.random.randint(1, min(4, len(speaker_ids)))
        speakers_in_sample = np.random.choice(speaker_ids, num_speakers, replace=False)
        speakers_str = ','.join(speakers_in_sample)
        
        # Generate audio events
        num_events = np.random.randint(1, 4)
        events_in_sample = np.random.choice(audio_events, num_events, replace=False)
        events_str = ','.join(events_in_sample)
        
        # Paralinguistic traits
        trait = np.random.choice(paralinguistic_traits)
        
        # Generate synthetic transcript
        transcript = generate_synthetic_transcript(language)
        
        # Add to metadata
        metadata.append({
            'audio_path': audio_filename,
            'transcript': transcript,
            'language': language,
            'speakers': speakers_str,
            'audio_events': events_str,
            'paralinguistic_trait': trait,
            'duration': duration
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    
    # Split into train/val/test
    train_split = int(0.8 * num_samples)
    val_split = int(0.9 * num_samples)
    
    train_metadata = metadata_df[:train_split]
    val_metadata = metadata_df[train_split:val_split]
    test_metadata = metadata_df[val_split:]
    
    # Save splits
    train_metadata.to_csv(os.path.join(output_dir, 'train_metadata.csv'), index=False)
    val_metadata.to_csv(os.path.join(output_dir, 'val_metadata.csv'), index=False)
    test_metadata.to_csv(os.path.join(output_dir, 'test_metadata.csv'), index=False)
    
    print(f"Dataset generated successfully!")
    print(f"  - Train samples: {len(train_metadata)}")
    print(f"  - Validation samples: {len(val_metadata)}")
    print(f"  - Test samples: {len(test_metadata)}")


def generate_composite_audio(sample_rate: int, duration: float) -> np.ndarray:
    """
    Generate composite audio signal with speech and non-speech elements
    
    Args:
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        
    Returns:
        Composite audio signal
    """
    total_samples = int(duration * sample_rate)
    audio = np.zeros(total_samples)
    
    # Add background noise
    noise_level = np.random.uniform(0.01, 0.1)
    audio += noise_level * np.random.randn(total_samples)
    
    # Add speech segments
    num_speech_segments = np.random.randint(1, 5)
    for _ in range(num_speech_segments):
        # Speech parameters
        start_time = np.random.uniform(0, duration - 0.5)
        segment_duration = np.random.uniform(0.5, min(3.0, duration - start_time))
        
        start_sample = int(start_time * sample_rate)
        end_sample = int((start_time + segment_duration) * sample_rate)
        
        # Generate speech-like signal (formants)
        speech_segment = generate_speech_segment(segment_duration, sample_rate)
        speech_level = np.random.uniform(0.3, 0.8)
        
        # Add to audio
        if end_sample <= total_samples:
            audio[start_sample:end_sample] += speech_level * speech_segment
    
    # Add non-speech events
    num_events = np.random.randint(0, 3)
    for _ in range(num_events):
        event_type = np.random.choice(['tone', 'noise', 'click'])
        event_time = np.random.uniform(0, duration)
        event_sample = int(event_time * sample_rate)
        
        if event_type == 'tone':
            # Add tone
            freq = np.random.uniform(200, 2000)
            tone_duration = np.random.uniform(0.1, 0.5)
            tone_samples = int(tone_duration * sample_rate)
            if event_sample + tone_samples <= total_samples:
                t = np.linspace(0, tone_duration, tone_samples)
                tone = np.sin(2 * np.pi * freq * t)
                audio[event_sample:event_sample + tone_samples] += 0.5 * tone
                
        elif event_type == 'noise':
            # Add noise burst
            noise_duration = np.random.uniform(0.05, 0.2)
            noise_samples = int(noise_duration * sample_rate)
            if event_sample + noise_samples <= total_samples:
                noise = np.random.randn(noise_samples)
                audio[event_sample:event_sample + noise_samples] += 0.4 * noise
                
        elif event_type == 'click':
            # Add click
            if event_sample < total_samples:
                audio[event_sample] = 1.0
                if event_sample + 1 < total_samples:
                    audio[event_sample + 1] = -1.0
    
    # Normalize audio
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp * 0.9  # Prevent clipping
    
    return audio


def generate_speech_segment(duration: float, sample_rate: int) -> np.ndarray:
    """
    Generate speech-like audio segment
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Speech-like audio segment
    """
    num_samples = int(duration * sample_rate)
    
    # Generate formants (simplified model of vocal tract)
    t = np.linspace(0, duration, num_samples)
    
    # Fundamental frequency (varies for different speakers)
    f0 = np.random.uniform(80, 250)
    
    # Formant frequencies (simplified)
    f1 = np.random.uniform(200, 800)   # First formant
    f2 = np.random.uniform(800, 2500)  # Second formant
    f3 = np.random.uniform(2000, 3500) # Third formant
    
    # Generate voiced component
    voiced = (np.sin(2 * np.pi * f0 * t) + 
              0.5 * np.sin(2 * np.pi * f1 * t) + 
              0.3 * np.sin(2 * np.pi * f2 * t) + 
              0.2 * np.sin(2 * np.pi * f3 * t))
    
    # Add some noise for unvoiced components
    noise = 0.1 * np.random.randn(num_samples)
    
    # Combine voiced and unvoiced
    speech = voiced + noise
    
    # Apply simple envelope to simulate speech rhythm
    envelope = np.ones(num_samples)
    if num_samples > 100:
        # Create simple amplitude modulation
        mod_freq = np.random.uniform(2, 8)  # Modulation frequency
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t[:len(envelope)])
    
    speech = speech * envelope
    
    return speech


def generate_synthetic_transcript(language: str) -> str:
    """
    Generate synthetic transcript for a given language
    
    Args:
        language: Language identifier
        
    Returns:
        Synthetic transcript
    """
    # Word lists for different languages (simplified)
    word_lists = {
        'english': ['hello', 'world', 'audio', 'language', 'model', 'deep', 'learning', 'speech', 'recognition', 'system'],
        'mandarin': ['你好', '世界', '音频', '语言', '模型', '深度', '学习', '语音', '识别', '系统'],
        'urdu': ['ہیلو', 'دنیا', 'آڈیو', 'زبان', 'ماڈل', 'گہری', 'سیکھنا', 'تقریر', 'شناخت', 'سسٹم'],
        'hindi': ['नमस्ते', 'दुनिया', 'ऑडियो', 'भाषा', 'मॉडल', 'गहरी', 'सीखना', 'भाषण', 'पहचान', 'प्रणाली'],
        'telugu': ['హలో', 'ప్రపంచం', 'ఆడియో', 'భాష', 'మోడల్', 'లోతైన', 'నేర్చుకోవడం', 'ప్రసంగం', 'గుర్తింపబడింది', 'వ్యవస్థ'],
        'tamil': ['ஹலோ', 'உலகம்', 'ஆடியோ', 'மொழி', 'மாதிரி', 'ஆழமான', 'கற்றல்', 'பேச்சு', 'அடையாளம்', 'அமைப்பு'],
        'bangla': ['হ্যালো', 'বিশ্ব', 'অডিও', 'ভাষা', 'মডেল', 'গভীর', 'শেখা', 'বক্তৃতা', 'সনাক্তকরণ', 'সিস্টেম']
    }
    
    # Select word list based on language
    words = word_lists.get(language, word_lists['english'])
    
    # Generate random transcript
    num_words = np.random.randint(3, 10)
    transcript_words = np.random.choice(words, num_words, replace=True)
    
    # Join words based on language
    if language in ['english']:
        transcript = ' '.join(transcript_words)
    else:
        transcript = ''.join(transcript_words)
    
    return transcript


def create_dataset_statistics(dataset_dir: str) -> Dict:
    """
    Create dataset statistics and save to JSON file
    
    Args:
        dataset_dir: Directory containing the dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    # Load metadata
    train_meta = pd.read_csv(os.path.join(dataset_dir, 'train_metadata.csv'))
    val_meta = pd.read_csv(os.path.join(dataset_dir, 'val_metadata.csv'))
    test_meta = pd.read_csv(os.path.join(dataset_dir, 'test_metadata.csv'))
    
    # Compute statistics
    stats = {
        'total_samples': len(train_meta) + len(val_meta) + len(test_meta),
        'train_samples': len(train_meta),
        'val_samples': len(val_meta),
        'test_samples': len(test_meta),
        'languages': train_meta['language'].value_counts().to_dict(),
        'average_duration': train_meta['duration'].mean(),
        'min_duration': train_meta['duration'].min(),
        'max_duration': train_meta['duration'].max()
    }
    
    # Save statistics
    stats_path = os.path.join(dataset_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


# Example usage
if __name__ == '__main__':
    # Dummy config for testing
    config = {
        'data': {
            'sample_rate': 16000,
            'max_audio_length': 30,
            'languages': ['english', 'mandarin', 'urdu', 'hindi', 'telugu', 'tamil', 'bangla']
        },
        'model': {
            'output_dims': {
                'audio_events': 100,
                'speaker_diarization': 10,
                'paralinguistics': 20
            }
        }
    }
    
    # Generate small dataset for testing
    print("Generating synthetic dataset for testing...")
    generate_synthetic_audio_dataset('data', config, num_samples=50)
    
    # Create statistics
    stats = create_dataset_statistics('data')
    print("Dataset statistics:", stats)