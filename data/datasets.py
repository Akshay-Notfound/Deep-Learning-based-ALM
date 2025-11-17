"""
Dataset module for the Audio Language Model (ALM)
Handles loading and preprocessing of multimodal audio data
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
from typing import Tuple, Dict, Any


class ALMDataset(Dataset):
    """
    Dataset class for Audio Language Model
    Handles speech, non-speech audio, and associated metadata
    """
    
    def __init__(self, data_dir: str, config: Dict[str, Any], split: str = 'train'):
        """
        Initialize the dataset
        
        Args:
            data_dir: Path to the data directory
            config: Configuration dictionary
            split: Data split ('train', 'val', 'test')
        """
        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.sample_rate = config['data']['sample_rate']
        self.max_length = config['data']['max_audio_length'] * self.sample_rate
        self.languages = config['data']['languages']
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> pd.DataFrame:
        """
        Load dataset metadata from CSV files
        
        Returns:
            DataFrame with metadata
        """
        metadata_path = os.path.join(self.data_dir, f'{self.split}_metadata.csv')
        if os.path.exists(metadata_path):
            return pd.read_csv(metadata_path)
        else:
            # Create dummy metadata for demonstration
            return self._create_dummy_metadata()
            
    def _create_dummy_metadata(self) -> pd.DataFrame:
        """
        Create dummy metadata for demonstration purposes
        
        Returns:
            DataFrame with dummy metadata
        """
        dummy_data = {
            'audio_path': [f'dummy_{i}.wav' for i in range(100)],
            'transcript': [f'Transcript {i}' for i in range(100)],
            'language': [self.languages[i % len(self.languages)] for i in range(100)],
            'speaker_ids': [f'speaker_{i % 10}' for i in range(100)],
            'audio_events': [f'event_{i % 5}' for i in range(100)],
            'emotions': [f'emotion_{i % 4}' for i in range(100)]
        }
        return pd.DataFrame(dummy_data)
        
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Preprocessed audio signal
        """
        try:
            # Load audio file
            audio, sr = sf.read(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                
            # Pad or truncate to max length
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            else:
                padding = self.max_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
                
            return audio
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return zeros if there's an error
            return np.zeros(self.max_length)
            
    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram from audio signal
        
        Args:
            audio: Audio signal
            
        Returns:
            Mel spectrogram
        """
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.config['data']['n_fft'],
            hop_length=self.config['data']['hop_length'],
            n_mels=self.config['data']['n_mels']
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
        
    def __len__(self) -> int:
        """Return the length of the dataset"""
        return len(self.metadata)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with input tensors and labels
        """
        # Get metadata for this item
        item = self.metadata.iloc[idx]
        
        # Load audio
        audio_path = os.path.join(self.data_dir, 'raw', item['audio_path'])
        audio = self._load_audio(audio_path)
        
        # Extract features
        mel_spectrogram = self._extract_mel_spectrogram(audio)
        
        # Prepare targets
        # In a real implementation, you would convert text to token IDs
        # For now, we'll create dummy targets
        speech_target = torch.randint(0, self.config['model']['output_dims']['speech_recognition'], (100,))
        audio_event_target = torch.randint(0, self.config['model']['output_dims']['audio_events'], (1,))
        speaker_target = torch.randint(0, self.config['model']['output_dims']['speaker_diarization'], (5,))
        paralinguistic_target = torch.randint(0, self.config['model']['output_dims']['paralinguistics'], (1,))
        
        return {
            'audio': torch.FloatTensor(audio),
            'mel_spectrogram': torch.FloatTensor(mel_spectrogram),
            'speech_target': speech_target,
            'audio_event_target': audio_event_target,
            'speaker_target': speaker_target,
            'paralinguistic_target': paralinguistic_target,
            'language': item['language']
        }


def get_data_loader(data_dir: str, config: Dict[str, Any], split: str = 'train', 
                   shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for the ALM dataset
    
    Args:
        data_dir: Path to the data directory
        config: Configuration dictionary
        split: Data split ('train', 'val', 'test')
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader instance
    """
    dataset = ALMDataset(data_dir, config, split)
    batch_size = config['training']['batch_size']
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )


# Example usage
if __name__ == '__main__':
    # Dummy config for testing
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
            'output_dims': {
                'speech_recognition': 50000,
                'audio_events': 100,
                'speaker_diarization': 10,
                'paralinguistics': 20
            }
        },
        'training': {
            'batch_size': 4
        }
    }
    
    # Create data loader
    data_loader = get_data_loader('data', config, 'train')
    
    # Test loading a batch
    for batch in data_loader:
        print("Batch keys:", batch.keys())
        print("Audio shape:", batch['audio'].shape)
        print("Mel spectrogram shape:", batch['mel_spectrogram'].shape)
        break