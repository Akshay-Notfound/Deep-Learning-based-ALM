"""
Audio Encoder Module for the Audio Language Model (ALM)
Handles non-speech audio event detection and environmental sound understanding
"""

import torch
import torch.nn as nn
import torchaudio
import torchvision.models as models
from typing import Tuple, Dict, Any


class AudioEncoder(nn.Module):
    """
    Audio encoder that processes mel spectrograms and extracts environmental sound features
    Handles audio event detection, speaker diarization, and paralinguistic analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the audio encoder
        
        Args:
            config: Configuration dictionary
        """
        super(AudioEncoder, self).__init__()
        
        self.config = config
        self.hidden_size = config['model']['audio_encoder']['hidden_size']
        self.num_layers = config['model']['audio_encoder']['num_layers']
        self.dropout = config['model']['audio_encoder']['dropout']
        
        # CNN layers for processing mel spectrograms
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling to fixed size
        )
        
        # Calculate the size after convolutions
        self.conv_output_size = 256 * 4 * 4  # 4096
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.conv_output_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Feature projection to match hidden size
        self.feature_projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, mel_spectrogram: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the audio encoder
        
        Args:
            mel_spectrogram: Input mel spectrogram tensor of shape (batch_size, n_mels, time_steps)
            
        Returns:
            Tuple of (encoded_features, pooled_features)
        """
        # Add channel dimension for CNN: (batch_size, 1, n_mels, time_steps)
        x = mel_spectrogram.unsqueeze(1)
        
        # Process through CNN layers
        cnn_features = self.conv_layers(x)
        
        # Reshape for LSTM: (batch_size, time_steps, features)
        batch_size, channels, height, width = cnn_features.shape
        lstm_input = cnn_features.view(batch_size, width, -1)
        
        # Process through LSTM
        lstm_output, (hidden, cell) = self.lstm(lstm_input)
        
        # Apply feature projection
        encoded_features = self.feature_projection(lstm_output)
        encoded_features = self.dropout_layer(encoded_features)
        
        # Pool the hidden states to get a single representation
        pooled_features = torch.mean(encoded_features, dim=1)
        pooled_features = self.dropout_layer(pooled_features)
        
        return encoded_features, pooled_features


class AudioEventHead(nn.Module):
    """
    Head for audio event detection task
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the audio event detection head
        
        Args:
            config: Configuration dictionary
        """
        super(AudioEventHead, self).__init__()
        
        self.hidden_size = config['model']['audio_encoder']['hidden_size']
        self.num_events = config['model']['output_dims']['audio_events']
        
        # Linear layer for event classification
        self.classifier = nn.Linear(self.hidden_size, self.num_events)
        
        # Dropout layer
        self.dropout = nn.Dropout(config['model']['audio_encoder']['dropout'])
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for audio event detection
        
        Args:
            features: Pooled features from audio encoder
            
        Returns:
            Logits for event classification
        """
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits


class SpeakerDiarizationHead(nn.Module):
    """
    Head for speaker diarization task
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the speaker diarization head
        
        Args:
            config: Configuration dictionary
        """
        super(SpeakerDiarizationHead, self).__init__()
        
        self.hidden_size = config['model']['audio_encoder']['hidden_size']
        self.max_speakers = config['model']['output_dims']['speaker_diarization']
        
        # Linear layer for speaker classification
        self.classifier = nn.Linear(self.hidden_size, self.max_speakers)
        
        # Dropout layer
        self.dropout = nn.Dropout(config['model']['audio_encoder']['dropout'])
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for speaker diarization
        
        Args:
            features: Encoded features from audio encoder
            
        Returns:
            Logits for speaker classification
        """
        # Apply to each time step for segmentation
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits


class ParalinguisticHead(nn.Module):
    """
    Head for paralinguistic analysis (emotion, tone, etc.)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the paralinguistic analysis head
        
        Args:
            config: Configuration dictionary
        """
        super(ParalinguisticHead, self).__init__()
        
        self.hidden_size = config['model']['audio_encoder']['hidden_size']
        self.num_paralinguistic_features = config['model']['output_dims']['paralinguistics']
        
        # Linear layer for paralinguistic classification
        self.classifier = nn.Linear(self.hidden_size, self.num_paralinguistic_features)
        
        # Dropout layer
        self.dropout = nn.Dropout(config['model']['audio_encoder']['dropout'])
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for paralinguistic analysis
        
        Args:
            features: Pooled features from audio encoder
            
        Returns:
            Logits for paralinguistic classification
        """
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits


# Example usage
if __name__ == '__main__':
    # Dummy config for testing
    config = {
        'model': {
            'audio_encoder': {
                'hidden_size': 768,
                'num_layers': 6,
                'dropout': 0.1
            },
            'output_dims': {
                'audio_events': 100,
                'speaker_diarization': 10,
                'paralinguistics': 20
            }
        }
    }
    
    # Create audio encoder and heads
    encoder = AudioEncoder(config)
    event_head = AudioEventHead(config)
    speaker_head = SpeakerDiarizationHead(config)
    paralinguistic_head = ParalinguisticHead(config)
    
    # Test with dummy data
    batch_size, n_mels, time_steps = 2, 80, 300  # 3 seconds of audio at 16kHz with 80 mel bands
    dummy_mel = torch.randn(batch_size, n_mels, time_steps)
    
    # Forward pass
    encoded_features, pooled_features = encoder(dummy_mel)
    event_logits = event_head(pooled_features)
    speaker_logits = speaker_head(encoded_features)
    paralinguistic_logits = paralinguistic_head(pooled_features)
    
    print(f"Encoded features shape: {encoded_features.shape}")
    print(f"Pooled features shape: {pooled_features.shape}")
    print(f"Event logits shape: {event_logits.shape}")
    print(f"Speaker logits shape: {speaker_logits.shape}")
    print(f"Paralinguistic logits shape: {paralinguistic_logits.shape}")