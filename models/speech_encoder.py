"""
Speech Encoder Module for the Audio Language Model (ALM)
Handles speech recognition and processing for multiple Asian languages
"""

import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Tuple, Dict, Any


class SpeechEncoder(nn.Module):
    """
    Speech encoder that processes audio signals and extracts linguistic features
    Supports multiple languages including Asian languages
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the speech encoder
        
        Args:
            config: Configuration dictionary
        """
        super(SpeechEncoder, self).__init__()
        
        self.config = config
        self.hidden_size = config['model']['speech_encoder']['hidden_size']
        self.num_layers = config['model']['speech_encoder']['num_layers']
        self.dropout = config['model']['speech_encoder']['dropout']
        
        # Use Wav2Vec2 as the base model for multilingual speech recognition
        # This model supports multiple languages including the Asian languages we need
        self.wav2vec_config = Wav2Vec2Config(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_layers,
            dropout=self.dropout,
            vocab_size=config['model']['output_dims']['speech_recognition']
        )
        
        # Initialize Wav2Vec2 model
        self.speech_model = Wav2Vec2Model(self.wav2vec_config)
        
        # Add language embedding layer
        self.language_embedding = nn.Embedding(
            num_embeddings=len(config['data']['languages']),
            embedding_dim=self.hidden_size
        )
        
        # Feature projection layer
        self.feature_projection = nn.Linear(
            self.hidden_size, 
            self.hidden_size
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, audio: torch.Tensor, language_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the speech encoder
        
        Args:
            audio: Input audio tensor of shape (batch_size, sequence_length)
            language_ids: Language identifiers for multilingual support
            
        Returns:
            Tuple of (encoded_features, pooled_features)
        """
        # Process audio through Wav2Vec2
        # Note: In practice, you would use a pre-trained multilingual model
        # and fine-tune it for your specific languages
        speech_outputs = self.speech_model(audio)
        
        # Extract last hidden states and pooled output
        hidden_states = speech_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        pooled_output = speech_outputs.pooler_output      # (batch_size, hidden_size)
        
        # Add language embeddings if provided
        if language_ids is not None:
            lang_embeds = self.language_embedding(language_ids)  # (batch_size, hidden_size)
            # Expand to match sequence length
            lang_embeds = lang_embeds.unsqueeze(1).expand(-1, hidden_states.size(1), -1)
            hidden_states = hidden_states + lang_embeds
        
        # Apply feature projection and dropout
        encoded_features = self.feature_projection(hidden_states)
        encoded_features = self.dropout_layer(encoded_features)
        
        pooled_features = self.feature_projection(pooled_output)
        pooled_features = self.dropout_layer(pooled_features)
        
        return encoded_features, pooled_features


class SpeechRecognitionHead(nn.Module):
    """
    Head for speech recognition task
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the speech recognition head
        
        Args:
            config: Configuration dictionary
        """
        super(SpeechRecognitionHead, self).__init__()
        
        self.hidden_size = config['model']['speech_encoder']['hidden_size']
        self.vocab_size = config['model']['output_dims']['speech_recognition']
        
        # Linear layer for token classification
        self.classifier = nn.Linear(self.hidden_size, self.vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(config['model']['speech_encoder']['dropout'])
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for speech recognition
        
        Args:
            features: Encoded features from speech encoder
            
        Returns:
            Logits for token classification
        """
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits


# Example usage
if __name__ == '__main__':
    # Dummy config for testing
    config = {
        'data': {
            'languages': ['english', 'mandarin', 'urdu', 'hindi', 'telugu', 'tamil', 'bangla']
        },
        'model': {
            'speech_encoder': {
                'hidden_size': 768,
                'num_layers': 12,
                'dropout': 0.1
            },
            'output_dims': {
                'speech_recognition': 50000
            }
        }
    }
    
    # Create speech encoder
    encoder = SpeechEncoder(config)
    head = SpeechRecognitionHead(config)
    
    # Test with dummy data
    batch_size, seq_length = 2, 16000  # 1 second of audio at 16kHz
    dummy_audio = torch.randn(batch_size, seq_length)
    dummy_lang_ids = torch.tensor([0, 1])  # Language IDs
    
    # Forward pass
    encoded_features, pooled_features = encoder(dummy_audio, dummy_lang_ids)
    logits = head(encoded_features)
    
    print(f"Encoded features shape: {encoded_features.shape}")
    print(f"Pooled features shape: {pooled_features.shape}")
    print(f"Logits shape: {logits.shape}")