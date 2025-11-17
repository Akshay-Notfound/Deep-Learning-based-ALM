"""
Fusion Module for the Audio Language Model (ALM)
Combines speech and audio features for joint understanding
"""

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Tuple, Dict, Any


class FusionModule(nn.Module):
    """
    Fusion module that combines speech and audio features using transformer-based attention
    Enables joint understanding of speech and non-speech elements
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the fusion module
        
        Args:
            config: Configuration dictionary
        """
        super(FusionModule, self).__init__()
        
        self.config = config
        self.hidden_size = config['model']['fusion_module']['hidden_size']
        self.num_heads = config['model']['fusion_module']['num_heads']
        self.num_layers = config['model']['fusion_module']['num_layers']
        self.dropout = config['model']['fusion_module'].get('dropout', 0.1)
        
        # Projection layers to match hidden dimensions
        self.speech_projection = nn.Linear(
            config['model']['speech_encoder']['hidden_size'], 
            self.hidden_size
        )
        
        self.audio_projection = nn.Linear(
            config['model']['audio_encoder']['hidden_size'], 
            self.hidden_size
        )
        
        # Positional encoding for sequence modeling
        self.positional_encoding = PositionalEncoding(self.hidden_size, self.dropout)
        
        # Transformer encoder for multimodal fusion
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            batch_first=True
        )
        
        self.transformer_encoder = TransformerEncoder(encoder_layer, self.num_layers)
        
        # Cross-modal attention layers
        self.cross_attention_speech_to_audio = CrossModalAttention(self.hidden_size, self.num_heads)
        self.cross_attention_audio_to_speech = CrossModalAttention(self.hidden_size, self.num_heads)
        
        # Modality fusion layer
        self.fusion_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
    def forward(self, speech_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the fusion module
        
        Args:
            speech_features: Features from speech encoder (batch_size, seq_len, hidden_size)
            audio_features: Features from audio encoder (batch_size, seq_len, hidden_size)
            
        Returns:
            Fused multimodal features
        """
        # Project features to common dimension
        projected_speech = self.speech_projection(speech_features)
        projected_audio = self.audio_projection(audio_features)
        
        # Apply cross-modal attention
        speech_enhanced = self.cross_attention_speech_to_audio(projected_speech, projected_audio)
        audio_enhanced = self.cross_attention_audio_to_speech(projected_audio, projected_speech)
        
        # Combine enhanced features
        combined_features = torch.cat([speech_enhanced, audio_enhanced], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        
        # Apply positional encoding
        fused_features = self.positional_encoding(fused_features)
        
        # Apply transformer encoder for deep fusion
        fused_features = self.transformer_encoder(fused_features)
        
        # Apply layer normalization
        fused_features = self.layer_norm(fused_features)
        
        return fused_features


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence modeling
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize positional encoding
        
        Args:
            d_model: Dimension of the model
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for inter-modal interaction
    """
    
    def __init__(self, hidden_size: int, num_heads: int):
        """
        Initialize cross-modal attention
        
        Args:
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads
        """
        super(CrossModalAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Ensure hidden_size is divisible by num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for cross-modal attention
        
        Args:
            query: Query tensor (batch_size, seq_len_q, hidden_size)
            key_value: Key and value tensor (batch_size, seq_len_kv, hidden_size)
            
        Returns:
            Attention output tensor
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_size
        )
        
        # Output projection
        output = self.out_proj(attended_values)
        
        return output


# Example usage
if __name__ == '__main__':
    # Dummy config for testing
    config = {
        'model': {
            'fusion_module': {
                'hidden_size': 1024,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.1
            },
            'speech_encoder': {
                'hidden_size': 768
            },
            'audio_encoder': {
                'hidden_size': 768
            }
        }
    }
    
    # Create fusion module
    fusion_module = FusionModule(config)
    
    # Test with dummy data
    batch_size, seq_len, speech_hidden, audio_hidden = 2, 100, 768, 768
    dummy_speech = torch.randn(batch_size, seq_len, speech_hidden)
    dummy_audio = torch.randn(batch_size, seq_len, audio_hidden)
    
    # Forward pass
    fused_features = fusion_module(dummy_speech, dummy_audio)
    
    print(f"Speech features shape: {dummy_speech.shape}")
    print(f"Audio features shape: {dummy_audio.shape}")
    print(f"Fused features shape: {fused_features.shape}")