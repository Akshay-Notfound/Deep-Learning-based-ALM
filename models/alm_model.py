"""
Main Audio Language Model (ALM) Implementation
Integrates speech encoder, audio encoder, and fusion module for comprehensive audio understanding
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from .speech_encoder import SpeechEncoder, SpeechRecognitionHead
from .audio_encoder import AudioEncoder, AudioEventHead, SpeakerDiarizationHead, ParalinguisticHead
from .fusion_module import FusionModule


class ALMModel(nn.Module):
    """
    Audio Language Model that jointly understands speech and non-speech audio elements
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ALM model
        
        Args:
            config: Configuration dictionary
        """
        super(ALMModel, self).__init__()
        
        self.config = config
        
        # Initialize component models
        self.speech_encoder = SpeechEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.fusion_module = FusionModule(config)
        
        # Initialize task-specific heads
        self.speech_recognition_head = SpeechRecognitionHead(config)
        self.audio_event_head = AudioEventHead(config)
        self.speaker_diarization_head = SpeakerDiarizationHead(config)
        self.paralinguistic_head = ParalinguisticHead(config)
        
        # Question answering head for complex reasoning
        self.qa_head = QuestionAnsweringHead(config)
        
    def forward(self, audio: torch.Tensor, mel_spectrogram: torch.Tensor, 
                question: torch.Tensor = None, language_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete ALM model
        
        Args:
            audio: Raw audio waveform tensor (batch_size, sequence_length)
            mel_spectrogram: Mel spectrogram tensor (batch_size, n_mels, time_steps)
            question: Optional question tensor for QA task (batch_size, question_length)
            language_ids: Language identifiers for multilingual support
            
        Returns:
            Dictionary of output logits for all tasks
        """
        # Process speech
        speech_features, speech_pooled = self.speech_encoder(audio, language_ids)
        speech_logits = self.speech_recognition_head(speech_features)
        
        # Process audio
        audio_features, audio_pooled = self.audio_encoder(mel_spectrogram)
        event_logits = self.audio_event_head(audio_pooled)
        speaker_logits = self.speaker_diarization_head(audio_features)
        paralinguistic_logits = self.paralinguistic_head(audio_pooled)
        
        # Fuse modalities
        fused_features = self.fusion_module(speech_features, audio_features)
        
        # Global pooling of fused features
        fused_pooled = torch.mean(fused_features, dim=1)
        
        # Question answering if question is provided
        qa_logits = None
        if question is not None:
            qa_logits = self.qa_head(fused_pooled, question)
        
        return {
            'speech_logits': speech_logits,
            'audio_event_logits': event_logits,
            'speaker_logits': speaker_logits,
            'paralinguistic_logits': paralinguistic_logits,
            'qa_logits': qa_logits,
            'fused_features': fused_features,
            'speech_features': speech_features,
            'audio_features': audio_features
        }


class QuestionAnsweringHead(nn.Module):
    """
    Head for question answering task that enables complex reasoning
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the question answering head
        
        Args:
            config: Configuration dictionary
        """
        super(QuestionAnsweringHead, self).__init__()
        
        self.hidden_size = config['model']['fusion_module']['hidden_size']
        self.dropout = config['model']['fusion_module'].get('dropout', 0.1)
        
        # Linear layers for processing fused features and questions
        self.fused_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.question_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Attention mechanism for question-aware feature selection
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 1024)  # Fixed output size for text generation
        )
        
    def forward(self, fused_features: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for question answering
        
        Args:
            fused_features: Fused multimodal features (batch_size, hidden_size)
            question: Question embeddings (batch_size, question_length, hidden_size)
            
        Returns:
            Answer logits
        """
        # Project features
        fused_proj = self.fused_projection(fused_features)
        question_proj = self.question_projection(question)
        
        # Add fused features to question as context
        fused_expanded = fused_proj.unsqueeze(1).expand(-1, question.size(1), -1)
        
        # Apply attention
        attended_features, _ = self.attention(
            query=question_proj,
            key=fused_expanded,
            value=fused_expanded
        )
        
        # Combine attended features with original question
        combined = torch.cat([attended_features, question_proj], dim=-1)
        
        # Apply classifier
        logits = self.classifier(combined)
        
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
    
    # Create ALM model
    model = ALMModel(config)
    
    # Test with dummy data
    batch_size, audio_len, n_mels, time_steps = 2, 48000, 80, 300
    dummy_audio = torch.randn(batch_size, audio_len)
    dummy_mel = torch.randn(batch_size, n_mels, time_steps)
    dummy_question = torch.randn(batch_size, 20, 1024)  # 20 tokens with 1024-dim embeddings
    dummy_lang_ids = torch.tensor([0, 1])
    
    # Forward pass
    outputs = model(
        audio=dummy_audio,
        mel_spectrogram=dummy_mel,
        question=dummy_question,
        language_ids=dummy_lang_ids
    )
    
    print("Model outputs:")
    for key, value in outputs.items():
        if value is not None:
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: None")