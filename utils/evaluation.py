"""
Evaluation utilities for the Audio Language Model (ALM)
Handles model evaluation, metrics computation, and result visualization
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def compute_classification_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                                 average: str = 'macro') -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        predictions: Predicted logits or class indices
        targets: Ground truth labels
        average: Averaging method for multiclass metrics
        
    Returns:
        Dictionary of metrics
    """
    # Convert logits to predictions if necessary
    if predictions.dim() > 1 and predictions.size(-1) > 1:
        pred_classes = torch.argmax(predictions, dim=-1)
    else:
        pred_classes = predictions
    
    # Convert to numpy for sklearn
    y_pred = pred_classes.cpu().numpy().flatten()
    y_true = targets.cpu().numpy().flatten()
    
    # Filter out ignored indices (e.g., -100)
    mask = y_true != -100
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def compute_word_error_rate(predictions: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate (WER) for speech recognition
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Word Error Rate
    """
    total_words = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        # Simple implementation - in practice, you would use a library like jiwer
        pred_words = pred.split()
        ref_words = ref.split()
        
        total_words += len(ref_words)
        # Count substitutions, insertions, deletions (simplified)
        errors = sum(p != r for p, r in zip(pred_words, ref_words))
        errors += abs(len(pred_words) - len(ref_words))
        total_errors += errors
    
    return total_errors / max(total_words, 1) if total_words > 0 else 0.0


def compute_diarization_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute speaker diarization metrics
    
    Args:
        predictions: Predicted speaker labels (batch_size, seq_len)
        targets: Ground truth speaker labels (batch_size, seq_len)
        
    Returns:
        Dictionary of diarization metrics
    """
    # Convert to numpy
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # Compute DER (Diarization Error Rate) components
    # This is a simplified implementation
    correct = (pred_np == target_np).sum()
    total = pred_np.size
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'diarization_accuracy': accuracy,
        'diarization_error_rate': 1.0 - accuracy
    }


def evaluate_alm_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                      device: torch.device) -> Dict[str, Any]:
    """
    Evaluate the ALM model on a dataset
    
    Args:
        model: ALM model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_metrics = defaultdict(list)
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                audio=batch['audio'],
                mel_spectrogram=batch['mel_spectrogram']
            )
            
            # Compute metrics for each task
            if outputs['speech_logits'] is not None and 'speech_target' in batch:
                speech_metrics = compute_classification_metrics(
                    outputs['speech_logits'], 
                    batch['speech_target']
                )
                for k, v in speech_metrics.items():
                    all_metrics[f'speech_{k}'].append(v)
            
            if outputs['audio_event_logits'] is not None and 'audio_event_target' in batch:
                event_metrics = compute_classification_metrics(
                    outputs['audio_event_logits'], 
                    batch['audio_event_target']
                )
                for k, v in event_metrics.items():
                    all_metrics[f'event_{k}'].append(v)
            
            if outputs['speaker_logits'] is not None and 'speaker_target' in batch:
                speaker_metrics = compute_diarization_metrics(
                    torch.argmax(outputs['speaker_logits'], dim=-1),
                    batch['speaker_target']
                )
                for k, v in speaker_metrics.items():
                    all_metrics[f'speaker_{k}'].append(v)
            
            if outputs['paralinguistic_logits'] is not None and 'paralinguistic_target' in batch:
                paralinguistic_metrics = compute_classification_metrics(
                    outputs['paralinguistic_logits'], 
                    batch['paralinguistic_target']
                )
                for k, v in paralinguistic_metrics.items():
                    all_metrics[f'paralinguistic_{k}'].append(v)
    
    # Average metrics across batches
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    return avg_metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], title: str = "Confusion Matrix"):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()


def print_evaluation_report(metrics: Dict[str, float]) -> None:
    """
    Print formatted evaluation report
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("ALM Model Evaluation Report")
    print("="*50)
    
    # Group metrics by task
    task_metrics = defaultdict(dict)
    for key, value in metrics.items():
        if '_' in key:
            task, metric = key.split('_', 1)
            task_metrics[task][metric] = value
        else:
            task_metrics['overall'][key] = value
    
    # Print metrics for each task
    for task, task_metrics_dict in task_metrics.items():
        print(f"\n{task.upper()} TASK METRICS:")
        print("-" * 30)
        for metric, value in task_metrics_dict.items():
            print(f"{metric.capitalize()}: {value:.4f}")
    
    print("="*50)


# Example usage
if __name__ == '__main__':
    # Test metrics computation
    print("Testing evaluation metrics...")
    
    # Create dummy predictions and targets
    batch_size, seq_len, num_classes = 2, 10, 5
    dummy_predictions = torch.randn(batch_size, seq_len, num_classes)
    dummy_targets = torch.randint(0, num_classes, (batch_size, seq_len))
    
    # Compute classification metrics
    metrics = compute_classification_metrics(dummy_predictions, dummy_targets)
    print("Classification metrics:", metrics)
    
    # Test WER computation
    pred_texts = ["hello world", "this is a test"]
    ref_texts = ["hello word", "this is test"]
    wer = compute_word_error_rate(pred_texts, ref_texts)
    print(f"Word Error Rate: {wer:.4f}")
    
    # Test diarization metrics
    dummy_speaker_pred = torch.randint(0, 3, (batch_size, seq_len))
    dummy_speaker_target = torch.randint(0, 3, (batch_size, seq_len))
    diar_metrics = compute_diarization_metrics(dummy_speaker_pred, dummy_speaker_target)
    print("Diarization metrics:", diar_metrics)