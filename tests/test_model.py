"""
Simple tests for VAP Turn Detector model
"""

import torch
import numpy as np
import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vap.models import VAPTurnDetector


def test_model_creation():
    """Test that the model can be created with default parameters."""
    model = VAPTurnDetector()
    assert model is not None
    
    # Check model size
    model_size = model.get_model_size()
    assert model_size['total_parameters'] > 0
    assert model_size['model_size_mb'] > 0
    
    print(f"Model created with {model_size['total_parameters']:,} parameters")


def test_model_forward_pass():
    """Test that the model can perform a forward pass."""
    model = VAPTurnDetector()
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 16000  # 1 second at 16kHz
    audio_a = torch.randn(batch_size, seq_len)
    audio_b = torch.randn(batch_size, seq_len)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(audio_a, audio_b)
    
    # Check output structure
    assert 'vap_logits' in outputs
    assert 'eot_logits' in outputs
    assert 'backchannel_logits' in outputs
    assert 'overlap_logits' in outputs
    assert 'vad_logits' in outputs
    
    # Check output shapes
    assert outputs['vap_logits'].shape[0] == batch_size
    assert outputs['eot_logits'].shape[0] == batch_size
    assert outputs['backchannel_logits'].shape[0] == batch_size
    assert outputs['overlap_logits'].shape[0] == batch_size
    assert outputs['vad_logits'].shape[0] == batch_size
    
    print("Forward pass completed successfully")


def test_model_turn_shift_prediction():
    """Test turn shift prediction functionality."""
    model = VAPTurnDetector()
    model.eval()
    
    # Create dummy input
    batch_size = 1
    seq_len = 16000  # 1 second at 16kHz
    audio_a = torch.randn(batch_size, seq_len)
    audio_b = torch.randn(batch_size, seq_len)
    
    # Predict turn shifts
    with torch.no_grad():
        predictions = model.predict_turn_shift(
            audio_a, 
            audio_b,
            threshold=0.5,
            min_gap_ms=200
        )
    
    # Check prediction structure
    assert 'turn_shift' in predictions
    assert 'eot_probability' in predictions
    assert 'backchannel_probability' in predictions
    assert 'overlap_probability' in predictions
    
    print("Turn shift prediction completed successfully")


def test_model_parameters():
    """Test that model parameters are trainable."""
    model = VAPTurnDetector()
    
    # Check that parameters require gradients
    for name, param in model.named_parameters():
        assert param.requires_grad, f"Parameter {name} should require gradients"
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    assert trainable_params == total_params
    print(f"All {trainable_params:,} parameters are trainable")


def test_model_device_moving():
    """Test that the model can be moved to different devices."""
    if torch.cuda.is_available():
        model = VAPTurnDetector()
        
        # Move to GPU
        model = model.cuda()
        assert next(model.parameters()).device.type == 'cuda'
        
        # Move back to CPU
        model = model.cpu()
        assert next(model.parameters()).device.type == 'cpu'
        
        print("Model device moving test passed")
    else:
        print("CUDA not available, skipping device moving test")


if __name__ == "__main__":
    print("Running VAP model tests...")
    
    try:
        test_model_creation()
        test_model_forward_pass()
        test_model_turn_shift_prediction()
        test_model_parameters()
        test_model_device_moving()
        
        print("\nAll tests passed! ✅")
    except Exception as e:
        print(f"\nTest failed: {e}")
        raise 