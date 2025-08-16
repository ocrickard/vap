#!/usr/bin/env python3
"""
Setup test script to verify VAP Turn Detector project structure
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vap.models import VAPTurnDetector
from vap.streaming import StreamingTurnDetector


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def test_basic_imports():
    """Test that all basic imports work."""
    logging.info("Testing basic imports...")
    
    try:
        import torch
        import torchaudio
        import numpy as np
        logging.info("✓ PyTorch ecosystem imports successful")
    except ImportError as e:
        logging.error(f"✗ PyTorch ecosystem import failed: {e}")
        return False
    
    try:
        import lhotse
        logging.info("✓ Lhotse import successful")
    except ImportError as e:
        logging.warning(f"⚠ Lhotse import failed (optional): {e}")
    
    try:
        import pyannote.audio
        logging.info("✓ Pyannote import successful")
    except ImportError as e:
        logging.warning(f"⚠ Pyannote import failed (optional): {e}")
    
    return True


def test_model_creation():
    """Test that the VAP model can be created."""
    logging.info("Testing model creation...")
    
    try:
        model = VAPTurnDetector()
        model_size = model.get_model_size()
        
        logging.info(f"✓ Model created successfully")
        logging.info(f"  - Total parameters: {model_size['total_parameters']:,}")
        logging.info(f"  - Model size: {model_size['model_size_mb']:.1f} MB")
        
        return True
    except Exception as e:
        logging.error(f"✗ Model creation failed: {e}")
        return False


def test_model_forward_pass():
    """Test that the model can perform a forward pass."""
    logging.info("Testing model forward pass...")
    
    try:
        model = VAPTurnDetector()
        model.eval()
        
        # Create dummy input
        batch_size = 1
        seq_len = 16000  # 1 second at 16kHz
        audio_a = torch.randn(batch_size, seq_len)
        audio_b = torch.randn(batch_size, seq_len)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(audio_a, audio_b)
        
        # Check outputs
        expected_keys = ['vap_logits', 'eot_logits', 'backchannel_logits', 'overlap_logits', 'vad_logits']
        for key in expected_keys:
            if key not in outputs:
                raise ValueError(f"Missing output key: {key}")
        
        logging.info("✓ Forward pass successful")
        logging.info(f"  - VAP logits shape: {outputs['vap_logits'].shape}")
        logging.info(f"  - EoT logits shape: {outputs['eot_logits'].shape}")
        
        return True
    except Exception as e:
        logging.error(f"✗ Forward pass failed: {e}")
        return False


def test_streaming_detector():
    """Test that the streaming detector can be created."""
    logging.info("Testing streaming detector creation...")
    
    try:
        model = VAPTurnDetector()
        streaming_detector = StreamingTurnDetector(
            model=model,
            device="cpu",
            buffer_duration=6.0,
            hop_duration=0.02
        )
        
        logging.info("✓ Streaming detector created successfully")
        logging.info(f"  - Buffer duration: {streaming_detector.buffer_duration}s")
        logging.info(f"  - Hop duration: {streaming_detector.hop_duration}s")
        
        return True
    except Exception as e:
        logging.error(f"✗ Streaming detector creation failed: {e}")
        return False


def test_device_compatibility():
    """Test device compatibility."""
    logging.info("Testing device compatibility...")
    
    try:
        model = VAPTurnDetector()
        
        # Test CPU
        model_cpu = model.cpu()
        assert next(model_cpu.parameters()).device.type == 'cpu'
        logging.info("✓ CPU compatibility confirmed")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            assert next(model_cuda.parameters()).device.type == 'cuda'
            logging.info("✓ CUDA compatibility confirmed")
        else:
            logging.info("⚠ CUDA not available, skipping CUDA test")
        
        return True
    except Exception as e:
        logging.error(f"✗ Device compatibility test failed: {e}")
        return False


def test_training_components():
    """Test that training components can be imported."""
    logging.info("Testing training components...")
    
    try:
        from vap.tasks import VAPTurnDetectionTask
        logging.info("✓ Training task import successful")
        
        # Create a simple training task
        model = VAPTurnDetector()
        task = VAPTurnDetectionTask(model=model)
        logging.info("✓ Training task creation successful")
        
        return True
    except Exception as e:
        logging.error(f"✗ Training components test failed: {e}")
        return False


def main():
    """Main test function."""
    setup_logging()
    logging.info("Starting VAP Turn Detector setup test...")
    
    tests = [
        ("Basic imports", test_basic_imports),
        ("Model creation", test_model_creation),
        ("Model forward pass", test_model_forward_pass),
        ("Streaming detector", test_streaming_detector),
        ("Device compatibility", test_device_compatibility),
        ("Training components", test_training_components),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logging.info(f"\n{'='*50}")
        logging.info(f"Running: {test_name}")
        logging.info(f"{'='*50}")
        
        if test_func():
            passed += 1
            logging.info(f"✓ {test_name} PASSED")
        else:
            logging.error(f"✗ {test_name} FAILED")
    
    logging.info(f"\n{'='*50}")
    logging.info(f"TEST SUMMARY: {passed}/{total} tests passed")
    logging.info(f"{'='*50}")
    
    if passed == total:
        logging.info("🎉 All tests passed! The project is ready to use.")
        return True
    else:
        logging.error(f"❌ {total - passed} tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 