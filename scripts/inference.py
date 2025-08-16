#!/usr/bin/env python3
"""
Inference Script for VAP Turn Detector

This script provides inference capabilities for the trained VAP turn detector.
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trained_model(checkpoint_path=None):
    """Load the trained VAP turn detector model"""
    from vap.models.vap_model import VAPTurnDetector
    
    # Use default checkpoint if none specified
    if checkpoint_path is None:
        checkpoint_dir = Path("checkpoints/simple_baseline")
        if not checkpoint_dir.exists():
            raise FileNotFoundError("No checkpoints found. Train the model first.")
        
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoint files found.")
        
        checkpoint_path = str(max(checkpoints, key=lambda x: x.stat().st_mtime))
    
    logger.info(f"Loading model from: {checkpoint_path}")
    
    # Create model with same architecture as training
    model = VAPTurnDetector(
        hidden_dim=64,
        num_layers=1,
        num_heads=2
    )
    
    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if it exists
        if any(key.startswith('model.') for key in state_dict.keys()):
            state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info("✅ Model loaded successfully")
    return model

def run_inference(model, audio_a, audio_b):
    """Run inference on audio inputs"""
    logger.info("Running inference...")
    
    with torch.no_grad():
        outputs = model(audio_a, audio_b)
    
    return outputs

def analyze_predictions(outputs):
    """Analyze and interpret model predictions"""
    logger.info("Analyzing predictions...")
    
    # Get sequence length
    seq_len = outputs['vap_logits'].shape[1]
    
    # VAP Pattern Analysis
    vap_probs = torch.softmax(outputs['vap_logits'], dim=-1)
    vap_preds = torch.argmax(vap_probs, dim=-1)
    
    # EoT Analysis
    eot_probs = torch.sigmoid(outputs['eot_logits'])
    eot_preds = (eot_probs > 0.5).float()
    
    # Backchannel Analysis
    backchannel_probs = torch.sigmoid(outputs['backchannel_logits'])
    backchannel_preds = (backchannel_probs > 0.5).float()
    
    # Overlap Analysis
    overlap_probs = torch.sigmoid(outputs['overlap_logits'])
    overlap_preds = (overlap_probs > 0.5).float()
    
    # VAD Analysis
    vad_probs = torch.softmax(outputs['vad_logits'], dim=-1)
    vad_preds = torch.argmax(vad_probs, dim=-1)
    
    analysis = {
        'sequence_length': seq_len,
        'vap_patterns': {
            'predictions': vap_preds.tolist(),
            'probabilities': vap_probs.max(dim=-1)[0].tolist()
        },
        'end_of_turn': {
            'predictions': eot_preds.tolist(),
            'probabilities': eot_probs.tolist()
        },
        'backchannel': {
            'predictions': backchannel_preds.tolist(),
            'probabilities': backchannel_probs.tolist()
        },
        'overlap': {
            'predictions': overlap_preds.tolist(),
            'probabilities': overlap_probs.tolist()
        },
        'voice_activity': {
            'predictions': vad_preds.tolist(),
            'probabilities': vad_probs.max(dim=-1)[0].tolist()
        }
    }
    
    return analysis

def main():
    """Main inference function"""
    logger.info("🎯 VAP Turn Detector Inference")
    logger.info("="*50)
    
    try:
        # Load model
        model = load_trained_model()
        
        # Create test audio (30 seconds at 16kHz)
        audio_length = 16000 * 30
        audio_a = torch.randn(1, audio_length)  # Speaker A
        audio_b = torch.randn(1, audio_length)  # Speaker B
        
        logger.info(f"Test audio created: A={audio_a.shape}, B={audio_b.shape}")
        
        # Run inference
        outputs = run_inference(model, audio_a, audio_b)
        
        # Analyze predictions
        analysis = analyze_predictions(outputs)
        
        # Print summary
        logger.info("\n📊 INFERENCE RESULTS")
        logger.info("="*50)
        logger.info(f"Sequence Length: {analysis['sequence_length']} time steps")
        logger.info(f"Audio Duration: 30 seconds")
        logger.info("")
        logger.info("Prediction Summary:")
        logger.info(f"  VAP Patterns: {len(analysis['vap_patterns']['predictions'][0])} time steps")
        logger.info(f"  EoT Events: {sum(analysis['end_of_turn']['predictions'][0])} detected")
        logger.info(f"  Backchannels: {sum(analysis['backchannel']['predictions'][0])} detected")
        logger.info(f"  Overlaps: {sum(analysis['overlap']['predictions'][0])} detected")
        logger.info(f"  VAD Changes: {len(set(analysis['voice_activity']['predictions'][0]))} states")
        
        logger.info("="*50)
        logger.info("✅ Inference completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 