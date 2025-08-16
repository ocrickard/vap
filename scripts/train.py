#!/usr/bin/env python3
"""
Main Training Script for VAP Turn Detector

This script provides the main entry point for training the VAP turn detector.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main training entry point"""
    logger.info("🚀 VAP Turn Detector Training")
    logger.info("="*50)
    
    # Check if we have the required data
    manifest_path = "data/realtime_dataset/manifest.json"
    if not Path(manifest_path).exists():
        logger.error("❌ Dataset not found. Please run setup first:")
        logger.info("  python scripts/setup_phase1.py")
        return False
    
    # Import and run baseline training
    try:
        from scripts.train_baseline import run_simple_baseline_training
        success = run_simple_baseline_training()
        
        if success:
            logger.info("🎉 Training completed successfully!")
            logger.info("Next: python scripts/evaluate_baseline.py")
        else:
            logger.error("❌ Training failed")
            return False
            
    except ImportError:
        logger.error("❌ Training module not found")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 