#!/usr/bin/env python3
"""
Complete VAP Pipeline Runner

This script runs the complete VAP turn detector pipeline:
1. Setup and data preparation
2. Training
3. Evaluation
4. Inference demo
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline():
    """Run the complete VAP pipeline"""
    logger.info("🚀 VAP Turn Detector - Complete Pipeline")
    logger.info("="*60)
    
    # Check if we're in the right directory
    if not Path("vap").exists():
        logger.error("❌ Please run this script from the project root directory")
        return False
    
    # Step 1: Check if data is ready
    manifest_path = "data/realtime_dataset/manifest.json"
    if not Path(manifest_path).exists():
        logger.info("📥 Step 1: Setting up data...")
        logger.info("Running: python scripts/setup_phase1.py")
        
        try:
            from scripts.setup_phase1 import main as setup_main
            if not setup_main():
                logger.error("❌ Data setup failed")
                return False
        except ImportError:
            logger.error("❌ Setup script not found")
            return False
        
        logger.info("✅ Data setup completed")
    else:
        logger.info("✅ Data already available")
    
    # Step 2: Check if model is trained
    checkpoint_dir = Path("checkpoints/simple_baseline")
    if not checkpoint_dir.exists() or not list(checkpoint_dir.glob("*.ckpt")):
        logger.info("🏋️  Step 2: Training model...")
        logger.info("Running: python scripts/train_baseline.py")
        
        try:
            from scripts.train_baseline import run_simple_baseline_training
            if not run_simple_baseline_training():
                logger.error("❌ Training failed")
                return False
        except ImportError:
            logger.error("❌ Training script not found")
            return False
        
        logger.info("✅ Training completed")
    else:
        logger.info("✅ Model already trained")
    
    # Step 3: Evaluate model
    logger.info("📊 Step 3: Evaluating model...")
    logger.info("Running: python scripts/evaluate_baseline.py")
    
    try:
        from scripts.evaluate_baseline import evaluate_trained_model
        if not evaluate_trained_model():
            logger.error("❌ Evaluation failed")
            return False
    except ImportError:
        logger.error("❌ Evaluation script not found")
        return False
    
    logger.info("✅ Evaluation completed")
    
    # Step 4: Run inference demo
    logger.info("🎯 Step 4: Running inference demo...")
    logger.info("Running: python scripts/inference.py")
    
    try:
        from scripts.inference import main as inference_main
        if not inference_main():
            logger.error("❌ Inference failed")
            return False
    except ImportError:
        logger.error("❌ Inference script not found")
        return False
    
    logger.info("✅ Inference completed")
    
    # Summary
    logger.info("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info("✅ Data preparation: Complete")
    logger.info("✅ Model training: Complete")
    logger.info("✅ Performance evaluation: Complete")
    logger.info("✅ Inference demo: Complete")
    logger.info("")
    logger.info("Your VAP turn detector is ready!")
    logger.info("Check the results/ directory for detailed metrics.")
    
    return True

def main():
    """Main function"""
    success = run_pipeline()
    
    if success:
        logger.info("🎉 All done! VAP turn detector is ready for use.")
    else:
        logger.error("❌ Pipeline failed. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 