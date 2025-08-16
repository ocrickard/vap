#!/usr/bin/env python3
"""
Smart VAP Training Script for RunPod and Local Environments

This script automatically detects the environment (local vs RunPod) and configures
training accordingly. It uses the main training task for consistency.
"""

import os
import sys
import logging
import json
import time
import yaml
from pathlib import Path
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, 
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger

# Fix Python path to find the vap module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Now we can import from vap and scripts
from vap.models.vap_model import VAPTurnDetector
from scripts.train_optimized import OptimizedTrainingTask, ProgressTracker

# Configure logging for the environment
log_dir = "logs" if not os.path.exists('/workspace') else "/workspace/logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,  # Back to INFO level
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_runpod_dataloader(manifest_path, audio_root, batch_size, max_duration, config):
    """Create data loaders optimized for the detected environment"""
    # Import here to avoid circular imports
    from vap.data.realtime_training_dataset import RealTrainingDataset
    
    # Create datasets
    train_dataset = RealTrainingDataset(
        manifest_path=manifest_path,
        audio_root=audio_root,
        max_duration=max_duration,
        split='train'
    )
    
    val_dataset = RealTrainingDataset(
        manifest_path=manifest_path,
        audio_root=audio_root,
        max_duration=max_duration,
        split='val'
    )
    
    # Get environment-specific settings
    num_workers = config['num_workers']
    pin_memory = config['pin_memory']
    persistent_workers = config['persistent_workers']
    
    # Create data loaders with environment-optimized settings
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    return train_loader, val_loader

def load_smart_config(config_path: str) -> dict:
    """Load and merge configuration with smart environment detection"""
    
    # Load main configuration
    main_config_path = "configs/vap_optimized.yaml"
    if Path(main_config_path).exists():
        with open(main_config_path, 'r') as f:
            main_config = yaml.safe_load(f)
    else:
        logger.warning(f"Main config not found: {main_config_path}, using RunPod config")
        with open(config_path, 'r') as f:
            main_config = yaml.safe_load(f)
    
    # Load RunPod-specific overrides
    with open(config_path, 'r') as f:
        runpod_config = yaml.safe_load(f)
    
    # Detect environment
    is_runpod = os.path.exists('/workspace') or os.environ.get('RUNPOD_POD_ID') is not None
    has_gpu = torch.cuda.is_available()
    
    logger.info(f"🔍 Environment Detection:")
    logger.info(f"   • RunPod: {'✅ Yes' if is_runpod else '❌ No'}")
    logger.info(f"   • GPU Available: {'✅ Yes' if has_gpu else '❌ No'}")
    
    # Determine optimal configuration
    if is_runpod and has_gpu:
        env_config = runpod_config['environment']['runpod']
        logger.info("🚀 Using RunPod GPU configuration")
    else:
        env_config = runpod_config['environment']['local']
        logger.info("💻 Using Local CPU configuration")
    
    # Merge configurations
    config = main_config.copy()
    
    # Override with environment-specific settings
    config['training'].update({
        'batch_size': env_config['batch_size'],
        'num_workers': env_config['num_workers']
    })
    
    # Add hardware configuration
    config['hardware'] = {
        'precision': env_config['precision'],
        'accelerator': env_config['accelerator'],
        'devices': env_config['devices'],
        'strategy': 'auto'
    }
    
    # Add storage configuration
    config['storage'] = runpod_config['storage']
    
    # Add logging configuration if not present
    if 'logging' not in config:
        config['logging'] = runpod_config['logging']
    
    # Add checkpointing configuration if not present
    if 'checkpointing' not in config:
        config['checkpointing'] = runpod_config['checkpointing']
    
    return config

def run_smart_training():
    """Run smart training with automatic environment detection"""
    logger.info("🚀 Starting Smart VAP Training (Phase 3)")
    logger.info("="*70)
    
    try:
        # 1. Load configuration
        logger.info("📋 Step 1: Loading RunPod configuration...")
        config_path = "runpod/runpod_config.yaml"
        if not Path(config_path).exists():
            logger.error(f"❌ RunPod configuration not found: {config_path}")
            return False
        
        config = load_smart_config(config_path)
        
        logger.info("✅ RunPod configuration loaded")
        
        # 2. Create training task
        logger.info("��️  Step 2: Creating optimized training task...")
        # Use main config for consistency, but apply smart environment detection
        main_config_path = "configs/vap_optimized.yaml"
        if not Path(main_config_path).exists():
            logger.error(f"❌ Main configuration not found: {main_config_path}")
            return False
        
        task = OptimizedTrainingTask(main_config_path)
        logger.info("✅ Optimized training task created")
        
        # 3. Create data loaders
        logger.info("📊 Step 3: Creating environment-optimized data loaders...")
        manifest_path = "data/realtime_dataset/manifest.json"
        audio_root = "data/realtime_dataset/LibriSpeech/dev-clean"
        
        if not Path(manifest_path).exists():
            logger.error("❌ Dataset not found. Please run dataset setup first.")
            return False
        
        # Use smart configuration for data loader settings
        smart_config = load_smart_config(config_path)
        
        has_gpu = torch.cuda.is_available()
        train_loader, val_loader = create_runpod_dataloader(
            manifest_path=manifest_path,
            audio_root=audio_root,
            batch_size=smart_config['training']['batch_size'],
            max_duration=smart_config['training'].get('max_duration', 30.0),
            config={
                'num_workers': smart_config['training'].get('num_workers', 2),
                'pin_memory': has_gpu,
                'persistent_workers': has_gpu
            }
        )
        
        logger.info(f"✅ GPU-optimized data loaders created:")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        
        # 4. Set up callbacks
        logger.info("⚙️  Step 4: Setting up RunPod callbacks...")
        callbacks = [
            ModelCheckpoint(
                monitor=config['logging']['monitor'],
                dirpath=config['checkpointing']['save_dir'],
                filename=config['checkpointing']['filename'],
                save_top_k=config['logging']['save_top_k'],
                mode=config['logging']['mode'],
                save_last=True
            ),
            EarlyStopping(
                monitor=config['logging']['monitor'],
                patience=config['advanced']['early_stopping']['patience'],
                min_delta=config['advanced']['early_stopping']['min_delta'],
                mode=config['logging']['mode']
            ),
            LearningRateMonitor(logging_interval='epoch'),
            RichProgressBar()  # This will show progress locally
        ]
        logger.info("✅ RunPod callbacks configured")
        
        # 5. Set up logger
        logger.info("📝 Step 5: Setting up RunPod logger...")
        logger_tb = TensorBoardLogger(
            save_dir=config['storage']['logs_path'],
            name='vap_runpod_optimized',
            version=datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        logger.info("✅ RunPod logger configured")
        
        # 6. Create trainer
        logger.info("🏃 Step 6: Creating environment-optimized trainer...")
        trainer = pl.Trainer(
            max_epochs=smart_config['training']['num_epochs'],
            callbacks=callbacks,
            logger=logger_tb,
            gradient_clip_val=smart_config['training']['gradient_clip_val'],
            accumulate_grad_batches=smart_config['training']['accumulate_grad_batches'],
            log_every_n_steps=smart_config['logging']['log_every_n_steps'],
            val_check_interval=smart_config['logging']['val_every_n_epochs'],
            enable_progress_bar=True,
            enable_model_summary=True,
            precision=smart_config['hardware']['precision'],
            accelerator=smart_config['hardware']['accelerator'],
            devices=smart_config['hardware']['devices'],
            strategy=smart_config['hardware']['strategy']
        )
        
        # Initialize progress tracker
        total_train_batches = len(train_loader)
        total_val_batches = len(val_loader)
        task.progress_tracker = ProgressTracker(
            total_epochs=smart_config['training']['num_epochs'],
            total_train_batches=total_train_batches,
            total_val_batches=total_val_batches
        )
        
        logger.info("✅ RunPod trainer configured")
        
        # Display training configuration summary
        logger.info("\n📋 SMART TRAINING CONFIGURATION SUMMARY")
        logger.info("="*60)
        logger.info(f"🔥 Hardware Configuration:")
        logger.info(f"  Accelerator: {smart_config['hardware']['accelerator'].upper()}")
        if torch.cuda.is_available():
            logger.info(f"  GPU Type: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA Version: {torch.version.cuda}")
        else:
            logger.info(f"  GPU Type: CPU")
            logger.info(f"  CUDA Version: N/A")
        logger.info(f"  Precision: {smart_config['hardware']['precision']}")
        logger.info(f"")
        logger.info(f"Model Architecture:")
        logger.info(f"  Hidden Dim: {smart_config['training'].get('hidden_dim', 128)}")
        logger.info(f"  Layers: {smart_config['training'].get('num_layers', 2)}")
        logger.info(f"  Heads: {smart_config['training'].get('num_heads', 4)}")
        logger.info(f"  Parameters: {sum(p.numel() for p in task.parameters()):,}")
        logger.info(f"")
        logger.info(f"Training Parameters:")
        logger.info(f"  Epochs: {smart_config['training']['num_epochs']}")
        logger.info(f"  Batch Size: {smart_config['training']['batch_size']}")
        logger.info(f"  Learning Rate: {smart_config['training']['learning_rate']}")
        logger.info(f"  Weight Decay: {smart_config['training']['weight_decay']}")
        logger.info(f"")
        logger.info(f"Data Configuration:")
        logger.info(f"  Train Batches: {total_train_batches}")
        logger.info(f"  Val Batches: {total_val_batches}")
        logger.info(f"  Max Duration: {smart_config['training'].get('max_duration', 30.0)}s")
        logger.info(f"  Workers: {smart_config['training'].get('num_workers', 2)}")
        logger.info("="*60)
        
        # 7. Start training
        env_name = "RunPod GPU" if config['hardware']['accelerator'] == 'gpu' else "Local CPU"
        logger.info(f"\n🚀 Starting {env_name} training...")
        start_time = time.time()
        
        trainer.fit(task, train_loader, val_loader)
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")
        
        # 8. Evaluate final model
        logger.info("📊 Evaluating final model...")
        results = trainer.test(task, val_loader)
        
        # 9. Save results
        results_file = f"{smart_config['storage']['results_path']}/smart_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'smart_config': smart_config,
                'model_parameters': sum(p.numel() for p in task.parameters()),
                'training_time_minutes': training_time / 60,
                'final_results': results,
                'best_validation_accuracy': task.progress_tracker.best_val_accuracy if task.progress_tracker else 0,
                'gpu_info': {
                    'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                    'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
                },
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"✅ Results saved: {results_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Smart training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_smart_training()
    if not success:
        sys.exit(1) 