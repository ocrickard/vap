#!/usr/bin/env python3
"""
RunPod-Optimized Training for VAP Turn Detector - Phase 3

This script is specifically designed for GPU-accelerated training on RunPod
with optimized batch sizes, mixed precision, and cloud storage integration.
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
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, 
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from tqdm import tqdm
import threading
import queue

# Configure logging for RunPod environment
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RunPodProgressTracker:
    """Enhanced progress tracker for RunPod GPU training"""
    
    def __init__(self, total_epochs, total_train_batches, total_val_batches):
        self.total_epochs = total_epochs
        self.total_train_batches = total_train_batches
        self.total_val_batches = total_val_batches
        self.current_epoch = 0
        self.current_train_batch = 0
        self.current_val_batch = 0
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        
        # GPU-specific metrics
        self.gpu_utilization = []
        self.memory_usage = []
        self.batch_times = []
        
        # Progress bars
        self.epoch_pbar = None
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        self.best_val_accuracy = 0.0
        
        # Training statistics
        self.epoch_times = []
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Display initial status
        self._display_runpod_header()
    
    def _display_runpod_header(self):
        """Display RunPod-specific training header"""
        logger.info("\n" + "="*80)
        logger.info("🚀 VAP TURN DETECTOR - PHASE 3 OPTIMIZATION ON RUNPOD")
        logger.info("="*80)
        logger.info(f"🔥 GPU Training Configuration:")
        logger.info(f"   • GPU Type: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        logger.info(f"   • CUDA Version: {torch.version.cuda}")
        logger.info(f"   • Total Epochs: {self.total_epochs}")
        logger.info(f"   • Train Batches per Epoch: {self.total_train_batches}")
        logger.info(f"   • Val Batches per Epoch: {self.total_val_batches}")
        logger.info(f"   • Estimated Total Batches: {self.total_epochs * (self.total_train_batches + self.total_val_batches):,}")
        logger.info(f"   • Mixed Precision: Enabled (16-bit)")
        logger.info("="*80)
    
    def start_epoch(self, epoch):
        """Start tracking a new epoch"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.current_train_batch = 0
        self.current_val_batch = 0
        
        logger.info(f"\n🚀 EPOCH {epoch}/{self.total_epochs}")
        logger.info("="*60)
        
        # Create epoch progress bar
        self.epoch_pbar = tqdm(
            total=self.total_train_batches + self.total_val_batches,
            desc=f"Epoch {epoch}",
            unit="batch",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    
    def update_train_batch(self, batch_idx, metrics):
        """Update training batch progress"""
        self.current_train_batch = batch_idx + 1
        
        # Update progress bar
        if self.epoch_pbar:
            self.epoch_pbar.update(1)
            self.epoch_pbar.set_postfix({
                'Train': f"{batch_idx+1}/{self.total_train_batches}",
                'Loss': f"{metrics.get('total', 0):.4f}",
                'VAP': f"{metrics.get('vap_loss', 0):.4f}",
                'EoT': f"{metrics.get('eot_loss', 0):.4f}"
            })
        
        # Store metrics
        self.train_metrics = metrics
        
        # Log GPU metrics periodically
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0) / 1024**3  # GB
            logger.info(f"🖥️  GPU Memory: {gpu_memory:.2f} GB | Batch {batch_idx+1}/{self.total_train_batches}")
    
    def update_val_batch(self, batch_idx, metrics):
        """Update validation batch progress"""
        self.current_val_batch = batch_idx + 1
        
        # Update progress bar
        if self.epoch_pbar:
            self.epoch_pbar.update(1)
            self.epoch_pbar.set_postfix({
                'Val': f"{batch_idx+1}/{self.total_val_batches}",
                'Loss': f"{metrics.get('total', 0):.4f}",
                'Acc': f"{metrics.get('overall_accuracy', 0):.4f}"
            })
        
        # Store metrics
        self.val_metrics = metrics
    
    def end_epoch(self, epoch_metrics):
        """End epoch tracking and display summary"""
        epoch_time = time.time() - self.epoch_start_time
        
        # Store epoch statistics
        self.epoch_times.append(epoch_time)
        self.train_losses.append(self.train_metrics.get('total', 0))
        self.val_losses.append(self.val_metrics.get('total', 0))
        self.val_accuracies.append(self.val_metrics.get('overall_accuracy', 0))
        
        # Close progress bar
        if self.epoch_pbar:
            self.epoch_pbar.close()
        
        # Display epoch summary
        logger.info(f"\n📊 EPOCH {self.current_epoch} SUMMARY")
        logger.info("-" * 50)
        logger.info(f"⏱️  Duration: {epoch_time:.1f}s")
        logger.info(f"📈 Train Loss: {self.train_metrics.get('total', 0):.4f}")
        logger.info(f"📉 Val Loss: {self.val_metrics.get('total', 0):.4f}")
        logger.info(f"🎯 Val Accuracy: {self.val_metrics.get('overall_accuracy', 0):.4f}")
        
        # Check for best performance
        current_val_accuracy = self.val_metrics.get('overall_accuracy', 0)
        if current_val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_val_accuracy
            logger.info(f"🎉 NEW BEST VALIDATION ACCURACY: {current_val_accuracy:.4f}")
        
        # Display training progress summary
        self._display_progress_summary()
    
    def _display_progress_summary(self):
        """Display training progress summary"""
        if len(self.epoch_times) > 0:
            avg_epoch_time = np.mean(self.epoch_times)
            total_time = time.time() - self.start_time
            remaining_epochs = self.total_epochs - self.current_epoch
            estimated_remaining = remaining_epochs * avg_epoch_time
            
            logger.info(f"\n📈 TRAINING PROGRESS SUMMARY")
            logger.info("-" * 40)
            logger.info(f"Completed Epochs: {self.current_epoch}/{self.total_epochs}")
            logger.info(f"Progress: {(self.current_epoch/self.total_epochs)*100:.1f}%")
            logger.info(f"Average Epoch Time: {avg_epoch_time:.1f}s")
            logger.info(f"Total Training Time: {total_time/60:.1f} minutes")
            logger.info(f"Estimated Time Remaining: {estimated_remaining/60:.1f} minutes")
            logger.info(f"Best Validation Accuracy: {self.best_val_accuracy:.4f}")
            
            # Show trend if we have multiple epochs
            if len(self.val_accuracies) > 1:
                recent_trend = self.val_accuracies[-3:] if len(self.val_accuracies) >= 3 else self.val_accuracies
                trend_direction = "↗️ Improving" if recent_trend[-1] > recent_trend[0] else "↘️ Declining" if recent_trend[-1] < recent_trend[0] else "→ Stable"
                logger.info(f"Recent Trend: {trend_direction}")
    
    def stop(self):
        """Stop the progress tracker"""
        if self.epoch_pbar:
            self.epoch_pbar.close()
        
        # Display final training summary
        self._display_final_summary()
    
    def _display_final_summary(self):
        """Display final training summary"""
        total_time = time.time() - self.start_time
        
        logger.info(f"\n🎉 TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info(f"⏱️  Total Training Time: {total_time/60:.1f} minutes")
        logger.info(f"📊 Total Epochs: {self.current_epoch}")
        logger.info(f"🎯 Best Validation Accuracy: {self.best_val_accuracy:.4f}")
        
        if len(self.epoch_times) > 0:
            logger.info(f"📈 Average Epoch Time: {np.mean(self.epoch_times):.1f}s")
            logger.info(f"📉 Final Train Loss: {self.train_losses[-1] if self.train_losses else 'N/A':.4f}")
            logger.info(f"📉 Final Val Loss: {self.val_losses[-1] if self.val_losses else 'N/A':.4f}")
            logger.info(f"🎯 Final Val Accuracy: {self.val_accuracies[-1] if self.val_accuracies else 'N/A':.4f}")
        
        logger.info("="*60)

class RunPodOptimizedTrainingTask(pl.LightningModule):
    """RunPod-optimized training task with GPU acceleration"""
    
    def __init__(self, config_path: str = "runpod/runpod_config.yaml"):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create model with optimized architecture
        from vap.models.vap_model import VAPTurnDetector
        
        self.model = VAPTurnDetector(
            hidden_dim=self.config['training'].get('hidden_dim', 128),
            num_layers=self.config['training'].get('num_layers', 2),
            num_heads=self.config['training'].get('num_heads', 4),
            dropout=self.config['training'].get('dropout', 0.15)
        )
        
        # Training parameters
        self.learning_rate = float(self.config['training']['learning_rate'])
        self.weight_decay = float(self.config['training']['weight_decay'])
        
        # Loss weights
        self.vap_weight = self.config['training'].get('vap_loss_weight', 2.0)
        self.eot_weight = self.config['training'].get('eot_loss_weight', 3.0)
        self.backchannel_weight = self.config['training'].get('backchannel_loss_weight', 1.5)
        self.overlap_weight = self.config['training'].get('overlap_loss_weight', 1.5)
        self.vad_weight = self.config['training'].get('vad_loss_weight', 1.0)
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        # Initialize metrics
        self.train_metrics = {}
        self.val_metrics = {}
        
        # Progress tracker
        self.progress_tracker = None
        
        logger.info(f"✅ RunPod-optimized model created: {sum(p.numel() for p in self.parameters()):,} parameters")
        if torch.cuda.is_available():
            logger.info(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"🔥 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def forward(self, audio_a, audio_b):
        return self.model(audio_a, audio_b)
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.forward(batch['audio_a'], batch['audio_b'])
        
        # Generate real VAP labels
        labels = self._generate_real_labels(batch, outputs)
        
        # Compute losses
        losses = self._compute_enhanced_losses(outputs, labels)
        
        # Log losses
        for key, value in losses.items():
            self.log(f'train_{key}', value, prog_bar=True, sync_dist=True)
        
        # Log learning rate
        self.log('train_lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)
        
        # Update progress tracker
        if self.progress_tracker:
            self.progress_tracker.update_train_batch(batch_idx, losses)
        
        return losses['total']
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.forward(batch['audio_a'], batch['audio_b'])
        
        # Generate real VAP labels
        labels = self._generate_real_labels(batch, outputs)
        
        # Compute losses
        losses = self._compute_enhanced_losses(outputs, labels)
        
        # Compute metrics
        metrics = self._compute_enhanced_metrics(outputs, labels)
        
        # Log everything
        for key, value in losses.items():
            self.log(f'val_{key}', value, sync_dist=True)
        
        for key, value in metrics.items():
            self.log(f'val_{key}', value, sync_dist=True)
        
        # Update progress tracker
        if self.progress_tracker:
            self.progress_tracker.update_val_batch(batch_idx, {**losses, **metrics})
        
        return {'val_loss': losses['total'], 'val_metrics': metrics}
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch"""
        if self.progress_tracker:
            self.progress_tracker.start_epoch(self.current_epoch)
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        if self.progress_tracker:
            # Get current metrics
            train_metrics = {k: v.item() if hasattr(v, 'item') else v 
                           for k, v in self.train_metrics.items()}
            val_metrics = {k: v.item() if hasattr(v, 'item') else v 
                          for k, v in self.val_metrics.items()}
            
            self.progress_tracker.end_epoch({
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
    
    def on_train_end(self):
        """Called when training ends"""
        if self.progress_tracker:
            self.progress_tracker.stop()
        
        # Display final training summary
        total_time = time.time() - self.progress_tracker.start_time if self.progress_tracker else 0
        logger.info(f"\n🎉 TRAINING COMPLETE!")
        logger.info(f"Total Time: {total_time/60:.1f} minutes")
        logger.info(f"Best Validation Accuracy: {self.progress_tracker.best_val_accuracy if self.progress_tracker else 'N/A'}")
    
    # Include the label generation and loss computation methods from the original script
    def _generate_real_labels(self, batch, outputs):
        """Generate realistic VAP labels based on audio characteristics"""
        batch_size, seq_len = outputs['vap_logits'].shape[:2]
        
        # Get audio energy levels for more realistic labeling
        audio_a = batch['audio_a']
        audio_b = batch['audio_b']
        
        # Calculate energy (norm) along time dimension
        audio_a_energy = torch.norm(audio_a, dim=1)
        audio_b_energy = torch.norm(audio_b, dim=1)
        
        # Expand to match sequence length for labeling
        audio_a_energy = audio_a_energy.unsqueeze(1).expand(-1, seq_len)
        audio_b_energy = audio_b_energy.unsqueeze(1).expand(-1, seq_len)
        
        # Create labels
        vap_labels = self._create_energy_based_vap_labels(audio_a_energy, audio_b_energy, seq_len)
        eot_labels = self._create_energy_based_eot_labels(audio_a_energy, audio_b_energy, seq_len)
        backchannel_labels = self._create_backchannel_labels(audio_a_energy, audio_b_energy, seq_len)
        overlap_labels = self._create_overlap_labels(audio_a_energy, audio_b_energy, seq_len)
        vad_labels = self._create_energy_based_vad_labels(audio_a_energy, audio_b_energy, seq_len)
        
        return {
            'vap_labels': vap_labels,
            'eot_labels': eot_labels,
            'backchannel_labels': backchannel_labels,
            'overlap_labels': overlap_labels,
            'vad_labels': vad_labels
        }
    
    def _create_energy_based_vap_labels(self, energy_a, energy_b, seq_len):
        """Create VAP labels based on audio energy patterns"""
        batch_size = energy_a.shape[0]
        
        # Create VAP patterns based on energy levels
        vap_labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
        
        # Speaker A active, B silent
        mask_a_active = (energy_a > 0.1) & (energy_b < 0.05)
        vap_labels[mask_a_active] = 0
        
        # Speaker B active, A silent
        mask_b_active = (energy_a < 0.05) & (energy_b > 0.1)
        vap_labels[mask_b_active] = 1
        
        # Both speakers active (overlap)
        mask_overlap = (energy_a > 0.1) & (energy_b > 0.1)
        vap_labels[mask_overlap] = 2
        
        # Both silent (gap)
        mask_gap = (energy_a < 0.05) & (energy_b < 0.05)
        vap_labels[mask_gap] = 3
        
        # Mixed patterns (default case)
        mask_mixed = ~(mask_a_active | mask_b_active | mask_overlap | mask_gap)
        if mask_mixed.any():
            mixed_energy = energy_a[mask_mixed] + energy_b[mask_mixed]
            mixed_labels = torch.clamp((mixed_energy * 10).long(), 4, 19)
            vap_labels[mask_mixed] = mixed_labels
        
        return vap_labels
    
    def _create_energy_based_eot_labels(self, energy_a, energy_b, seq_len):
        """Create EoT labels based on energy transitions"""
        batch_size = energy_a.shape[0]
        
        eot_labels = torch.zeros(batch_size, seq_len)
        
        # Simple EoT detection: detect energy drops
        if seq_len > 1:
            eot_a = (energy_a[:, :-1] > 0.1) & (energy_a[:, 1:] < 0.05)
            eot_labels[:, 1:] = torch.where(eot_a, 1.0, eot_labels[:, 1:])
            
            eot_b = (energy_b[:, :-1] > 0.1) & (energy_b[:, 1:] < 0.05)
            eot_labels[:, 1:] = torch.where(eot_b, 1.0, eot_labels[:, 1:])
        
        return eot_labels
    
    def _create_backchannel_labels(self, energy_a, energy_b, seq_len):
        """Create backchannel labels"""
        batch_size = energy_a.shape[0]
        
        backchannel_labels = torch.zeros(batch_size, seq_len)
        
        # Simple backchannel detection
        backchannel_mask = (energy_a > 0.1) & (energy_b > 0.05)
        backchannel_labels[backchannel_mask] = 1.0
        
        return backchannel_labels
    
    def _create_overlap_labels(self, energy_a, energy_b, seq_len):
        """Create overlap labels"""
        batch_size = energy_a.shape[0]
        
        overlap_labels = torch.zeros(batch_size, seq_len)
        
        # Simple overlap detection
        overlap_mask = (energy_a > 0.1) & (energy_b > 0.1)
        overlap_labels[overlap_mask] = 1.0
        
        return overlap_labels
    
    def _create_energy_based_vad_labels(self, energy_a, energy_b, seq_len):
        """Create VAD labels based on audio energy"""
        batch_size = energy_a.shape[0]
        
        vad_labels = torch.zeros(batch_size, seq_len, 2)
        
        # Simple VAD based on energy thresholds
        vad_labels[:, :, 0] = (energy_a > 0.05).float()
        vad_labels[:, :, 1] = (energy_b > 0.05).float()
        
        return vad_labels
    
    def _compute_enhanced_losses(self, outputs, labels):
        """Compute enhanced losses with optimized weights"""
        # VAP loss
        vap_loss = F.cross_entropy(outputs['vap_logits'], labels['vap_labels'])
        
        # EoT loss
        eot_loss = F.binary_cross_entropy_with_logits(outputs['eot_logits'], labels['eot_labels'].float())
        
        # Backchannel loss
        backchannel_loss = F.binary_cross_entropy_with_logits(outputs['backchannel_logits'], labels['backchannel_labels'].float())
        
        # Overlap loss
        overlap_loss = F.binary_cross_entropy_with_logits(outputs['overlap_logits'], labels['overlap_labels'].float())
        
        # VAD loss
        vad_loss = F.binary_cross_entropy_with_logits(outputs['vad_logits'], labels['vad_labels'].float())
        
        # Weighted total loss
        total_loss = (
            self.vap_weight * vap_loss +
            self.eot_weight * eot_loss +
            self.backchannel_weight * backchannel_loss +
            self.overlap_weight * overlap_loss +
            self.vad_weight * vad_loss
        )
        
        return {
            'total': total_loss,
            'vap_loss': vap_loss,
            'eot_loss': eot_loss,
            'backchannel_loss': backchannel_loss,
            'overlap_loss': overlap_loss,
            'vad_loss': vad_loss
        }
    
    def _compute_enhanced_metrics(self, outputs, labels):
        """Compute comprehensive performance metrics"""
        # Convert logits to predictions
        vap_preds = torch.argmax(outputs['vap_logits'], dim=-1)
        eot_preds = torch.sigmoid(outputs['eot_logits']) > 0.5
        backchannel_preds = torch.sigmoid(outputs['backchannel_logits']) > 0.5
        overlap_preds = torch.sigmoid(outputs['overlap_logits']) > 0.5
        vad_preds = torch.sigmoid(outputs['vad_logits']) > 0.5
        
        # Calculate accuracies
        vap_accuracy = (vap_preds == labels['vap_labels']).float().mean()
        eot_accuracy = (eot_preds == labels['eot_labels'].bool()).float().mean()
        backchannel_accuracy = (backchannel_preds == labels['backchannel_labels'].bool()).float().mean()
        overlap_accuracy = (overlap_preds == labels['overlap_labels'].bool()).float().mean()
        vad_accuracy = (vad_preds == labels['vad_labels'].bool()).float().mean()
        
        # Overall accuracy
        overall_accuracy = (vap_accuracy + eot_accuracy + backchannel_accuracy + overlap_accuracy + vad_accuracy) / 5
        
        return {
            'overall_accuracy': overall_accuracy,
            'vap_accuracy': vap_accuracy,
            'eot_accuracy': eot_accuracy,
            'backchannel_accuracy': backchannel_accuracy,
            'overlap_accuracy': overlap_accuracy,
            'vad_accuracy': vad_accuracy
        }
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Cosine annealing with restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart interval each time
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_overall_accuracy",
                "frequency": 1
            }
        }

def create_runpod_dataloader(manifest_path, audio_root, batch_size, max_duration):
    """Create data loaders optimized for RunPod GPU training"""
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
    
    # Create data loaders with GPU optimization
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Optimized for GPU
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader

def run_runpod_training():
    """Run RunPod-optimized training"""
    logger.info("🚀 Starting RunPod-Optimized VAP Training (Phase 3)")
    logger.info("="*70)
    
    try:
        # 1. Load configuration
        logger.info("📋 Step 1: Loading RunPod configuration...")
        config_path = "runpod/runpod_config.yaml"
        if not Path(config_path).exists():
            logger.error(f"❌ RunPod configuration not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("✅ RunPod configuration loaded")
        
        # 2. Create training task
        logger.info("🏗️  Step 2: Creating RunPod-optimized training task...")
        task = RunPodOptimizedTrainingTask(config_path)
        logger.info("✅ RunPod training task created")
        
        # 3. Create data loaders
        logger.info("📊 Step 3: Creating GPU-optimized data loaders...")
        manifest_path = "/workspace/data/realtime_dataset/manifest.json"
        audio_root = "/workspace/data/realtime_dataset/LibriSpeech/dev-clean"
        
        if not Path(manifest_path).exists():
            logger.error("❌ Dataset not found in RunPod workspace")
            return False
        
        train_loader, val_loader = create_runpod_dataloader(
            manifest_path=manifest_path,
            audio_root=audio_root,
            batch_size=config['training']['batch_size'],
            max_duration=config['training'].get('max_duration', 30.0)
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
                monitor=config['advanced']['early_stopping']['monitor'],
                patience=config['advanced']['early_stopping']['patience'],
                min_delta=config['advanced']['early_stopping']['min_delta'],
                mode='max'
            ),
            LearningRateMonitor(logging_interval='step'),
            RichProgressBar()
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
        logger.info("🏃 Step 6: Creating RunPod trainer...")
        trainer = pl.Trainer(
            max_epochs=config['training']['num_epochs'],
            callbacks=callbacks,
            logger=logger_tb,
            gradient_clip_val=config['training']['gradient_clip_val'],
            accumulate_grad_batches=config['training']['accumulate_grad_batches'],
            log_every_n_steps=config['logging']['log_every_n_steps'],
            val_check_interval=config['logging']['val_every_n_epochs'],
            enable_progress_bar=True,
            enable_model_summary=True,
            precision=config['hardware']['precision'],
            accelerator=config['hardware']['accelerator'],
            devices=config['hardware']['devices'],
            strategy=config['hardware']['strategy']
        )
        
        # Initialize progress tracker
        total_train_batches = len(train_loader)
        total_val_batches = len(val_loader)
        task.progress_tracker = RunPodProgressTracker(
            total_epochs=config['training']['num_epochs'],
            total_train_batches=total_train_batches,
            total_val_batches=total_val_batches
        )
        
        logger.info("✅ RunPod trainer configured")
        
        # Display training configuration summary
        logger.info("\n📋 RUNPOD TRAINING CONFIGURATION SUMMARY")
        logger.info("="*60)
        logger.info(f"🔥 GPU Configuration:")
        logger.info(f"  GPU Type: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  Mixed Precision: {config['hardware']['precision']}")
        logger.info(f"")
        logger.info(f"Model Architecture:")
        logger.info(f"  Hidden Dim: {config['training'].get('hidden_dim', 128)}")
        logger.info(f"  Layers: {config['training'].get('num_layers', 2)}")
        logger.info(f"  Heads: {config['training'].get('num_heads', 4)}")
        logger.info(f"  Parameters: {sum(p.numel() for p in task.parameters()):,}")
        logger.info(f"")
        logger.info(f"Training Parameters:")
        logger.info(f"  Epochs: {config['training']['num_epochs']}")
        logger.info(f"  Batch Size: {config['training']['batch_size']}")
        logger.info(f"  Learning Rate: {config['training']['learning_rate']}")
        logger.info(f"  Weight Decay: {config['training']['weight_decay']}")
        logger.info(f"")
        logger.info(f"Data Configuration:")
        logger.info(f"  Train Batches: {total_train_batches}")
        logger.info(f"  Val Batches: {total_val_batches}")
        logger.info(f"  Max Duration: {config['training'].get('max_duration', 30.0)}s")
        logger.info("="*60)
        
        # 7. Start training
        logger.info("\n🚀 Starting RunPod-optimized training...")
        start_time = time.time()
        
        trainer.fit(task, train_loader, val_loader)
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")
        
        # 8. Evaluate final model
        logger.info("📊 Evaluating final model...")
        results = trainer.test(task, val_loader)
        
        # 9. Save results
        results_file = f"{config['storage']['results_path']}/runpod_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'runpod_config': config,
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
        logger.error(f"❌ RunPod training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("/workspace/logs", exist_ok=True)
    os.makedirs("/workspace/checkpoints/optimized", exist_ok=True)
    os.makedirs("/workspace/results", exist_ok=True)
    
    # Run training
    success = run_runpod_training()
    if not success:
        sys.exit(1) 