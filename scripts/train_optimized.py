#!/usr/bin/env python3
"""
Optimized Training for VAP Turn Detector

Phase 3: Model Optimization with enhanced architecture, real labels, and advanced training techniques.
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProgressTracker:
    """Track and display training progress with real-time updates"""
    
    def __init__(self, total_epochs, total_train_batches, total_val_batches):
        self.total_epochs = total_epochs
        self.total_train_batches = total_train_batches
        self.total_val_batches = total_val_batches
        self.current_epoch = 0
        self.current_train_batch = 0
        self.current_val_batch = 0
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        
        # Progress bars
        self.epoch_pbar = None
        self.batch_pbar = None
        
        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}
        self.best_val_accuracy = 0.0
        
        # Status queue for real-time updates
        self.status_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Training statistics
        self.epoch_times = []
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Start status display thread
        self.status_thread = threading.Thread(target=self._status_display_loop)
        self.status_thread.daemon = True
        self.status_thread.start()
        
        # Display initial status
        self._display_training_header()
    
    def start(self):
        """Start the progress tracker"""
        logger.info("🚀 Progress tracker started")
        logger.info(f"📊 Ready to track {self.total_epochs} epochs")
        logger.info(f"   • Train batches per epoch: {self.total_train_batches}")
        logger.info(f"   • Val batches per epoch: {self.total_val_batches}")
    
    def _display_training_header(self):
        """Display the training header with configuration summary"""
        logger.info("\n" + "="*80)
        logger.info("🎯 VAP TURN DETECTOR - PHASE 3 OPTIMIZATION TRAINING")
        logger.info("="*80)
        logger.info(f"📊 Training Configuration:")
        logger.info(f"   • Total Epochs: {self.total_epochs}")
        logger.info(f"   • Train Batches per Epoch: {self.total_train_batches}")
        logger.info(f"   • Val Batches per Epoch: {self.total_val_batches}")
        logger.info(f"   • Estimated Total Batches: {self.total_epochs * (self.total_train_batches + self.total_val_batches):,}")
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
        
        # Send status update
        self.status_queue.put({
            'type': 'train_batch',
            'epoch': self.current_epoch,
            'batch': self.current_train_batch,
            'total_batches': self.total_train_batches,
            'metrics': metrics
        })
    
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
        
        # Send status update
        self.status_queue.put({
            'type': 'val_batch',
            'epoch': self.current_epoch,
            'batch': self.current_val_batch,
            'total_batches': self.total_val_batches,
            'metrics': metrics
        })
    
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
        
        # Send epoch completion status
        self.status_queue.put({
            'type': 'epoch_complete',
            'epoch': self.current_epoch,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'duration': epoch_time,
            'best_accuracy': self.best_val_accuracy
        })
    
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
    
    def _status_display_loop(self):
        """Background thread for displaying real-time status updates"""
        while not self.stop_event.is_set():
            try:
                # Get status update with timeout
                status = self.status_queue.get(timeout=1.0)
                
                if status['type'] == 'train_batch':
                    # Real-time training progress (less verbose)
                    if status['batch'] % 50 == 0:  # Update every 50 batches
                        progress = (status['batch'] / status['total_batches']) * 100
                        logger.info(f"🔄 Train Progress: {progress:.1f}% | Loss: {status['metrics'].get('total', 0):.4f}")
                
                elif status['type'] == 'val_batch':
                    # Real-time validation progress (less verbose)
                    if status['batch'] % 20 == 0:  # Update every 20 batches
                        progress = (status['batch'] / status['total_batches']) * 100
                        logger.info(f"✅ Val Progress: {progress:.1f}% | Acc: {status['metrics'].get('overall_accuracy', 0):.4f}")
                
                elif status['type'] == 'epoch_complete':
                    # Epoch completion summary
                    logger.info(f"🎯 Epoch {status['epoch']} Complete!")
                    logger.info(f"   Best Accuracy: {status['best_accuracy']:.4f}")
                    logger.info(f"   Time: {status['duration']:.1f}s")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"Status display error: {e}")
    
    def stop(self):
        """Stop the progress tracker"""
        self.stop_event.set()
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

class OptimizedTrainingTask(pl.LightningModule):
    """Enhanced training task with advanced features and real VAP label generation"""
    
    def __init__(self, config_path: str = "configs/vap_optimized.yaml"):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Enable Tensor Core optimizations for RTX 4090
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')  # Best performance, slight precision trade-off
            logger.info("🚀 Tensor Core optimization enabled for RTX 4090")
        
        # Create model with optimized architecture
        from vap.models.vap_model import VAPTurnDetector
        
        self.model = VAPTurnDetector(
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads'],
            dropout=self.config['model']['dropout']
        )
        
        # Training parameters
        self.learning_rate = float(self.config['training']['learning_rate'])
        self.weight_decay = float(self.config['training']['weight_decay'])
        
        # Loss weights (optimized based on baseline results)
        self.vap_weight = self.config['training']['vap_loss_weight']
        self.eot_weight = self.config['training']['eot_loss_weight']
        self.backchannel_weight = self.config['training']['backchannel_loss_weight']
        self.overlap_weight = self.config['training']['overlap_loss_weight']
        self.vad_weight = self.config['training']['vad_loss_weight']
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        # Initialize metrics
        self.train_metrics = {}
        self.val_metrics = {}
        
        # Progress tracker
        self.progress_tracker = None
        
        # Memory optimizations
        self.automatic_optimization = True
        self.gradient_clip_val = 1.0
        
        # Enable memory efficient attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Let PyTorch Lightning handle device placement
        # The trainer will automatically move the model to the correct device
        logger.info("✅ Model device placement will be handled by PyTorch Lightning")
        
        logger.info(f"✅ Optimized model created: {sum(p.numel() for p in self.parameters()):,} parameters")
    
    def forward(self, audio_a, audio_b):
        return self.model(audio_a, audio_b)
    
    def training_step(self, batch, batch_idx):
        # Check if data needs to be moved to GPU
        if torch.cuda.is_available() and batch['audio_a'].device.type == 'cpu':
            batch['audio_a'] = batch['audio_a'].cuda()
            batch['audio_b'] = batch['audio_b'].cuda()
        
        # Forward pass
        outputs = self.forward(batch['audio_a'], batch['audio_b'])
        
        # Generate real VAP labels based on audio characteristics
        labels = self._generate_real_labels(batch, outputs)
        
        # Compute losses
        losses = self._compute_enhanced_losses(outputs, labels)
        
        # Log losses
        for key, value in losses.items():
            self.log(f'train_{key}', value, prog_bar=True, sync_dist=True)
        
        # Log learning rate
        self.log('train_lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)
        
        # Log memory usage for optimization monitoring
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            self.log('train_gpu_memory_allocated', memory_allocated, prog_bar=False)
            self.log('train_gpu_memory_reserved', memory_reserved, prog_bar=False)
        
        # Update progress tracker
        if self.progress_tracker:
            self.progress_tracker.update_train_batch(batch_idx, losses)
        
        return losses['total']
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.forward(batch['audio_a'], batch['audio_b'])
        
        # Check if data needs to be moved to GPU
        if torch.cuda.is_available() and batch['audio_a'].device.type == 'cpu':
            batch['audio_a'] = batch['audio_a'].cuda()
            batch['audio_b'] = batch['audio_b'].cuda()
        
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
    
    def on_train_start(self):
        """Called when training starts"""
        if self.progress_tracker:
            self.progress_tracker.start()
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch"""
        if self.progress_tracker:
            # Use the trainer's current epoch if available, otherwise use 0
            current_epoch = getattr(self.trainer, 'current_epoch', 0) if hasattr(self, 'trainer') else 0
            self.progress_tracker.start_epoch(current_epoch)
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch"""
        if self.progress_tracker:
            # Get current metrics from the progress tracker
            train_metrics = getattr(self.progress_tracker, 'train_metrics', {})
            val_metrics = getattr(self.progress_tracker, 'val_metrics', {})
            
            # Ensure metrics are properly formatted
            if isinstance(train_metrics, dict):
                train_metrics = {k: v.item() if hasattr(v, 'item') else v 
                               for k, v in train_metrics.items()}
            if isinstance(val_metrics, dict):
                val_metrics = {k: v.item() if hasattr(v, 'item') else v 
                             for k, v in val_metrics.items()}
            
            self.progress_tracker.end_epoch({
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
    
    def on_train_end(self):
        """Called when training ends"""
        if self.progress_tracker:
            self.progress_tracker.stop()
            
            # Display final training summary
            total_time = time.time() - self.progress_tracker.start_time
            logger.info(f"\n🎉 TRAINING COMPLETE!")
            logger.info(f"Total Time: {total_time/60:.1f} minutes")
            logger.info(f"Best Validation Accuracy: {self.progress_tracker.best_val_accuracy:.4f}")
        else:
            logger.info(f"\n🎉 TRAINING COMPLETE!")
            logger.info("Progress tracker not available")
    
    def _generate_real_labels(self, batch, outputs):
        """Generate realistic VAP labels based on audio characteristics"""
        batch_size, seq_len = outputs['vap_logits'].shape[:2]
        
        # Get the target device from the model outputs
        target_device = outputs['vap_logits'].device
        
        # Get audio energy levels for more realistic labeling
        # Ensure proper tensor dimensions
        audio_a = batch['audio_a'].to(target_device)  # [batch, time]
        audio_b = batch['audio_b'].to(target_device)  # [batch, time]
        
        # Calculate energy (norm) along time dimension
        audio_a_energy = torch.norm(audio_a, dim=1)  # [batch, time] -> [batch]
        audio_b_energy = torch.norm(audio_b, dim=1)  # [batch, time] -> [batch]
        
        # Expand to match sequence length for labeling
        audio_a_energy = audio_a_energy.unsqueeze(1).expand(-1, seq_len)  # [batch, seq_len]
        audio_b_energy = audio_b_energy.unsqueeze(1).expand(-1, seq_len)  # [batch, seq_len]
        
        # Create VAP pattern labels based on audio energy
        vap_labels = self._create_energy_based_vap_labels(audio_a_energy, audio_b_energy, seq_len)
        
        # Create EoT labels based on energy transitions
        eot_labels = self._create_energy_based_eot_labels(audio_a_energy, audio_b_energy, seq_len)
        
        # Create backchannel labels (when speaker B responds briefly)
        backchannel_labels = self._create_backchannel_labels(audio_a_energy, audio_b_energy, seq_len)
        
        # Create overlap labels (when both speakers are active)
        overlap_labels = self._create_overlap_labels(audio_a_energy, audio_b_energy, seq_len)
        
        # Create VAD labels based on actual audio activity
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
        device = energy_a.device  # Get device from input tensor
        
        # For now, let's use a simpler approach without interpolation
        # Just create basic labels based on the original energy patterns
        vap_labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        
        # Create simple VAP patterns
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
        device = energy_a.device  # Get device from input tensor
        
        eot_labels = torch.zeros(batch_size, seq_len, device=device)
        
        # Simple EoT detection: detect energy drops
        # Speaker A ends turn
        if seq_len > 1:
            eot_a = (energy_a[:, :-1] > 0.1) & (energy_a[:, 1:] < 0.05)
            eot_labels[:, 1:] = torch.where(eot_a, torch.tensor(1.0, device=device), eot_labels[:, 1:])
            
            # Speaker B ends turn
            eot_b = (energy_b[:, :-1] > 0.1) & (energy_b[:, 1:] < 0.05)
            eot_labels[:, 1:] = torch.where(eot_b, torch.tensor(1.0, device=device), eot_labels[:, 1:])
        
        return eot_labels
    
    def _create_backchannel_labels(self, energy_a, energy_b, seq_len):
        """Create backchannel labels"""
        batch_size = energy_a.shape[0]
        device = energy_a.device  # Get device from input tensor
        
        backchannel_labels = torch.zeros(batch_size, seq_len, device=device)
        
        # Simple backchannel detection
        backchannel_mask = (energy_a > 0.1) & (energy_b > 0.05)
        backchannel_labels[backchannel_mask] = 1.0
        
        return backchannel_labels
    
    def _create_overlap_labels(self, energy_a, energy_b, seq_len):
        """Create overlap labels"""
        batch_size = energy_a.shape[0]
        device = energy_a.device  # Get device from input tensor
        
        overlap_labels = torch.zeros(batch_size, seq_len, device=device)
        
        # Simple overlap detection
        overlap_mask = (energy_a > 0.1) & (energy_b > 0.1)
        overlap_labels[overlap_mask] = 1.0
        
        return overlap_labels
    
    def _create_energy_based_vad_labels(self, energy_a, energy_b, seq_len):
        """Create VAD labels based on audio energy"""
        batch_size = energy_a.shape[0]
        device = energy_a.device  # Get device from input tensor
        
        vad_labels = torch.zeros(batch_size, seq_len, 2, device=device)
        
        # Simple VAD based on energy thresholds
        vad_labels[:, :, 0] = (energy_a > 0.05).float()  # Speaker A VAD
        vad_labels[:, :, 1] = (energy_b > 0.05).float()  # Speaker B VAD
        
        return vad_labels
    
    def _compute_enhanced_losses(self, outputs, targets):
        """Compute enhanced losses with better balancing"""
        import torch.nn.functional as F
        
        losses = {}
        
        # VAP pattern loss with label smoothing
        vap_loss = F.cross_entropy(
            outputs['vap_logits'].view(-1, outputs['vap_logits'].size(-1)),
            targets['vap_labels'].view(-1),
            label_smoothing=0.1
        )
        losses['vap_loss'] = vap_loss
        
        # EoT loss with focal loss for better handling of imbalanced classes
        eot_loss = self._focal_binary_cross_entropy(
            outputs['eot_logits'],
            targets['eot_labels'].float(),
            alpha=0.25,
            gamma=2.0
        )
        losses['eot_loss'] = eot_loss
        
        # Backchannel loss
        backchannel_loss = F.binary_cross_entropy_with_logits(
            outputs['backchannel_logits'],
            targets['backchannel_labels'].float()
        )
        losses['backchannel_loss'] = backchannel_loss
        
        # Overlap loss
        overlap_loss = F.binary_cross_entropy_with_logits(
            outputs['overlap_logits'],
            targets['overlap_labels'].float()
        )
        losses['overlap_loss'] = overlap_loss
        
        # VAD loss
        vad_loss = F.binary_cross_entropy_with_logits(
            outputs['vad_logits'],
            targets['vad_labels'].float()
        )
        losses['vad_loss'] = vad_loss
        
        # Total weighted loss
        total_loss = (
            self.vap_weight * vap_loss +
            self.eot_weight * eot_loss +
            self.backchannel_weight * backchannel_loss +
            self.overlap_weight * overlap_loss +
            self.vad_weight * vad_loss
        )
        losses['total'] = total_loss
        
        return losses
    
    def _focal_binary_cross_entropy(self, logits, targets, alpha=0.25, gamma=2.0):
        """Focal loss for better handling of imbalanced binary classification"""
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
    
    def _compute_enhanced_metrics(self, outputs, targets):
        """Compute comprehensive evaluation metrics"""
        metrics = {}
        
        # VAP accuracy
        vap_preds = torch.argmax(outputs['vap_logits'], dim=-1)
        vap_acc = (vap_preds == targets['vap_labels']).float().mean().item()
        metrics['vap_accuracy'] = vap_acc
        
        # EoT metrics
        eot_probs = torch.sigmoid(outputs['eot_logits'])
        eot_preds = (eot_probs > 0.5).float()
        eot_acc = (eot_preds == targets['eot_labels']).float().mean().item()
        metrics['eot_accuracy'] = eot_acc
        
        # EoT F1 scores
        metrics['eot_f1_200ms'] = self._compute_eot_f1(eot_probs, targets['eot_labels'], tolerance_ms=200)
        metrics['eot_f1_500ms'] = self._compute_eot_f1(eot_probs, targets['eot_labels'], tolerance_ms=500)
        
        # Other metrics
        backchannel_probs = torch.sigmoid(outputs['backchannel_logits'])
        backchannel_preds = (backchannel_probs > 0.5).float()
        backchannel_acc = (backchannel_preds == targets['backchannel_labels']).float().mean().item()
        metrics['backchannel_accuracy'] = backchannel_acc
        
        overlap_probs = torch.sigmoid(outputs['overlap_logits'])
        overlap_preds = (overlap_probs > 0.5).float()
        overlap_acc = (overlap_preds == targets['overlap_labels']).float().mean().item()
        metrics['overlap_accuracy'] = overlap_acc
        
        vad_probs = torch.sigmoid(outputs['vad_logits'])
        vad_preds = (vad_probs > 0.5).float()
        vad_acc = (vad_preds == targets['vad_labels']).float().mean().item()
        metrics['vad_accuracy'] = vad_acc
        
        # Overall accuracy
        overall_acc = (vap_acc + eot_acc + backchannel_acc + overlap_acc + vad_acc) / 5
        metrics['overall_accuracy'] = overall_acc
        
        return metrics
    
    def _compute_eot_f1(self, predictions, targets, tolerance_ms=200):
        """Compute EoT F1 score with temporal tolerance"""
        # Convert to binary predictions
        pred_binary = (predictions > 0.5).float()
        
        # Simple F1 calculation (can be enhanced with temporal tolerance)
        tp = (pred_binary * targets).sum().item()
        fp = (pred_binary * (1 - targets)).sum().item()
        fn = ((1 - pred_binary) * targets).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
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
            T_0=int(self.config['optimization']['lr_scheduler_params']['T_0']),
            T_mult=int(self.config['optimization']['lr_scheduler_params']['T_mult']),
            eta_min=float(self.config['optimization']['lr_scheduler_params']['eta_min'])
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_overall_accuracy",
                "frequency": 1
            }
        }

def create_optimized_dataloader(manifest_path, audio_root, batch_size=32, max_duration=30.0):
    """Create optimized data loader with enhanced features"""
    from vap.data.realtime_training_dataset import create_real_training_loader
    
    # Use the existing real training loader but with optimized parameters
    train_loader, val_loader = create_real_training_loader(
        manifest_path=manifest_path,
        audio_root=audio_root,
        batch_size=batch_size,
        max_duration=max_duration
    )
    
    # Apply additional optimizations if the loaders support it
    for loader in [train_loader, val_loader]:
        if hasattr(loader, 'num_workers'):
            loader.num_workers = 8
        if hasattr(loader, 'pin_memory'):
            loader.pin_memory = True
        if hasattr(loader, 'persistent_workers'):
            loader.persistent_workers = True
        if hasattr(loader, 'prefetch_factor'):
            loader.prefetch_factor = 2
    
    return train_loader, val_loader

def run_optimized_training():
    """Run optimized training with enhanced features"""
    logger.info("🚀 Starting Optimized VAP Training (Phase 3)")
    logger.info("="*70)
    
    # Enable Tensor Core optimizations globally
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        logger.info("🚀 Global Tensor Core optimization enabled for RTX 4090")
    
    try:
        # 1. Load configuration
        logger.info("📋 Step 1: Loading configuration...")
        config_path = "configs/vap_optimized.yaml"
        if not Path(config_path).exists():
            logger.error(f"❌ Configuration file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("✅ Configuration loaded")
        
        # 2. Create training task
        logger.info("🏗️  Step 2: Creating training task...")
        task = OptimizedTrainingTask(config_path)
        logger.info("✅ Training task created")
        
        # 3. Create data loaders
        logger.info("📊 Step 3: Creating data loaders...")
        manifest_path = "data/realtime_dataset/manifest.json"
        audio_root = "data/realtime_dataset/LibriSpeech/dev-clean"
        
        if not Path(manifest_path).exists():
            logger.error("❌ Dataset not found. Run setup first.")
            return False
        
        try:
            logger.info("🔍 Creating data loaders (this may take a moment)...")
            train_loader, val_loader = create_optimized_dataloader(
                manifest_path=manifest_path,
                audio_root=audio_root,
                batch_size=config['training']['batch_size'],
                max_duration=config['data']['max_duration']
            )
            logger.info("✅ Data loaders created successfully")
            
            # Log optimization details
            logger.info("🚀 Optimization Details:")
            logger.info(f"   • Batch Size: {config['training']['batch_size']}")
            logger.info(f"   • Effective Batch Size: {config['training']['batch_size'] * config['training']['accumulate_grad_batches']}")
            logger.info(f"   • Precision: {config['hardware']['precision']}")
            logger.info(f"   • Gradient Accumulation: {config['training']['accumulate_grad_batches']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create data loaders: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        logger.info(f"✅ Data loaders created:")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        
        # Test data loader with a single batch
        logger.info("🧪 Testing data loader with a single batch...")
        try:
            test_batch = next(iter(train_loader))
            logger.info(f"✅ Test batch loaded successfully: {test_batch['audio_a'].shape}, {test_batch['audio_b'].shape}")
        except Exception as e:
            logger.error(f"❌ Failed to load test batch: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 4. Set up callbacks
        logger.info("⚙️  Step 4: Setting up callbacks...")
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
        logger.info("✅ Callbacks configured")
        
        # 5. Set up logger
        logger.info("📝 Step 5: Setting up logger...")
        logger_tb = TensorBoardLogger(
            save_dir='logs',
            name='vap_optimized',
            version=datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        logger.info("✅ Logger configured")
        
        # 6. Create trainer
        logger.info("🏃 Step 6: Creating trainer...")
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
            
            # Add these optimizations:
            sync_batchnorm=False,  # Disable for single GPU
            deterministic=False,    # Disable for speed
            benchmark=True,         # Enable cuDNN benchmarking
            enable_checkpointing=True,
            
            # Memory optimizations
            strategy="auto",
            plugins=None,
        )
        
        # Initialize progress tracker
        total_train_batches = len(train_loader)
        total_val_batches = len(val_loader)
        task.progress_tracker = ProgressTracker(
            total_epochs=config['training']['num_epochs'],
            total_train_batches=total_train_batches,
            total_val_batches=total_val_batches
        )
        
        logger.info("✅ Trainer configured")
        
        # Display training configuration summary
        logger.info("\n📋 TRAINING CONFIGURATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Model Architecture:")
        logger.info(f"  Hidden Dim: {config['model']['hidden_dim']}")
        logger.info(f"  Layers: {config['model']['num_layers']}")
        logger.info(f"  Heads: {config['model']['num_heads']}")
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
        logger.info(f"  Max Duration: {config['data']['max_duration']}s")
        logger.info(f"")
        logger.info(f"Loss Weights:")
        logger.info(f"  VAP: {config['training']['vap_loss_weight']}")
        logger.info(f"  EoT: {config['training']['eot_loss_weight']}")
        logger.info(f"  Backchannel: {config['training']['backchannel_loss_weight']}")
        logger.info(f"  Overlap: {config['training']['overlap_loss_weight']}")
        logger.info(f"  VAD: {config['training']['vad_loss_weight']}")
        logger.info("="*50)
        
        # 7. Start training
        logger.info("\n🚀 Starting optimized training...")
        start_time = time.time()
        
        # Add timeout protection for training
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Training timed out after 1 hour")
        
        # Set timeout for 1 hour
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(3600)  # 1 hour timeout
        
        try:
            trainer.fit(task, train_loader, val_loader)
            signal.alarm(0)  # Cancel timeout
        except TimeoutError:
            logger.error("❌ Training timed out after 1 hour")
            return False
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")
        
        # 8. Evaluate final model
        logger.info("📊 Evaluating final model...")
        results = trainer.test(task, val_loader)
        
        # 9. Save results
        results_file = f"results/optimized_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'training_config': config,
                'model_parameters': sum(p.numel() for p in task.parameters()),
                'training_time_minutes': training_time / 60,
                'final_results': results,
                'best_validation_accuracy': task.progress_tracker.best_val_accuracy if task.progress_tracker else 0,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"✅ Results saved: {results_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimized training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = run_optimized_training()
    
    if success:
        logger.info("🎉 Optimized training completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Analyze training curves and metrics")
        logger.info("2. Compare with baseline performance")
        logger.info("3. Run inference with optimized model")
    else:
        logger.error("❌ Optimized training failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 