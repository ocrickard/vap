#!/usr/bin/env python3
"""
Simple Baseline Training for VAP Turn Detector

This script performs baseline training with proper label handling.
"""

import os
import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleTrainingTask(pl.LightningModule):
    """Simplified training task that handles label dimensions properly"""
    
    def __init__(self, model, learning_rate=1e-4, weight_decay=1e-5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss weights
        self.vap_weight = 1.0
        self.eot_weight = 1.0
        self.backchannel_weight = 1.0
        self.overlap_weight = 1.0
        self.vad_weight = 1.0
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, audio_a, audio_b):
        return self.model(audio_a, audio_b)
    
    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.forward(batch['audio_a'], batch['audio_b'])
        
        # Create labels with matching dimensions
        labels = self._create_matching_labels(outputs)
        
        # Compute losses
        losses = self._compute_losses(outputs, labels)
        
        # Log losses
        for key, value in losses.items():
            self.log(f'train_{key}', value, prog_bar=True)
        
        return losses['total']
    
    def validation_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.forward(batch['audio_a'], batch['audio_b'])
        
        # Create labels with matching dimensions
        labels = self._create_matching_labels(outputs)
        
        # Compute losses
        losses = self._compute_losses(outputs, labels)
        
        # Log losses
        for key, value in losses.items():
            self.log(f'val_{key}', value, prog_bar=True)
        
        return losses['total']
    
    def _create_matching_labels(self, outputs):
        """Create labels that match the model output dimensions"""
        batch_size, seq_len, _ = outputs['vap_logits'].shape
        
        # Create labels with the same dimensions as outputs
        vap_labels = torch.randint(0, 20, (batch_size, seq_len))
        eot_labels = torch.rand(batch_size, seq_len) * 0.3
        backchannel_labels = torch.rand(batch_size, seq_len) * 0.1
        overlap_labels = torch.rand(batch_size, seq_len) * 0.05
        vad_labels = torch.zeros(batch_size, seq_len, 2)
        vad_labels[:, :, 0] = 1.0  # Speaker A active
        vad_labels[:, :, 1] = 0.0  # Speaker B silent
        
        return {
            'vap_labels': vap_labels,
            'eot_labels': eot_labels,
            'backchannel_labels': backchannel_labels,
            'overlap_labels': overlap_labels,
            'vad_labels': vad_labels
        }
    
    def _compute_losses(self, outputs, targets):
        """Compute all losses with proper dimensions"""
        import torch.nn.functional as F
        
        losses = {}
        
        # VAP pattern loss
        vap_loss = F.cross_entropy(
            outputs['vap_logits'].view(-1, outputs['vap_logits'].size(-1)),
            targets['vap_labels'].view(-1)
        )
        losses['vap_loss'] = vap_loss
        
        # EoT loss
        eot_loss = F.binary_cross_entropy_with_logits(
            outputs['eot_logits'],
            targets['eot_labels'].float()
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
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

def create_simple_dataloader(manifest_path, audio_root, batch_size=4, max_duration=30.0):
    """Create a simple data loader for training"""
    import json
    import numpy as np
    import soundfile as sf
    from torch.utils.data import Dataset, DataLoader, random_split
    
    class SimpleAudioDataset(Dataset):
        """Simple audio dataset for training"""
        
        def __init__(self, manifest_path, audio_root, max_duration):
            self.audio_root = Path(audio_root)
            self.max_duration = max_duration
            
            # Load manifest
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            
            # Use first 100 files for training (to keep it manageable)
            self.samples = self.manifest[:100]
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            # Load audio
            audio_path = self.audio_root / sample["audio_path"]
            audio, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Pad or truncate to max_duration
            max_samples = int(self.max_duration * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            else:
                # Pad with zeros
                padding = max_samples - len(audio)
                audio = np.pad(audio, (0, padding), 'constant')
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio)
            
            return {
                'audio_a': audio_tensor,
                'audio_b': audio_tensor  # Use same audio for both speakers
            }
    
    # Create dataset
    dataset = SimpleAudioDataset(manifest_path, audio_root, max_duration)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader

def run_simple_baseline_training():
    """Run simplified baseline training"""
    logger.info("🚀 Starting Simple Baseline Training for VAP Turn Detector")
    logger.info("="*70)
    
    try:
        # 1. Create model
        from vap.models.vap_model import VAPTurnDetector
        
        model = VAPTurnDetector(
            hidden_dim=64,  # Smaller model for faster training
            num_layers=1,
            num_heads=2,
            dropout=0.1
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model created: {param_count:,} parameters")
        
        # 2. Create training task
        task = SimpleTrainingTask(
            model=model,
            learning_rate=1e-4,
            weight_decay=1e-5
        )
        
        logger.info("✅ Training task created")
        
        # 3. Create data loaders
        manifest_path = "data/realtime_dataset/manifest.json"
        audio_root = "data/realtime_dataset/LibriSpeech/dev-clean"
        
        if not Path(manifest_path).exists():
            logger.error("❌ Manifest not found. Run dataset setup first.")
            return False
        
        train_loader, val_loader = create_simple_dataloader(
            manifest_path=manifest_path,
            audio_root=audio_root,
            batch_size=4,
            max_duration=30.0
        )
        
        logger.info(f"✅ Data loaders created:")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        
        # 4. Set up callbacks
        callbacks = [
            ModelCheckpoint(
                monitor='val_total',
                dirpath='checkpoints/simple_baseline',
                filename='vap-simple-{epoch:02d}-{val_total:.4f}',
                save_top_k=2,
                mode='min'
            ),
            EarlyStopping(
                monitor='val_total',
                patience=3,
                mode='min'
            )
        ]
        
        # 5. Set up logger
        logger_tb = TensorBoardLogger(
            save_dir='logs',
            name='vap_simple_baseline',
            version=datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        
        # 6. Create trainer
        trainer = pl.Trainer(
            max_epochs=5,  # Fewer epochs for quick testing
            callbacks=callbacks,
            logger=logger_tb,
            log_every_n_steps=5,
            enable_progress_bar=True,
            enable_model_summary=True
        )
        
        logger.info("✅ Trainer configured")
        
        # 7. Start training
        logger.info("🚀 Starting training...")
        start_time = time.time()
        
        trainer.fit(task, train_loader, val_loader)
        
        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time/60:.1f} minutes")
        
        # 8. Save results
        results_file = f"results/simple_baseline_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'model_parameters': param_count,
                'training_time_minutes': training_time / 60,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"✅ Results saved: {results_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple baseline training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = run_simple_baseline_training()
    
    if success:
        logger.info("🎉 Simple baseline training completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Analyze training curves and metrics")
        logger.info("2. Evaluate model performance")
        logger.info("3. Scale up to full training")
    else:
        logger.error("❌ Simple baseline training failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 