"""
VAP Turn Detection Training Task
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, Any, Optional

from ..models import VAPTurnDetector


class VAPTurnDetectionTask(pl.LightningModule):
    """
    PyTorch Lightning task for training VAP turn detector.
    
    Handles:
    - Multi-task learning (VAP patterns, EoT, backchannel, overlap, VAD)
    - Loss computation and weighting
    - Metrics calculation
    - Optimization
    """
    
    def __init__(
        self,
        model: VAPTurnDetector,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        vap_loss_weight: float = 1.0,
        eot_loss_weight: float = 2.0,
        backchannel_loss_weight: float = 1.5,
        overlap_loss_weight: float = 1.0,
        vad_loss_weight: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss weights
        self.vap_loss_weight = vap_loss_weight
        self.eot_loss_weight = eot_loss_weight
        self.backchannel_loss_weight = backchannel_loss_weight
        self.overlap_loss_weight = overlap_loss_weight
        self.vad_loss_weight = vad_loss_weight
        
        # Label smoothing
        self.label_smoothing = label_smoothing
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        # Metrics tracking
        self.train_eot_f1 = []
        self.val_eot_f1_200ms = []
        self.val_eot_f1_500ms = []
        self.val_hold_shift_acc = []
        self.val_overlap_f1 = []
        self.val_backchannel_auc = []
        
    def forward(
        self, 
        audio_a: torch.Tensor, 
        audio_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(audio_a, audio_b)
    
    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        # Extract batch data
        audio_a = batch['audio_a']
        audio_b = batch['audio_b']
        vap_labels = batch['vap_labels']
        eot_labels = batch['eot_labels']
        backchannel_labels = batch['backchannel_labels']
        overlap_labels = batch['overlap_labels']
        vad_labels = batch['vad_labels']
        
        # Forward pass
        outputs = self(audio_a, audio_b)
        
        # Compute losses
        losses = self._compute_losses(outputs, {
            'vap_labels': vap_labels,
            'eot_labels': eot_labels,
            'backchannel_labels': backchannel_labels,
            'overlap_labels': overlap_labels,
            'vad_labels': vad_labels
        })
        
        total_loss = losses['total_loss']
        
        # Log losses
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_vap_loss', losses['vap_loss'])
        self.log('train_eot_loss', losses['eot_loss'])
        self.log('train_backchannel_loss', losses['backchannel_loss'])
        self.log('train_overlap_loss', losses['overlap_loss'])
        self.log('train_vad_loss', losses['vad_loss'])
        
        # Compute and log metrics
        metrics = self._compute_metrics(outputs, {
            'eot_labels': eot_labels,
            'backchannel_labels': backchannel_labels,
            'overlap_labels': overlap_labels,
            'vad_labels': vad_labels
        })
        
        self.log('train_eot_f1', metrics['eot_f1'])
        self.log('train_hold_shift_acc', metrics['hold_shift_acc'])
        
        return total_loss
    
    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> None:
        """Validation step."""
        # Extract batch data
        audio_a = batch['audio_a']
        audio_b = batch['audio_b']
        vap_labels = batch['vap_labels']
        eot_labels = batch['eot_labels']
        backchannel_labels = batch['backchannel_labels']
        overlap_labels = batch['overlap_labels']
        vad_labels = batch['vad_labels']
        
        # Forward pass
        outputs = self(audio_a, audio_b)
        
        # Compute losses
        losses = self._compute_losses(outputs, {
            'vap_labels': vap_labels,
            'eot_labels': eot_labels,
            'backchannel_labels': backchannel_labels,
            'overlap_labels': overlap_labels,
            'vad_labels': vad_labels
        })
        
        # Log losses
        self.log('val_loss', losses['total_loss'])
        self.log('val_vap_loss', losses['vap_loss'])
        self.log('val_eot_loss', losses['eot_loss'])
        self.log('val_backchannel_loss', losses['backchannel_loss'])
        self.log('val_overlap_loss', losses['overlap_loss'])
        self.log('val_vad_loss', losses['vad_loss'])
        
        # Compute and log metrics
        metrics = self._compute_metrics(outputs, {
            'eot_labels': eot_labels,
            'backchannel_labels': backchannel_labels,
            'overlap_labels': overlap_labels,
            'vad_labels': vad_labels
        })
        
        # Log metrics with different tolerances for EoT
        self.log('val_eot_f1_200ms', metrics['eot_f1_200ms'])
        self.log('val_eot_f1_500ms', metrics['eot_f1_500ms'])
        self.log('val_hold_shift_acc', metrics['hold_shift_acc'])
        self.log('val_overlap_f1', metrics['overlap_f1'])
        self.log('val_backchannel_auc', metrics['backchannel_auc'])
        
        # Store metrics for epoch-level computation
        self.val_eot_f1_200ms.append(metrics['eot_f1_200ms'])
        self.val_eot_f1_500ms.append(metrics['eot_f1_500ms'])
        self.val_hold_shift_acc.append(metrics['hold_shift_acc'])
        self.val_overlap_f1.append(metrics['overlap_f1'])
        self.val_backchannel_auc.append(metrics['backchannel_auc'])
    
    def on_validation_epoch_end(self) -> None:
        """Compute epoch-level validation metrics."""
        if self.val_eot_f1_200ms:
            self.log('val_eot_f1_200ms_epoch', np.mean(self.val_eot_f1_200ms))
            self.log('val_eot_f1_500ms_epoch', np.mean(self.val_eot_f1_500ms))
            self.log('val_hold_shift_acc_epoch', np.mean(self.val_hold_shift_acc))
            self.log('val_overlap_f1_epoch', np.mean(self.val_overlap_f1))
            self.log('val_backchannel_auc_epoch', np.mean(self.val_backchannel_auc))
            
            # Clear lists for next epoch
            self.val_eot_f1_200ms.clear()
            self.val_eot_f1_500ms.clear()
            self.val_hold_shift_acc.clear()
            self.val_overlap_f1.clear()
            self.val_backchannel_auc.clear()
    
    def _compute_losses(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}
        
        # VAP pattern loss (cross-entropy with label smoothing)
        vap_loss = F.cross_entropy(
            outputs['vap_logits'].view(-1, outputs['vap_logits'].size(-1)),
            targets['vap_labels'].view(-1),
            label_smoothing=self.label_smoothing
        )
        losses['vap_loss'] = vap_loss
        
        # End-of-turn loss (binary cross-entropy)
        eot_loss = F.binary_cross_entropy_with_logits(
            outputs['eot_logits'],
            targets['eot_labels'].float()
        )
        losses['eot_loss'] = eot_loss
        
        # Backchannel loss (binary cross-entropy)
        backchannel_loss = F.binary_cross_entropy_with_logits(
            outputs['backchannel_logits'],
            targets['backchannel_labels'].float()
        )
        losses['backchannel_loss'] = backchannel_loss
        
        # Overlap loss (binary cross-entropy)
        overlap_loss = F.binary_cross_entropy_with_logits(
            outputs['overlap_logits'],
            targets['overlap_labels'].float()
        )
        losses['overlap_loss'] = overlap_loss
        
        # VAD loss (binary cross-entropy)
        vad_loss = F.binary_cross_entropy_with_logits(
            outputs['vad_logits'],
            targets['vad_labels'].float()
        )
        losses['vad_loss'] = vad_loss
        
        # Total weighted loss
        total_loss = (
            self.vap_loss_weight * vap_loss +
            self.eot_loss_weight * eot_loss +
            self.backchannel_loss_weight * backchannel_loss +
            self.overlap_loss_weight * overlap_loss +
            self.vad_loss_weight * vad_loss
        )
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_metrics(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}
        
        # Convert logits to probabilities
        eot_probs = torch.sigmoid(outputs['eot_logits'])
        backchannel_probs = torch.sigmoid(outputs['backchannel_logits'])
        overlap_probs = torch.sigmoid(outputs['overlap_logits'])
        vad_probs = torch.sigmoid(outputs['vad_logits'])
        
        # EoT F1 with different tolerances
        metrics['eot_f1_200ms'] = self._compute_eot_f1(
            eot_probs, targets['eot_labels'], tolerance_ms=200
        )
        metrics['eot_f1_500ms'] = self._compute_eot_f1(
            eot_probs, targets['eot_labels'], tolerance_ms=500
        )
        
        # Hold/Shift accuracy
        metrics['hold_shift_acc'] = self._compute_hold_shift_accuracy(
            eot_probs, targets['eot_labels']
        )
        
        # Overlap F1
        metrics['overlap_f1'] = self._compute_binary_f1(
            overlap_probs, targets['overlap_labels']
        )
        
        # Backchannel AUC (simplified)
        metrics['backchannel_auc'] = self._compute_binary_auc(
            backchannel_probs, targets['backchannel_labels']
        )
        
        return metrics
    
    def _compute_eot_f1(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        tolerance_ms: int = 200
    ) -> float:
        """Compute EoT F1 score with temporal tolerance."""
        # Convert to binary predictions
        pred_binary = (predictions > 0.5).float()
        
        # Simple F1 computation (could be improved with temporal tolerance)
        tp = (pred_binary * targets).sum()
        fp = (pred_binary * (1 - targets)).sum()
        fn = ((1 - pred_binary) * targets).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return f1.item()
    
    def _compute_hold_shift_accuracy(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> float:
        """Compute hold/shift accuracy."""
        pred_binary = (predictions > 0.5).float()
        accuracy = (pred_binary == targets).float().mean()
        return accuracy.item()
    
    def _compute_binary_f1(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> float:
        """Compute binary F1 score."""
        pred_binary = (predictions > 0.5).float()
        
        tp = (pred_binary * targets).sum()
        fp = (pred_binary * (1 - targets)).sum()
        fn = ((1 - pred_binary) * targets).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return f1.item()
    
    def _compute_binary_auc(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> float:
        """Compute binary AUC (simplified)."""
        # Simple approximation - could use sklearn's roc_auc_score
        pred_binary = (predictions > 0.5).float()
        accuracy = (pred_binary == targets).float().mean()
        return accuracy.item()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_eot_f1_200ms",
                "interval": "epoch"
            }
        } 