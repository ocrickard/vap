"""
Main VAP Turn Detector model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .encoders import AudioEncoder, CrossAttentionTransformer
from .heads import VAPHead, TurnTakingHead


class VAPTurnDetector(nn.Module):
    """
    Voice Activity Projection (VAP) based turn detector.
    
    This model predicts future speech activity patterns from raw audio streams,
    enabling early turn-taking decisions with low latency.
    
    Architecture:
    1. Audio feature extraction (Log-Mel or Neural Codec)
    2. Conv downsampling to ~50Hz
    3. Cross-attention transformer between speaker channels
    4. Multi-head prediction (VAP patterns, EoT, backchannel, overlap)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        feature_dim: int = 80,
        hidden_dim: int = 192,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        future_horizon: float = 2.0,  # seconds
        time_bins: list = None,  # [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        use_neural_codec: bool = False,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.future_horizon = future_horizon
        
        # Default time bins for 2-second horizon
        if time_bins is None:
            self.time_bins = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        else:
            self.time_bins = time_bins
            
        self.num_time_bins = len(self.time_bins)
        
        # Audio feature encoder
        self.audio_encoder = AudioEncoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            use_neural_codec=use_neural_codec
        )
        
        # Cross-attention transformer
        self.transformer = CrossAttentionTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Prediction heads
        self.vap_head = VAPHead(
            hidden_dim=hidden_dim,
            num_time_bins=self.num_time_bins,
            num_speakers=2
        )
        
        self.turn_taking_head = TurnTakingHead(
            hidden_dim=hidden_dim,
            num_time_bins=self.num_time_bins
        )
        
        # Positional encoding will be created dynamically based on sequence length
        self.max_seq_len = 1000  # Maximum expected sequence length
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create ALiBi-style positional encoding for efficient long sequences."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def forward(
        self,
        audio_a: torch.Tensor,  # [batch, time] - Speaker A audio
        audio_b: torch.Tensor,  # [batch, time] - Speaker B audio
        return_attn: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VAP model.
        
        Args:
            audio_a: Speaker A audio tensor
            audio_b: Speaker B audio tensor
            return_attn: Whether to return attention weights
            
        Returns:
            Dictionary containing:
            - vap_logits: [batch, time, num_patterns] - VAP pattern predictions
            - eot_logits: [batch, time] - End-of-turn probabilities
            - backchannel_logits: [batch, time] - Backchannel likelihoods
            - overlap_logits: [batch, time] - Overlap detection
            - vad_logits: [batch, time, 2] - Per-speaker VAD
        """
        batch_size = audio_a.shape[0]
        seq_len = audio_a.shape[1]
        
        # Extract audio features
        features_a = self.audio_encoder(audio_a)  # [batch, time, hidden_dim]
        features_b = self.audio_encoder(audio_b)  # [batch, time, hidden_dim]
        
        # Get actual sequence length after encoding
        actual_seq_len = features_a.shape[1]
        
        # Create simple positional encoding for actual sequence length
        pos_embedding = torch.arange(actual_seq_len, dtype=torch.float32, device=features_a.device)
        pos_embedding = pos_embedding.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        
        # Add positional encoding (scaled down to avoid overwhelming the features)
        features_a = features_a + 0.01 * pos_embedding.expand(-1, -1, self.hidden_dim)
        features_b = features_b + 0.01 * pos_embedding.expand(-1, -1, self.hidden_dim)
        
        # Cross-attention between speakers
        if return_attn:
            fused_features, attn_weights = self.transformer(
                features_a, features_b, return_attn=True
            )
        else:
            fused_features = self.transformer(features_a, features_b)
        
        # VAP pattern prediction (future speech activity)
        vap_logits = self.vap_head(fused_features)  # [batch, time, num_patterns]
        
        # Turn-taking specific predictions
        turn_outputs = self.turn_taking_head(fused_features)
        
        # Combine outputs
        outputs = {
            'vap_logits': vap_logits,
            'eot_logits': turn_outputs['eot_logits'],
            'backchannel_logits': turn_outputs['backchannel_logits'],
            'overlap_logits': turn_outputs['overlap_logits'],
            'vad_logits': turn_outputs['vad_logits']
        }
        
        if return_attn:
            outputs['attention_weights'] = attn_weights
            
        return outputs
    
    def predict_turn_shift(
        self,
        audio_a: torch.Tensor,
        audio_b: torch.Tensor,
        threshold: float = 0.5,
        min_gap_ms: int = 200
    ) -> Dict[str, torch.Tensor]:
        """
        Predict turn shift events from audio streams.
        
        Args:
            audio_a: Speaker A audio
            audio_b: Speaker B audio
            threshold: Probability threshold for turn shift
            min_gap_ms: Minimum gap in ms before confirming turn shift
            
        Returns:
            Dictionary with turn shift predictions and confidence scores
        """
        with torch.no_grad():
            outputs = self.forward(audio_a, audio_b)
            
            # Get probabilities
            eot_probs = torch.sigmoid(outputs['eot_logits'])
            backchannel_probs = torch.sigmoid(outputs['backchannel_logits'])
            overlap_probs = torch.sigmoid(outputs['overlap_logits'])
            
            # VAP pattern probabilities
            vap_probs = F.softmax(outputs['vap_logits'], dim=-1)
            
            # Detect turn shifts
            turn_shift = (eot_probs > threshold) & (backchannel_probs < 0.3)
            
            # Apply minimum gap constraint
            if min_gap_ms > 0:
                min_gap_frames = int(min_gap_ms * self.sample_rate / 1000 / 160)  # Assuming 160 hop
                # Simple temporal smoothing - could be improved with HMM
                turn_shift = self._apply_min_gap_constraint(turn_shift, min_gap_frames)
            
            return {
                'turn_shift': turn_shift,
                'eot_probability': eot_probs,
                'backchannel_probability': backchannel_probs,
                'overlap_probability': overlap_probs,
                'vap_patterns': vap_probs
            }
    
    def _apply_min_gap_constraint(
        self, 
        turn_shift: torch.Tensor, 
        min_gap_frames: int
    ) -> torch.Tensor:
        """Apply minimum gap constraint to turn shift predictions."""
        # Simple implementation - could be improved with proper HMM
        batch_size, seq_len = turn_shift.shape
        constrained = turn_shift.clone()
        
        for b in range(batch_size):
            last_shift = -min_gap_frames
            for t in range(seq_len):
                if turn_shift[b, t]:
                    if t - last_shift < min_gap_frames:
                        constrained[b, t] = False
                    else:
                        last_shift = t
                        
        return constrained
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        } 