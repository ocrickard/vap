"""
Audio encoders and transformer components for VAP model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Tuple, Optional


class AudioEncoder(nn.Module):
    """
    Audio feature encoder that converts raw audio to hidden representations.
    
    Supports both Log-Mel spectrogram and Neural Codec features.
    Includes downsampling to ~50Hz for efficient processing.
    """
    
    def __init__(
        self,
        feature_dim: int = 80,
        hidden_dim: int = 192,
        sample_rate: int = 16000,
        use_neural_codec: bool = False,
        hop_length: int = 160,  # 10ms at 16kHz
        win_length: int = 400,  # 25ms at 16kHz
        n_mels: int = 80,
        f_min: int = 0,
        f_max: int = 8000,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.sample_rate = sample_rate
        self.use_neural_codec = use_neural_codec
        self.hop_length = hop_length
        
        if use_neural_codec:
            # Neural codec encoder (placeholder for EnCodec integration)
            self.feature_extractor = self._create_neural_codec_encoder()
        else:
            # Log-Mel spectrogram encoder
            self.feature_extractor = self._create_mel_encoder(
                n_mels, f_min, f_max, win_length, hop_length
            )
        
        # Downsampling to ~50Hz (from ~100Hz mel features)
        self.downsampler = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final projection to hidden dimension
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def _create_mel_encoder(
        self, 
        n_mels: int, 
        f_min: int, 
        f_max: int, 
        win_length: int, 
        hop_length: int
    ) -> nn.Module:
        """Create Log-Mel spectrogram encoder."""
        return nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=1024,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                mel_scale='htk'
            ),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0)
        )
    
    def _create_neural_codec_encoder(self) -> nn.Module:
        """Create neural codec encoder (placeholder for EnCodec integration)."""
        # This would integrate with EnCodec or similar neural codec
        # For now, return a placeholder that raises an error
        raise NotImplementedError(
            "Neural codec encoder not yet implemented. "
            "Use Log-Mel features for now."
        )
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to hidden representations.
        
        Args:
            audio: [batch, time] raw audio tensor
            
        Returns:
            [batch, time, hidden_dim] encoded features
        """
        batch_size, seq_len = audio.shape
        
        if self.use_neural_codec:
            # Neural codec path (placeholder)
            features = self.feature_extractor(audio)
        else:
            # Log-Mel path
            # Apply mel spectrogram
            mel_spec = self.feature_extractor(audio)  # [batch, n_mels, time]
            
            # Normalize features
            mel_spec = (mel_spec - mel_spec.mean(dim=1, keepdim=True)) / (
                mel_spec.std(dim=1, keepdim=True) + 1e-8
            )
            
            features = mel_spec
        
        # Downsample to ~50Hz
        # Input: [batch, feature_dim, time]
        # Output: [batch, hidden_dim, time//4] (due to stride=2 twice)
        downsampled = self.downsampler(features)
        
        # Transpose to [batch, time, hidden_dim]
        downsampled = downsampled.transpose(1, 2)
        
        # Project to final hidden dimension
        encoded = self.projection(downsampled)
        encoded = self.layer_norm(encoded)
        
        return encoded


class CrossAttentionTransformer(nn.Module):
    """
    Cross-attention transformer for fusing speaker representations.
    
    Uses efficient attention mechanisms for real-time processing.
    """
    
    def __init__(
        self,
        hidden_dim: int = 192,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_relative_pos: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_relative_pos = use_relative_pos
        
        # Speaker-specific encoders
        self.speaker_encoder_a = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.speaker_encoder_b = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Relative positional encoding
        if use_relative_pos:
            self.rel_pos_encoder = RelativePositionalEncoding(hidden_dim, max_len=1000)
        
    def forward(
        self, 
        features_a: torch.Tensor, 
        features_b: torch.Tensor,
        return_attn: bool = False
    ) -> torch.Tensor:
        """
        Apply cross-attention between speaker representations.
        
        Args:
            features_a: [batch, time, hidden_dim] Speaker A features
            features_b: [batch, time, hidden_dim] Speaker B features
            return_attn: Whether to return attention weights
            
        Returns:
            [batch, time, hidden_dim] Fused speaker features
        """
        batch_size, seq_len, hidden_dim = features_a.shape
        
        # Encode each speaker independently
        encoded_a = self.speaker_encoder_a(features_a)
        encoded_b = self.speaker_encoder_b(features_b)
        
        # Simple fusion for now (skip complex cross-attention)
        fused_features = self.fusion_layer(
            torch.cat([encoded_a, encoded_b], dim=-1)
        )
        
        if return_attn:
            return fused_features, attn_weights
        else:
            return fused_features


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block between speakers."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Cross-attention: A attends to B, B attends to A
        self.cross_attn_a_to_b = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attn_b_to_a = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward networks
        self.ffn_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.ffn_b = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer norms
        self.norm1_a = nn.LayerNorm(hidden_dim)
        self.norm1_b = nn.LayerNorm(hidden_dim)
        self.norm2_a = nn.LayerNorm(hidden_dim)
        self.norm2_b = nn.LayerNorm(hidden_dim)
        
    def forward(
        self, 
        features_a: torch.Tensor, 
        features_b: torch.Tensor,
        return_attn: bool = False
    ) -> torch.Tensor:
        """Apply cross-attention between speakers."""
        
        # Cross-attention: A attends to B
        if return_attn:
            attn_a, attn_weights_a = self.cross_attn_a_to_b(
                features_a, features_b, features_b, return_attn=True
            )
        else:
            attn_a = self.cross_attn_a_to_b(features_a, features_b, features_b)
        
        # Cross-attention: B attends to A
        if return_attn:
            attn_b, attn_weights_b = self.cross_attn_b_to_a(
                features_b, features_a, features_a, return_attn=True
            )
        else:
            attn_b = self.cross_attn_b_to_a(features_b, features_a, features_a)
        
        # Residual connections and layer norms
        features_a = self.norm1_a(features_a + attn_a)
        features_b = self.norm1_b(features_b + attn_b)
        
        # Feed-forward networks
        ffn_a = self.ffn_a(features_a)
        ffn_b = self.ffn_b(features_b)
        
        # Final residual connections
        features_a = self.norm2_a(features_a + ffn_a)
        features_b = self.norm2_b(features_b + ffn_b)
        
        if return_attn:
            return features_a, features_b, attn_weights_a, attn_weights_b
        else:
            return features_a, features_b


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for efficient long sequences."""
    
    def __init__(self, hidden_dim: int, max_len: int = 1000):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        # Create relative position embeddings
        self.rel_pos_embeddings = nn.Parameter(
            torch.randn(2 * max_len + 1, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add relative positional encoding to input."""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Create relative position indices
        pos_indices = torch.arange(seq_len, device=x.device)
        rel_pos_indices = pos_indices.unsqueeze(1) - pos_indices.unsqueeze(0)
        rel_pos_indices = rel_pos_indices + self.max_len
        
        # Get relative position embeddings
        rel_pos_emb = self.rel_pos_embeddings[rel_pos_indices]
        
        return x + rel_pos_emb 