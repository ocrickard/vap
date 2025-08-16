"""
Prediction heads for VAP turn detection model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class VAPHead(nn.Module):
    """
    Voice Activity Projection head that predicts future speech activity patterns.
    
    Outputs a distribution over 2^num_speakers * num_time_bins possible patterns.
    For 2 speakers and 10 time bins, this gives 2^20 = 1,048,576 classes.
    To make this tractable, we use a hierarchical approach.
    """
    
    def __init__(
        self,
        hidden_dim: int = 192,
        num_time_bins: int = 10,
        num_speakers: int = 2,
        use_hierarchical: bool = True,
        max_patterns: int = 256,  # Limit for tractable training
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_time_bins = num_time_bins
        self.num_speakers = num_speakers
        self.use_hierarchical = use_hierarchical
        self.max_patterns = max_patterns
        
        if use_hierarchical:
            # Hierarchical approach: predict per-speaker, per-time-bin independently
            self.speaker_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, num_time_bins)
                )
                for _ in range(num_speakers)
            ])
        else:
            # Direct approach: predict all patterns at once (limited by max_patterns)
            if 2 ** (num_speakers * num_time_bins) > max_patterns:
                raise ValueError(
                    f"Too many patterns ({2 ** (num_speakers * num_time_bins)}) "
                    f"for direct prediction. Use hierarchical approach."
                )
            
            self.pattern_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, 2 ** (num_speakers * num_time_bins))
            )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict future speech activity patterns.
        
        Args:
            features: [batch, time, hidden_dim] fused speaker features
            
        Returns:
            [batch, time, num_patterns] logits for VAP patterns
        """
        if self.use_hierarchical:
            # Predict per-speaker, per-time-bin independently
            speaker_logits = []
            for speaker_head in self.speaker_heads:
                speaker_logits.append(speaker_head(features))  # [batch, time, num_time_bins]
            
            # Stack speaker predictions
            # Shape: [batch, time, num_speakers, num_time_bins]
            stacked_logits = torch.stack(speaker_logits, dim=2)
            
            # Reshape to [batch, time, num_speakers * num_time_bins]
            batch_size, seq_len, _, _ = stacked_logits.shape
            return stacked_logits.view(batch_size, seq_len, -1)
        else:
            # Direct pattern prediction
            return self.pattern_head(features)


class TurnTakingHead(nn.Module):
    """
    Turn-taking specific prediction heads.
    
    Predicts:
    - End-of-turn (EoT) probability
    - Backchannel likelihood
    - Overlap detection
    - Per-speaker VAD
    """
    
    def __init__(
        self,
        hidden_dim: int = 192,
        num_time_bins: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_time_bins = num_time_bins
        
        # Shared feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # End-of-turn prediction
        self.eot_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Backchannel detection
        self.backchannel_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Overlap detection
        self.overlap_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Per-speaker VAD prediction
        self.vad_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2)  # 2 speakers
        )
        
        # Future VAD prediction (using time bins)
        self.future_vad_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_time_bins * 2)  # 2 speakers * time bins
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict turn-taking events.
        
        Args:
            features: [batch, time, hidden_dim] fused speaker features
            
        Returns:
            Dictionary with prediction logits
        """
        # Process features
        processed_features = self.feature_processor(features)
        
        # Make predictions
        eot_logits = self.eot_head(processed_features)  # [batch, time, 1]
        backchannel_logits = self.backchannel_head(processed_features)  # [batch, time, 1]
        overlap_logits = self.overlap_head(processed_features)  # [batch, time, 1]
        vad_logits = self.vad_head(processed_features)  # [batch, time, 2]
        future_vad_logits = self.future_vad_head(processed_features)  # [batch, time, num_time_bins*2]
        
        return {
            'eot_logits': eot_logits.squeeze(-1),  # [batch, time]
            'backchannel_logits': backchannel_logits.squeeze(-1),  # [batch, time]
            'overlap_logits': overlap_logits.squeeze(-1),  # [batch, time]
            'vad_logits': vad_logits,  # [batch, time, 2]
            'future_vad_logits': future_vad_logits  # [batch, time, num_time_bins*2]
        }


class VAPPatternEncoder:
    """
    Utility class for encoding/decoding VAP patterns.
    
    Converts between binary VAD patterns and integer class indices.
    """
    
    def __init__(self, num_speakers: int = 2, num_time_bins: int = 10):
        self.num_speakers = num_speakers
        self.num_time_bins = num_time_bins
        self.num_patterns = 2 ** (num_speakers * num_time_bins)
        
        # Create pattern lookup tables
        self.pattern_to_idx = {}
        self.idx_to_pattern = {}
        
        for idx in range(min(self.num_patterns, 10000)):  # Limit for memory
            pattern = self._idx_to_binary_pattern(idx)
            self.pattern_to_idx[pattern] = idx
            self.idx_to_pattern[idx] = pattern
    
    def _idx_to_binary_pattern(self, idx: int) -> str:
        """Convert integer index to binary VAD pattern string."""
        pattern_length = self.num_speakers * self.num_time_bins
        binary = format(idx, f'0{pattern_length}b')
        return binary
    
    def encode_pattern(self, vad_pattern: torch.Tensor) -> torch.Tensor:
        """
        Encode VAD pattern tensor to class indices.
        
        Args:
            vad_pattern: [batch, time, num_speakers, num_time_bins] binary tensor
            
        Returns:
            [batch, time] class indices
        """
        batch_size, seq_len, _, _ = vad_pattern.shape
        
        # Flatten speaker and time dimensions
        flattened = vad_pattern.view(batch_size, seq_len, -1)
        
        # Convert to binary strings
        binary_strings = []
        for b in range(batch_size):
            for t in range(seq_len):
                pattern = ''.join(['1' if x > 0.5 else '0' for x in flattened[b, t]])
                binary_strings.append(pattern)
        
        # Convert to indices
        indices = []
        for pattern in binary_strings:
            if pattern in self.pattern_to_idx:
                indices.append(self.pattern_to_idx[pattern])
            else:
                indices.append(0)  # Default to first pattern
        
        return torch.tensor(indices).view(batch_size, seq_len)
    
    def decode_pattern(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode class indices back to VAD patterns.
        
        Args:
            indices: [batch, time] class indices
            
        Returns:
            [batch, time, num_speakers, num_time_bins] binary tensor
        """
        batch_size, seq_len = indices.shape
        
        patterns = []
        for b in range(batch_size):
            for t in range(seq_len):
                idx = indices[b, t].item()
                if idx in self.idx_to_pattern:
                    pattern_str = self.idx_to_pattern[idx]
                    pattern = torch.tensor([
                        int(bit) for bit in pattern_str
                    ], dtype=torch.float32)
                else:
                    pattern = torch.zeros(self.num_speakers * self.num_time_bins)
                
                patterns.append(pattern)
        
        # Reshape to [batch, time, num_speakers, num_time_bins]
        stacked = torch.stack(patterns).view(batch_size, seq_len, -1)
        return stacked.view(batch_size, seq_len, self.num_speakers, self.num_time_bins) 