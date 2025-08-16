"""
Model architectures for VAP turn detection
"""

from .vap_model import VAPTurnDetector
from .encoders import AudioEncoder, CrossAttentionTransformer
from .heads import VAPHead, TurnTakingHead

__all__ = [
    "VAPTurnDetector",
    "AudioEncoder", 
    "CrossAttentionTransformer",
    "VAPHead",
    "TurnTakingHead"
] 