"""
VAP Turn Detector - Audio-domain turn taking detection using Voice Activity Projection
"""

__version__ = "0.1.0"
__author__ = "VAP Team"

from .models import VAPTurnDetector
from .streaming import StreamingTurnDetector

__all__ = ["VAPTurnDetector", "StreamingTurnDetector"] 