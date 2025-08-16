"""
Streaming turn detector for real-time inference
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque
import time

from ..models import VAPTurnDetector
from .ring_buffer import AudioRingBuffer


class StreamingTurnDetector:
    """
    Streaming turn detector that processes audio in real-time.
    
    Maintains a ring buffer of audio and provides incremental updates
    for turn-taking decisions.
    """
    
    def __init__(
        self,
        model: VAPTurnDetector,
        buffer_duration: float = 6.0,  # seconds
        hop_duration: float = 0.02,    # 20ms hop
        sample_rate: int = 16000,
        device: str = "cpu",
        smoothing_window: int = 5,      # frames for smoothing
        eot_threshold: float = 0.5,
        backchannel_threshold: float = 0.3,
        overlap_threshold: float = 0.4,
        min_gap_ms: int = 200,
    ):
        self.model = model.to(device)
        self.device = device
        self.sample_rate = sample_rate
        self.hop_duration = hop_duration
        self.smoothing_window = smoothing_window
        
        # Thresholds
        self.eot_threshold = eot_threshold
        self.backchannel_threshold = backchannel_threshold
        self.overlap_threshold = overlap_threshold
        self.min_gap_ms = min_gap_ms
        
        # Audio buffers
        self.buffer_duration = buffer_duration
        self.buffer_size = int(buffer_duration * sample_rate)
        self.hop_size = int(hop_duration * sample_rate)
        
        # Ring buffers for each speaker
        self.buffer_a = AudioRingBuffer(self.buffer_size)
        self.buffer_b = AudioRingBuffer(self.buffer_size)
        
        # State tracking
        self.last_eot_prob = 0.0
        self.last_backchannel_prob = 0.0
        self.last_overlap_prob = 0.0
        
        # Smoothing buffers
        self.eot_history = deque(maxlen=smoothing_window)
        self.backchannel_history = deque(maxlen=smoothing_window)
        self.overlap_history = deque(maxlen=smoothing_window)
        
        # Turn state
        self.current_speaker = "A"  # or "B"
        self.last_turn_shift_time = 0.0
        self.is_speaking = False
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        
        # Set model to evaluation mode
        self.model.eval()
        
    def add_audio(
        self, 
        audio_a: np.ndarray, 
        audio_b: np.ndarray,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Add new audio frames to the buffers.
        
        Args:
            audio_a: Speaker A audio samples
            audio_b: Speaker B audio samples
            timestamp: Optional timestamp for the audio
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Add to ring buffers
        self.buffer_a.add_samples(audio_a)
        self.buffer_b.add_samples(audio_b)
        
    def step(self) -> Dict[str, float]:
        """
        Process the current audio buffers and return predictions.
        
        Returns:
            Dictionary with current predictions and state
        """
        start_time = time.time()
        
        # Get current buffer contents
        audio_a = self.buffer_a.get_samples()
        audio_b = self.buffer_b.get_samples()
        
        if len(audio_a) < self.hop_size or len(audio_b) < self.hop_size:
            # Not enough audio yet
            return self._get_default_output()
        
        # Convert to tensors
        audio_a_tensor = torch.from_numpy(audio_a).float().unsqueeze(0).to(self.device)
        audio_b_tensor = torch.from_numpy(audio_b).float().unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(audio_a_tensor, audio_b_tensor)
            
            # Get probabilities
            eot_prob = torch.sigmoid(outputs['eot_logits']).cpu().numpy()[0, -1]
            backchannel_prob = torch.sigmoid(outputs['backchannel_logits']).cpu().numpy()[0, -1]
            overlap_prob = torch.sigmoid(outputs['overlap_logits']).cpu().numpy()[0, -1]
            
            # Get VAD probabilities
            vad_probs = torch.sigmoid(outputs['vad_logits']).cpu().numpy()[0, -1]  # [2]
            speaker_a_vad = vad_probs[0]
            speaker_b_vad = vad_probs[1]
        
        # Apply smoothing
        self.eot_history.append(eot_prob)
        self.backchannel_history.append(backchannel_prob)
        self.overlap_history.append(overlap_prob)
        
        # Get smoothed values
        smoothed_eot = np.mean(list(self.eot_history))
        smoothed_backchannel = np.mean(list(self.backchannel_history))
        smoothed_overlap = np.mean(list(self.overlap_history))
        
        # Update state
        self._update_turn_state(smoothed_eot, smoothed_backchannel, smoothed_overlap)
        
        # Record inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # Prepare output
        output = {
            'eot_probability': float(smoothed_eot),
            'backchannel_probability': float(smoothed_backchannel),
            'overlap_probability': float(smoothed_overlap),
            'speaker_a_vad': float(speaker_a_vad),
            'speaker_b_vad': float(speaker_b_vad),
            'current_speaker': self.current_speaker,
            'is_speaking': self.is_speaking,
            'turn_shift_detected': self._detect_turn_shift(smoothed_eot, smoothed_backchannel),
            'inference_time_ms': inference_time * 1000,
            'buffer_fill_percentage': self.buffer_a.get_fill_percentage()
        }
        
        return output
    
    def _update_turn_state(
        self, 
        eot_prob: float, 
        backchannel_prob: float, 
        overlap_prob: float
    ) -> None:
        """Update the current turn-taking state."""
        current_time = time.time()
        
        # Check if current speaker is still speaking
        if self.current_speaker == "A":
            self.is_speaking = eot_prob < self.eot_threshold
        else:
            self.is_speaking = eot_prob < self.eot_threshold
        
        # Detect turn shift
        if self._detect_turn_shift(eot_prob, backchannel_prob):
            if current_time - self.last_turn_shift_time > (self.min_gap_ms / 1000.0):
                # Switch speakers
                self.current_speaker = "B" if self.current_speaker == "A" else "A"
                self.last_turn_shift_time = current_time
                self.is_speaking = True
    
    def _detect_turn_shift(self, eot_prob: float, backchannel_prob: float) -> bool:
        """Detect if a turn shift should occur."""
        return (eot_prob > self.eot_threshold and 
                backchannel_prob < self.backchannel_threshold)
    
    def _get_default_output(self) -> Dict[str, float]:
        """Return default output when not enough audio is available."""
        return {
            'eot_probability': 0.0,
            'backchannel_probability': 0.0,
            'overlap_probability': 0.0,
            'speaker_a_vad': 0.0,
            'speaker_b_vad': 0.0,
            'current_speaker': self.current_speaker,
            'is_speaking': self.is_speaking,
            'turn_shift_detected': False,
            'inference_time_ms': 0.0,
            'buffer_fill_percentage': self.buffer_a.get_fill_percentage()
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.inference_times:
            return {}
        
        times = list(self.inference_times)
        return {
            'avg_inference_time_ms': np.mean(times) * 1000,
            'max_inference_time_ms': np.max(times) * 1000,
            'min_inference_time_ms': np.min(times) * 1000,
            'std_inference_time_ms': np.std(times) * 1000,
            'total_inferences': len(times)
        }
    
    def reset(self) -> None:
        """Reset the detector state."""
        self.buffer_a.clear()
        self.buffer_b.clear()
        self.eot_history.clear()
        self.backchannel_history.clear()
        self.overlap_history.clear()
        self.current_speaker = "A"
        self.last_turn_shift_time = 0.0
        self.is_speaking = False
        self.inference_times.clear()
    
    def set_thresholds(
        self,
        eot_threshold: Optional[float] = None,
        backchannel_threshold: Optional[float] = None,
        overlap_threshold: Optional[float] = None,
        min_gap_ms: Optional[int] = None
    ) -> None:
        """Update detection thresholds."""
        if eot_threshold is not None:
            self.eot_threshold = eot_threshold
        if backchannel_threshold is not None:
            self.backchannel_threshold = backchannel_threshold
        if overlap_threshold is not None:
            self.overlap_threshold = overlap_threshold
        if min_gap_ms is not None:
            self.min_gap_ms = min_gap_ms 