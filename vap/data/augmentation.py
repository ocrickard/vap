"""
Audio Augmentation Module for VAP Turn Detector

This module provides various audio augmentation techniques to improve model robustness
and generalization during training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AudioAugmenter:
    """Audio augmentation class with multiple augmentation techniques"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        noise_snr_range: Tuple[float, float] = (5, 20),
        speed_range: Tuple[float, float] = (0.9, 1.1),
        pitch_range: Tuple[float, float] = (-2, 2),
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        reverb_prob: float = 0.3,
        noise_prob: float = 0.5
    ):
        self.sample_rate = sample_rate
        self.noise_snr_range = noise_snr_range
        self.speed_range = speed_range
        self.pitch_range = pitch_range
        self.time_stretch_range = time_stretch_range
        self.reverb_prob = reverb_prob
        self.noise_prob = noise_prob
        
        # Initialize noise samples (MUSAN-like)
        self._init_noise_samples()
    
    def _init_noise_samples(self):
        """Initialize noise samples for augmentation"""
        # Create synthetic noise samples (in practice, you'd load real MUSAN samples)
        self.noise_samples = []
        
        # White noise
        white_noise = torch.randn(16000 * 10) * 0.1  # 10 seconds
        self.noise_samples.append(white_noise)
        
        # Pink noise (filtered white noise)
        pink_noise = self._create_pink_noise(16000 * 10)
        self.noise_samples.append(pink_noise)
        
        # Brown noise
        brown_noise = self._create_brown_noise(16000 * 10)
        self.noise_samples.append(brown_noise)
        
        logger.info(f"✅ Initialized {len(self.noise_samples)} noise samples")
    
    def _create_pink_noise(self, length: int) -> torch.Tensor:
        """Create pink noise using spectral filtering"""
        # Generate white noise
        white = torch.randn(length)
        
        # Apply 1/f filter in frequency domain
        fft = torch.fft.rfft(white)
        freqs = torch.fft.rfftfreq(length, 1/self.sample_rate)
        
        # 1/f filter (avoid division by zero)
        filter_response = 1.0 / torch.sqrt(freqs + 1e-8)
        filter_response = filter_response / filter_response.max()
        
        # Apply filter and convert back
        filtered_fft = fft * filter_response
        pink = torch.fft.irfft(filtered_fft, n=length)
        
        return pink * 0.1  # Normalize
    
    def _create_brown_noise(self, length: int) -> torch.Tensor:
        """Create brown noise using cumulative sum"""
        white = torch.randn(length) * 0.1
        brown = torch.cumsum(white, dim=0)
        brown = brown - brown.mean()
        return brown * 0.05  # Normalize
    
    def augment_audio(self, audio: torch.Tensor, augment: bool = True) -> torch.Tensor:
        """Apply audio augmentation to input audio"""
        if not augment:
            return audio
        
        augmented = audio.clone()
        
        # Apply augmentations with probabilities
        if random.random() < self.noise_prob:
            augmented = self._add_noise(augmented)
        
        if random.random() < self.reverb_prob:
            augmented = self._add_reverb(augmented)
        
        if random.random() < 0.5:
            augmented = self._speed_perturb(augmented)
        
        if random.random() < 0.5:
            augmented = self._pitch_shift(augmented)
        
        if random.random() < 0.3:
            augmented = self._time_stretch(augmented)
        
        return augmented
    
    def _add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Add noise to audio with specified SNR"""
        # Select random noise sample
        noise = random.choice(self.noise_samples)
        
        # Ensure noise is the same length as audio
        if len(noise) < len(audio):
            # Repeat noise if too short
            repeats = (len(audio) // len(noise)) + 1
            noise = noise.repeat(repeats)
        
        noise = noise[:len(audio)]
        
        # Calculate SNR
        snr_db = random.uniform(*self.noise_snr_range)
        snr_linear = 10 ** (snr_db / 20)
        
        # Calculate signal and noise power
        signal_power = torch.mean(audio ** 2)
        noise_power = torch.mean(noise ** 2)
        
        # Scale noise to achieve desired SNR
        noise_scale = torch.sqrt(signal_power / (noise_power * snr_linear))
        scaled_noise = noise * noise_scale
        
        # Add noise
        augmented = audio + scaled_noise
        
        return augmented
    
    def _add_reverb(self, audio: torch.Tensor) -> torch.Tensor:
        """Add simple reverb effect using delay and decay"""
        # Simple reverb: add delayed, attenuated copies
        reverb = torch.zeros_like(audio)
        
        # Multiple delay taps
        delays = [int(self.sample_rate * d) for d in [0.1, 0.2, 0.3, 0.4, 0.5]]
        decays = [0.7, 0.5, 0.3, 0.2, 0.1]
        
        for delay, decay in zip(delays, decays):
            if delay < len(audio):
                reverb[delay:] += audio[:-delay] * decay
        
        # Mix original and reverb
        augmented = audio + reverb * 0.3
        
        return augmented
    
    def _speed_perturb(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply speed perturbation (time stretching)"""
        speed_factor = random.uniform(*self.speed_range)
        
        # Resample to change speed
        new_length = int(len(audio) / speed_factor)
        
        # Use interpolation for speed change
        time_indices = torch.linspace(0, len(audio) - 1, new_length)
        time_indices = time_indices.long().clamp(0, len(audio) - 1)
        
        augmented = audio[time_indices]
        
        # Pad or truncate to original length
        if len(augmented) < len(audio):
            # Pad with zeros
            padding = len(audio) - len(augmented)
            augmented = torch.cat([augmented, torch.zeros(padding)])
        else:
            # Truncate
            augmented = augmented[:len(audio)]
        
        return augmented
    
    def _pitch_shift(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply pitch shifting using time stretching and resampling"""
        pitch_shift = random.uniform(*self.pitch_range)
        
        # Convert pitch shift to speed factor
        # Positive pitch shift = faster speed
        speed_factor = 2 ** (pitch_shift / 12)
        
        # Apply speed change
        new_length = int(len(audio) / speed_factor)
        time_indices = torch.linspace(0, len(audio) - 1, new_length)
        time_indices = time_indices.long().clamp(0, len(audio) - 1)
        
        augmented = audio[time_indices]
        
        # Resample back to original length
        if len(augmented) != len(audio):
            # Use interpolation to match original length
            time_indices = torch.linspace(0, len(augmented) - 1, len(audio))
            time_indices = time_indices.long().clamp(0, len(augmented) - 1)
            augmented = augmented[time_indices]
        
        return augmented
    
    def _time_stretch(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply time stretching without pitch change"""
        stretch_factor = random.uniform(*self.time_stretch_range)
        
        # Calculate new length
        new_length = int(len(audio) * stretch_factor)
        
        # Use interpolation for time stretching
        time_indices = torch.linspace(0, len(audio) - 1, new_length)
        time_indices = time_indices.long().clamp(0, len(audio) - 1)
        
        augmented = audio[time_indices]
        
        # Ensure output length matches input
        if len(augmented) != len(audio):
            # Pad or truncate
            if len(augmented) < len(audio):
                padding = len(audio) - len(augmented)
                augmented = torch.cat([augmented, torch.zeros(padding)])
            else:
                augmented = augmented[:len(audio)]
        
        return augmented
    
    def augment_batch(
        self, 
        audio_a: torch.Tensor, 
        audio_b: torch.Tensor,
        augment: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Augment a batch of audio pairs"""
        if not augment:
            return audio_a, audio_b
        
        # Apply augmentations to both speakers
        augmented_a = self.augment_audio(audio_a, augment)
        augmented_b = self.augment_audio(audio_b, augment)
        
        return augmented_a, augmented_b

class SpecAugment:
    """Spectrogram augmentation for frequency and time masking"""
    
    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to spectrogram"""
        # Frequency masking
        for _ in range(self.num_freq_masks):
            spec = self._freq_mask(spec)
        
        # Time masking
        for _ in range(self.num_time_masks):
            spec = self._time_mask(spec)
        
        return spec
    
    def _freq_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking"""
        batch_size, num_freq, time_steps = spec.shape
        
        for b in range(batch_size):
            # Random frequency band to mask
            f_start = random.randint(0, num_freq - self.freq_mask_param)
            f_end = f_start + self.freq_mask_param
            
            # Apply mask
            spec[b, f_start:f_end, :] = 0
        
        return spec
    
    def _time_mask(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply time masking"""
        batch_size, num_freq, time_steps = spec.shape
        
        for b in range(batch_size):
            # Random time band to mask
            t_start = random.randint(0, time_steps - self.time_mask_param)
            t_end = t_start + self.time_mask_param
            
            # Apply mask
            spec[b, :, t_start:t_end] = 0
        
        return spec

def create_augmentation_pipeline(
    sample_rate: int = 16000,
    use_audio_aug: bool = True,
    use_spec_aug: bool = True
) -> Tuple[Optional[AudioAugmenter], Optional[SpecAugment]]:
    """Create augmentation pipeline"""
    audio_aug = AudioAugmenter(sample_rate=sample_rate) if use_audio_aug else None
    spec_aug = SpecAugment() if use_spec_aug else None
    
    return audio_aug, spec_aug 