"""
Audio ring buffer for streaming audio processing
"""

import numpy as np
from typing import Optional, Tuple


class AudioRingBuffer:
    """
    Ring buffer for audio samples with efficient circular storage.
    
    Supports continuous streaming audio with configurable buffer size.
    """
    
    def __init__(self, buffer_size: int, dtype: np.dtype = np.float32):
        """
        Initialize the ring buffer.
        
        Args:
            buffer_size: Maximum number of samples to store
            dtype: Data type for audio samples
        """
        self.buffer_size = buffer_size
        self.dtype = dtype
        
        # Initialize buffer
        self.buffer = np.zeros(buffer_size, dtype=dtype)
        
        # Buffer state
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0
        self.is_full = False
        
    def add_samples(self, samples: np.ndarray) -> None:
        """
        Add new audio samples to the buffer.
        
        Args:
            samples: Audio samples to add
        """
        if len(samples) == 0:
            return
            
        # Handle case where samples exceed buffer size
        if len(samples) >= self.buffer_size:
            # Keep only the most recent samples
            samples = samples[-self.buffer_size:]
            self.write_pos = 0
            self.read_pos = 0
            self.count = self.buffer_size
            self.is_full = True
            self.buffer[:] = samples
            return
        
        # Add samples to buffer
        samples_to_write = len(samples)
        
        # Check if we need to wrap around
        if self.write_pos + samples_to_write <= self.buffer_size:
            # No wrap-around needed
            self.buffer[self.write_pos:self.write_pos + samples_to_write] = samples
        else:
            # Wrap-around needed
            first_part = self.buffer_size - self.write_pos
            second_part = samples_to_write - first_part
            
            self.buffer[self.write_pos:] = samples[:first_part]
            self.buffer[:second_part] = samples[first_part:]
        
        # Update write position
        self.write_pos = (self.write_pos + samples_to_write) % self.buffer_size
        
        # Update count and full status
        if not self.is_full:
            self.count = min(self.count + samples_to_write, self.buffer_size)
            if self.count == self.buffer_size:
                self.is_full = True
        else:
            # Buffer is full, update read position
            self.read_pos = (self.read_pos + samples_to_write) % self.buffer_size
    
    def get_samples(self, num_samples: Optional[int] = None) -> np.ndarray:
        """
        Get samples from the buffer.
        
        Args:
            num_samples: Number of samples to retrieve. If None, returns all available.
            
        Returns:
            Audio samples from the buffer
        """
        if num_samples is None:
            num_samples = self.count
        
        if num_samples <= 0 or self.count == 0:
            return np.array([], dtype=self.dtype)
        
        # Limit to available samples
        num_samples = min(num_samples, self.count)
        
        # Get samples from buffer
        if self.read_pos + num_samples <= self.buffer_size:
            # No wrap-around needed
            samples = self.buffer[self.read_pos:self.read_pos + num_samples].copy()
        else:
            # Wrap-around needed
            first_part = self.buffer_size - self.read_pos
            second_part = num_samples - first_part
            
            samples = np.concatenate([
                self.buffer[self.read_pos:],
                self.buffer[:second_part]
            ])
        
        return samples
    
    def get_latest_samples(self, num_samples: int) -> np.ndarray:
        """
        Get the most recent samples from the buffer.
        
        Args:
            num_samples: Number of samples to retrieve
            
        Returns:
            Most recent audio samples
        """
        if num_samples <= 0 or self.count == 0:
            return np.array([], dtype=self.dtype)
        
        # Limit to available samples
        num_samples = min(num_samples, self.count)
        
        # Calculate start position for latest samples
        if self.is_full:
            start_pos = (self.write_pos - num_samples) % self.buffer_size
        else:
            start_pos = max(0, self.write_pos - num_samples)
        
        # Get samples
        if start_pos + num_samples <= self.buffer_size:
            # No wrap-around needed
            samples = self.buffer[start_pos:start_pos + num_samples].copy()
        else:
            # Wrap-around needed
            first_part = self.buffer_size - start_pos
            second_part = num_samples - first_part
            
            samples = np.concatenate([
                self.buffer[start_pos:],
                self.buffer[:second_part]
            ])
        
        return samples
    
    def get_buffer_state(self) -> Tuple[int, int, int, bool]:
        """
        Get the current buffer state.
        
        Returns:
            Tuple of (write_pos, read_pos, count, is_full)
        """
        return self.write_pos, self.read_pos, self.count, self.is_full
    
    def get_fill_percentage(self) -> float:
        """Get the percentage of buffer that is filled."""
        return (self.count / self.buffer_size) * 100.0
    
    def is_empty(self) -> bool:
        """Check if the buffer is empty."""
        return self.count == 0
    
    def clear(self) -> None:
        """Clear the buffer and reset state."""
        self.buffer.fill(0)
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0
        self.is_full = False
    
    def resize(self, new_size: int) -> None:
        """
        Resize the buffer.
        
        Args:
            new_size: New buffer size
        """
        if new_size <= 0:
            raise ValueError("Buffer size must be positive")
        
        # Get current samples
        current_samples = self.get_samples()
        
        # Create new buffer
        self.buffer_size = new_size
        self.buffer = np.zeros(new_size, dtype=self.dtype)
        
        # Reset state
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0
        self.is_full = False
        
        # Add back current samples if they fit
        if len(current_samples) > 0:
            self.add_samples(current_samples)
    
    def get_buffer_info(self) -> dict:
        """Get comprehensive buffer information."""
        return {
            'buffer_size': self.buffer_size,
            'write_pos': self.write_pos,
            'read_pos': self.read_pos,
            'count': self.count,
            'is_full': self.is_full,
            'fill_percentage': self.get_fill_percentage(),
            'dtype': str(self.dtype)
        } 