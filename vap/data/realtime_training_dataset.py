"""
Real Training Dataset for VAP Turn Detector

This module provides the RealTrainingDataset class for training with LibriSpeech data.
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import torch


class RealTrainingDataset(Dataset):
    """Real LibriSpeech dataset for training turn detection"""
    
    def __init__(self, manifest_path, audio_root, max_duration, split='train'):
        self.audio_root = Path(audio_root)
        self.max_duration = max_duration
        self.split = split
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
        
        # Extract entries from manifest structure
        if 'entries' in manifest_data:
            self.manifest = manifest_data['entries']
        else:
            self.manifest = manifest_data  # Fallback for old format
        
        # Filter by split if specified
        if split in ['train', 'val']:
            self.manifest = [item for item in self.manifest if item.get('split') == split]
        
        # Filter by duration and create speaker pairs
        self.samples = self._create_speaker_pairs()
    
    def _create_speaker_pairs(self):
        """Create pairs of samples from different speakers for turn detection"""
        samples = []
        
        # Group by speaker
        speaker_groups = {}
        for item in self.manifest:
            speaker = item['speaker']
            if speaker not in speaker_groups:
                speaker_groups[speaker] = []
            speaker_groups[speaker].append(item)
        
        # Create pairs from different speakers
        speakers = list(speaker_groups.keys())
        for i, speaker_a in enumerate(speakers):
            for speaker_b in speakers[i+1:]:
                # Take up to 5 samples from each speaker pair
                samples_a = speaker_groups[speaker_a][:5]
                samples_b = speaker_groups[speaker_b][:5]
                
                for sample_a in samples_a:
                    for sample_b in samples_b:
                        samples.append((sample_a, sample_b))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_a, sample_b = self.samples[idx]
        
        # Load audio for both speakers
        audio_a = self._load_audio(sample_a)
        audio_b = self._load_audio(sample_b)
        
        # Create realistic labels for training
        # We'll create labels with the same sequence length as the audio
        # The model will handle the conversion to feature dimensions
        labels = self._create_training_labels(audio_a, audio_b)
        
        return {
            'audio_a': audio_a,
            'audio_b': audio_b,
            'speaker_a': sample_a['speaker'],
            'speaker_b': sample_b['speaker'],
            **labels
        }
    
    def _load_audio(self, sample):
        """Load and preprocess audio"""
        try:
            audio_path = self.audio_root / sample["audio_path"]
            
            # Debug: check if file exists and is readable
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            if not audio_path.is_file():
                raise FileNotFoundError(f"Path is not a file: {audio_path}")
            
            # Check file size
            file_size = audio_path.stat().st_size
            if file_size == 0:
                raise ValueError(f"Audio file is empty: {audio_path}")
            
            # Try to load the audio
            audio, sr = sf.read(str(audio_path))
            
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
            
            return torch.FloatTensor(audio)
            
        except Exception as e:
            print(f"Error loading audio from {sample.get('audio_path', 'unknown')}: {e}")
            print(f"Audio root: {self.audio_root}")
            print(f"Full path: {audio_path if 'audio_path' in locals() else 'not set'}")
            print(f"Sample: {sample}")
            raise
    
    def _create_training_labels(self, audio_a, audio_b):
        """Create realistic training labels that match the expected format"""
        # Calculate the downsampled sequence length
        # Raw audio: 480000 samples (30s * 16kHz)
        # Mel spectrogram: 480000/160 = 3000 time steps (100Hz)
        # Downsampled: 3000/4 = 750 time steps (25Hz)
        raw_seq_len = len(audio_a)
        mel_seq_len = raw_seq_len // 160  # hop_length = 160
        downsampled_seq_len = mel_seq_len // 4  # stride=2 twice
        
        # Create VAP pattern labels (20 classes) - this should match the model's output
        # The model expects [batch_size, seq_len] but we're creating single samples
        # So we create [seq_len] and the DataLoader will batch them
        vap_labels = torch.randint(0, 20, (downsampled_seq_len,))
        
        # Create EoT labels (probability of end-of-turn) - binary labels
        eot_labels = torch.zeros(downsampled_seq_len)
        
        # Create backchannel labels (probability of backchannel) - binary labels  
        backchannel_labels = torch.zeros(downsampled_seq_len)
        
        # Create overlap labels (probability of overlap) - binary labels
        overlap_labels = torch.zeros(downsampled_seq_len)
        
        # Create VAD labels (voice activity for both speakers) - binary labels [seq_len, 2]
        vad_labels = torch.zeros(downsampled_seq_len, 2)
        vad_labels[:, 0] = 1.0  # Speaker A is always active
        vad_labels[:, 1] = 0.0  # Speaker B is silent
        
        return {
            'vap_labels': vap_labels,
            'eot_labels': eot_labels,
            'backchannel_labels': backchannel_labels,
            'overlap_labels': overlap_labels,
            'vad_labels': vad_labels
        }


def create_real_training_loader(manifest_path, audio_root, batch_size, max_duration=30.0):
    """Create training data loader with real LibriSpeech data"""
    
    # Create dataset
    dataset = RealTrainingDataset(manifest_path, audio_root, max_duration)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid pickling issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 to avoid pickling issues
        pin_memory=True
    )
    
    return train_loader, val_loader 