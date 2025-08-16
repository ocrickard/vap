"""
Simple Data Loader for VAP Turn Detector

This is a basic data loader for testing the pipeline with the synthetic dataset.
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleAudioDataset(Dataset):
    """Simple audio dataset for testing"""
    
    def __init__(self, manifest_path: str, audio_root: str, max_duration: float = 6.0):
        self.audio_root = Path(audio_root)
        self.max_duration = max_duration
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Use all samples
        self.valid_samples = self.manifest
        
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # Load audio - handle both filename and audio_path keys
        if "filename" in sample:
            audio_path = self.audio_root / sample["filename"]
        elif "audio_path" in sample:
            # For LibriSpeech, the audio_path is relative to dev-clean
            if "librispeech" in str(self.audio_root).lower():
                audio_path = self.audio_root / "dev-clean" / sample["audio_path"]
            else:
                audio_path = self.audio_root / sample["audio_path"]
        else:
            raise KeyError(f"Neither 'filename' nor 'audio_path' found in sample: {sample}")
        
        audio, sr = sf.read(audio_path)
        
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
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio)
        
        # Create dummy labels for testing
        labels = {
            'vap_patterns': torch.zeros(256),  # 256 VAP pattern classes
            'eot_probability': torch.tensor(0.1),  # Dummy EoT probability
            'backchannel_probability': torch.tensor(0.05),  # Dummy backchannel probability
            'overlap_probability': torch.tensor(0.02),  # Dummy overlap probability
            'vad_scores': torch.ones(1)  # Dummy VAD score
        }
        
        # Handle both speaker and speaker_id keys
        speaker_id = sample.get('speaker_id', sample.get('speaker', 'unknown'))
        
        return {
            'audio': audio_tensor,
            'sample_rate': sr,
            'duration': len(audio) / sr,
            'speaker_id': speaker_id,
            'labels': labels
        }

def create_dataloader(manifest_path: str, audio_root: str, batch_size: int = 4):
    """Create a DataLoader for the simple dataset"""
    dataset = SimpleAudioDataset(manifest_path, audio_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

if __name__ == "__main__":
    # Test the data loader
    manifest_path = "data/synthetic_dataset/manifest.json"
    audio_root = "data/synthetic_dataset"
    
    if Path(manifest_path).exists() and Path(audio_root).exists():
        dataloader = create_dataloader(manifest_path, audio_root, batch_size=2)
        
        print("Testing data loader...")
        for i, batch in enumerate(dataloader):
            print(f"Batch {i+1}:")
            print(f"  Audio shape: {batch['audio'].shape}")
            print(f"  Sample rate: {batch['sample_rate']}")
            print(f"  Duration: {batch['duration']}")
            print(f"  Speaker: {batch['speaker_id']}")
            print(f"  Labels keys: {list(batch['labels'].keys())}")
            
            if i >= 1:  # Just test first 2 batches
                break
        
        print("✅ Data loader test complete!")
    else:
        print("❌ Dataset not found. Run setup first.")
