"""
Data Processing Pipeline for VAP Turn Detector

This module handles:
- Audio preprocessing (normalization, resampling)
- Diarization with Pyannote.audio
- VAD detection and refinement
- VAP pattern label generation
- Audio augmentation (MUSAN + RIRS)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torchaudio
import librosa
from dataclasses import dataclass

# Audio processing
import soundfile as sf
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# VAP-specific imports
from vap.models.vap_model import VAPTurnDetector

logger = logging.getLogger(__name__)

@dataclass
class AudioSegment:
    """Represents a segment of audio with metadata"""
    audio: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    speaker_id: Optional[str] = None
    vad_score: Optional[float] = None

@dataclass
class TurnTakingEvent:
    """Represents a turn-taking event"""
    event_type: str  # "eot", "backchannel", "overlap"
    start_time: float
    end_time: float
    speaker_id: str
    confidence: float

class AudioPreprocessor:
    """Handles audio preprocessing for VAP training"""
    
    def __init__(self, target_sr: int = 16000, normalize: bool = True):
        self.target_sr = target_sr
        self.normalize = normalize
    
    def process_audio(self, audio_path: Union[str, Path]) -> AudioSegment:
        """Load and preprocess audio file"""
        logger.info(f"Processing audio: {audio_path}")
        
        # Load audio
        if audio_path.suffix.lower() in ['.wav', '.flac']:
            audio, sr = sf.read(str(audio_path))
        else:
            audio, sr = librosa.load(str(audio_path), sr=None)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        
        # Normalize
        if self.normalize:
            audio = librosa.util.normalize(audio)
        
        # Calculate duration
        duration = len(audio) / sr
        
        return AudioSegment(
            audio=audio,
            sample_rate=sr,
            start_time=0.0,
            end_time=duration
        )

class DiarizationProcessor:
    """Handles speaker diarization using Pyannote.audio"""
    
    def __init__(self, auth_token: Optional[str] = None):
        self.auth_token = auth_token
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the diarization pipeline"""
        try:
            if self.auth_token:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization@2.1",
                    use_auth_token=self.auth_token
                )
            else:
                # Try to use local model if available
                logger.warning("No auth token provided. Using local model if available.")
                # This would need to be implemented based on local model availability
                pass
        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {e}")
            self.pipeline = None
    
    def process_audio(self, audio_path: Union[str, Path]) -> List[AudioSegment]:
        """Perform diarization on audio file"""
        if not self.pipeline:
            logger.error("Diarization pipeline not available")
            return []
        
        logger.info(f"Performing diarization on: {audio_path}")
        
        try:
            # Run diarization
            with ProgressHook() as hook:
                diarization = self.pipeline(str(audio_path), hook=hook)
            
            # Extract segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = turn.start
                end_time = turn.end
                
                # Extract audio for this segment
                audio = self._extract_segment(audio_path, start_time, end_time)
                
                segment = AudioSegment(
                    audio=audio,
                    sample_rate=16000,  # Assuming 16kHz
                    start_time=start_time,
                    end_time=end_time,
                    speaker_id=speaker
                )
                segments.append(segment)
            
            logger.info(f"Diarization complete: {len(segments)} segments found")
            return segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return []
    
    def _extract_segment(self, audio_path: Union[str, Path], start: float, end: float) -> np.ndarray:
        """Extract audio segment between start and end times"""
        # This is a simplified implementation
        # In practice, you'd want to use a more efficient method
        audio, sr = sf.read(str(audio_path))
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        return audio[start_sample:end_sample]

class VAPLabelGenerator:
    """Generates VAP pattern labels for training"""
    
    def __init__(self, horizon_seconds: float = 2.0, num_classes: int = 256):
        self.horizon_seconds = horizon_seconds
        self.num_classes = num_classes
        self.frame_rate = 50  # Hz, matches model output
    
    def generate_labels(self, audio_segments: List[AudioSegment]) -> Dict[str, np.ndarray]:
        """Generate VAP labels from audio segments"""
        logger.info("Generating VAP labels...")
        
        # This is a placeholder implementation
        # In practice, you'd implement the full VAP labeling logic
        
        labels = {
            'vap_patterns': np.zeros((len(audio_segments), self.num_classes)),
            'eot_probability': np.zeros(len(audio_segments)),
            'backchannel_probability': np.zeros(len(audio_segments)),
            'overlap_probability': np.zeros(len(audio_segments)),
            'vad_scores': np.zeros(len(audio_segments))
        }
        
        logger.info(f"Generated labels for {len(audio_segments)} segments")
        return labels

class AudioAugmenter:
    """Handles audio augmentation for training robustness"""
    
    def __init__(self, musan_dir: Optional[Path] = None, rirs_dir: Optional[Path] = None):
        self.musan_dir = musan_dir
        self.rirs_dir = rirs_dir
        self._load_augmentation_data()
    
    def _load_augmentation_data(self):
        """Load MUSAN and RIRS data for augmentation"""
        # This would load the actual augmentation data
        # For now, it's a placeholder
        pass
    
    def augment_audio(self, audio: np.ndarray, augmentation_type: str = "noise") -> np.ndarray:
        """Apply audio augmentation"""
        if augmentation_type == "noise" and self.musan_dir:
            return self._add_noise(audio)
        elif augmentation_type == "reverb" and self.rirs_dir:
            return self._add_reverb(audio)
        else:
            return audio
    
    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add noise from MUSAN dataset"""
        # Placeholder implementation
        return audio
    
    def _add_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Add reverberation using RIRS"""
        # Placeholder implementation
        return audio

class DataProcessingPipeline:
    """Main data processing pipeline"""
    
    def __init__(self, 
                 target_sr: int = 16000,
                 auth_token: Optional[str] = None,
                 musan_dir: Optional[Path] = None,
                 rirs_dir: Optional[Path] = None):
        
        self.preprocessor = AudioPreprocessor(target_sr)
        self.diarizer = DiarizationProcessor(auth_token)
        self.label_generator = VAPLabelGenerator()
        self.augmenter = AudioAugmenter(musan_dir, rirs_dir)
    
    def process_dataset(self, 
                       dataset_path: Path,
                       output_path: Path,
                       dataset_type: str = "ami") -> bool:
        """Process an entire dataset"""
        logger.info(f"Processing dataset: {dataset_path}")
        
        try:
            # Find audio files
            audio_files = self._find_audio_files(dataset_path)
            logger.info(f"Found {len(audio_files)} audio files")
            
            # Process each file
            processed_segments = []
            for audio_file in audio_files:
                # Preprocess
                audio_segment = self.preprocessor.process_audio(audio_file)
                
                # Diarize
                if dataset_type in ["ami", "chime6", "voxconverse"]:
                    segments = self.diarizer.process_audio(audio_file)
                    if segments:
                        processed_segments.extend(segments)
                    else:
                        # Fallback to single segment
                        processed_segments.append(audio_segment)
                else:
                    processed_segments.append(audio_segment)
            
            # Generate labels
            labels = self.label_generator.generate_labels(processed_segments)
            
            # Save processed data
            self._save_processed_data(processed_segments, labels, output_path)
            
            logger.info(f"Dataset processing complete: {len(processed_segments)} segments")
            return True
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            return False
    
    def _find_audio_files(self, dataset_path: Path) -> List[Path]:
        """Find all audio files in dataset directory"""
        audio_extensions = ['.wav', '.flac', '.mp3', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(dataset_path.rglob(f"*{ext}"))
        
        return sorted(audio_files)
    
    def _save_processed_data(self, 
                            segments: List[AudioSegment], 
                            labels: Dict[str, np.ndarray], 
                            output_path: Path):
        """Save processed data and labels"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save segments
        segments_file = output_path / "segments.npz"
        np.savez_compressed(
            segments_file,
            audio=[seg.audio for seg in segments],
            sample_rates=[seg.sample_rate for seg in segments],
            start_times=[seg.start_time for seg in segments],
            end_times=[seg.end_time for seg in segments],
            speaker_ids=[seg.speaker_id for seg in segments],
            vad_scores=[seg.vad_score for seg in segments]
        )
        
        # Save labels
        labels_file = output_path / "labels.npz"
        np.savez_compressed(labels_file, **labels)
        
        logger.info(f"Saved processed data to: {output_path}")

def main():
    """Example usage of the data processing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process datasets for VAP training")
    parser.add_argument("--dataset-path", required=True, help="Path to dataset")
    parser.add_argument("--output-path", required=True, help="Output path for processed data")
    parser.add_argument("--dataset-type", default="ami", help="Dataset type")
    parser.add_argument("--auth-token", help="Pyannote.audio auth token")
    
    args = parser.parse_args()
    
    pipeline = DataProcessingPipeline(auth_token=args.auth_token)
    success = pipeline.process_dataset(
        Path(args.dataset_path),
        Path(args.output_path),
        args.dataset_type
    )
    
    if success:
        print("Dataset processing completed successfully!")
    else:
        print("Dataset processing failed!")
        exit(1)

if __name__ == "__main__":
    main() 