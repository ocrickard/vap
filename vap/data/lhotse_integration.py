"""
Lhotse Integration for VAP Turn Detector

This module provides Lhotse-based data loading and manifest generation
for efficient training with the VAP model.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

# Lhotse imports
try:
    import lhotse
    from lhotse import CutSet, RecordingSet, SupervisionSet
    from lhotse.audio import Recording
    from lhotse.supervision import SupervisionSegment
    from lhotse.dataset import DynamicBucketingSampler
    from lhotse.dataset.input_strategies import AudioSamples
    from lhotse.utils import Pathlike
    LHOTSE_AVAILABLE = True
except ImportError:
    LHOTSE_AVAILABLE = False
    logger.warning("Lhotse not available. Install with: pip install lhotse")

logger = logging.getLogger(__name__)

class LhotseDataManager:
    """Manages Lhotse data loading and manifest generation"""
    
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.manifests_dir = self.data_root / "manifests"
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
    
    def create_ami_manifest(self, ami_dir: Path) -> Optional[CutSet]:
        """Create Lhotse manifest for AMI dataset"""
        if not LHOTSE_AVAILABLE:
            logger.error("Lhotse not available")
            return None
        
        logger.info("Creating AMI manifest...")
        
        try:
            # This would integrate with Lhotse's AMI dataset
            # For now, we'll create a placeholder
            logger.info("AMI manifest creation not yet implemented")
            return None
            
        except Exception as e:
            logger.error(f"Failed to create AMI manifest: {e}")
            return None
    
    def create_chime6_manifest(self, chime6_dir: Path) -> Optional[CutSet]:
        """Create Lhotse manifest for CHiME-6 dataset"""
        if not LHOTSE_AVAILABLE:
            logger.error("Lhotse not available")
            return None
        
        logger.info("Creating CHiME-6 manifest...")
        
        try:
            # CHiME-6 is natively supported by Lhotse
            from lhotse.dataset.chime6 import download_chime6
            
            # Download and prepare CHiME-6
            chime6_cuts = download_chime6(chime6_dir)
            
            # Save manifest
            manifest_path = self.manifests_dir / "chime6_cuts.jsonl.gz"
            chime6_cuts.to_file(manifest_path)
            
            logger.info(f"CHiME-6 manifest saved to: {manifest_path}")
            return chime6_cuts
            
        except Exception as e:
            logger.error(f"Failed to create CHiME-6 manifest: {e}")
            return None
    
    def create_voxconverse_manifest(self, voxconverse_dir: Path) -> Optional[CutSet]:
        """Create Lhotse manifest for VoxConverse dataset"""
        if not LHOTSE_AVAILABLE:
            logger.error("Lhotse not available")
            return None
        
        logger.info("Creating VoxConverse manifest...")
        
        try:
            # VoxConverse is natively supported by Lhotse
            from lhotse.dataset.voxconverse import download_voxconverse
            
            # Download and prepare VoxConverse
            voxconverse_cuts = download_voxconverse(voxconverse_dir)
            
            # Save manifest
            manifest_path = self.manifests_dir / "voxconverse_cuts.jsonl.gz"
            voxconverse_cuts.to_file(manifest_path)
            
            logger.info(f"VoxConverse manifest saved to: {manifest_path}")
            return voxconverse_cuts
            
        except Exception as e:
            logger.error(f"Failed to create VoxConverse manifest: {e}")
            return None
    
    def create_custom_manifest(self, 
                              audio_files: List[Path],
                              labels: Optional[Dict] = None) -> Optional[CutSet]:
        """Create custom Lhotse manifest from processed audio files"""
        if not LHOTSE_AVAILABLE:
            logger.error("Lhotse not available")
            return None
        
        logger.info("Creating custom manifest...")
        
        try:
            recordings = []
            supervisions = []
            
            for i, audio_file in enumerate(audio_files):
                # Create recording
                recording = Recording.from_file(audio_file)
                recordings.append(recording)
                
                # Create supervision (if labels available)
                if labels and 'speaker_ids' in labels:
                    speaker_id = labels['speaker_ids'][i] if i < len(labels['speaker_ids']) else f"speaker_{i}"
                    
                    supervision = SupervisionSegment(
                        id=f"segment_{i:06d}",
                        recording_id=recording.id,
                        start=0.0,
                        duration=recording.duration,
                        speaker=speaker_id
                    )
                    supervisions.append(supervision)
            
            # Create sets
            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions) if supervisions else None
            
            # Create cuts
            if supervision_set:
                cuts = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
            else:
                cuts = CutSet.from_recordings(recordings=recording_set)
            
            # Save manifest
            manifest_path = self.manifests_dir / "custom_cuts.jsonl.gz"
            cuts.to_file(manifest_path)
            
            logger.info(f"Custom manifest saved to: {manifest_path}")
            return cuts
            
        except Exception as e:
            logger.error(f"Failed to create custom manifest: {e}")
            return None
    
    def create_training_sampler(self, 
                               cuts: CutSet,
                               max_duration: float = 30.0,
                               shuffle: bool = True) -> Optional[DynamicBucketingSampler]:
        """Create training sampler for efficient batching"""
        if not LHOTSE_AVAILABLE:
            logger.error("Lhotse not available")
            return None
        
        try:
            sampler = DynamicBucketingSampler(
                cuts,
                max_duration=max_duration,
                shuffle=shuffle,
                drop_last=True
            )
            
            logger.info(f"Created training sampler with max_duration={max_duration}s")
            return sampler
            
        except Exception as e:
            logger.error(f"Failed to create training sampler: {e}")
            return None
    
    def load_manifest(self, manifest_path: Path) -> Optional[CutSet]:
        """Load existing Lhotse manifest"""
        if not LHOTSE_AVAILABLE:
            logger.error("Lhotse not available")
            return None
        
        try:
            cuts = CutSet.from_file(manifest_path)
            logger.info(f"Loaded manifest: {len(cuts)} cuts")
            return cuts
            
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return None
    
    def get_dataset_stats(self, cuts: CutSet) -> Dict:
        """Get statistics about the dataset"""
        if not LHOTSE_AVAILABLE:
            return {}
        
        try:
            total_duration = sum(cut.duration for cut in cuts)
            num_speakers = len(set(cut.supervision.speaker for cut in cuts if cut.supervision))
            
            stats = {
                'total_cuts': len(cuts),
                'total_duration_hours': total_duration / 3600,
                'total_duration_seconds': total_duration,
                'num_speakers': num_speakers,
                'avg_duration': total_duration / len(cuts) if cuts else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get dataset stats: {e}")
            return {}

def main():
    """Example usage of Lhotse data manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Lhotse manifests for VAP training")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    parser.add_argument("--ami-dir", help="AMI dataset directory")
    parser.add_argument("--chime6-dir", help="CHiME-6 dataset directory")
    parser.add_argument("--voxconverse-dir", help="VoxConverse dataset directory")
    
    args = parser.parse_args()
    
    if not LHOTSE_AVAILABLE:
        print("Lhotse not available. Install with: pip install lhotse")
        exit(1)
    
    data_manager = LhotseDataManager(args.data_root)
    
    # Create manifests for available datasets
    if args.ami_dir:
        ami_cuts = data_manager.create_ami_manifest(Path(args.ami_dir))
        if ami_cuts:
            stats = data_manager.get_dataset_stats(ami_cuts)
            print(f"AMI stats: {stats}")
    
    if args.chime6_dir:
        chime6_cuts = data_manager.create_chime6_manifest(Path(args.chime6_dir))
        if chime6_cuts:
            stats = data_manager.get_dataset_stats(chime6_cuts)
            print(f"CHiME-6 stats: {stats}")
    
    if args.voxconverse_dir:
        voxconverse_cuts = data_manager.create_voxconverse_manifest(Path(args.voxconverse_dir))
        if voxconverse_cuts:
            stats = data_manager.get_dataset_stats(voxconverse_cuts)
            print(f"VoxConverse stats: {stats}")
    
    print("Manifest creation complete!")

if __name__ == "__main__":
    main() 