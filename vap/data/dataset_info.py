"""
Dataset Information and Configuration for VAP Turn Detector

This module contains metadata about the datasets used for training,
including download URLs, processing requirements, and statistics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class DatasetInfo:
    name: str
    description: str
    url: str
    license: str
    size_gb: float
    duration_hours: float
    speakers: int
    language: str
    domain: str
    download_type: str  # "manual", "lhotse", "direct"
    processing_required: List[str]
    notes: Optional[str] = None

# Dataset configurations
DATASET_INFO = {
    "ami": DatasetInfo(
        name="AMI Meeting Corpus",
        description="100 hours of meeting recordings with rich annotations",
        url="https://catalog.ldc.upenn.edu/LDC2005T04",
        license="LDC License (commercial use requires license)",
        size_gb=15.0,
        duration_hours=100.0,
        speakers=4,
        language="English",
        domain="meeting",
        download_type="manual",
        processing_required=["diarization", "vad", "vap_labels"],
        notes="Primary training dataset. Requires LDC license for commercial use."
    ),
    
    "chime6": DatasetInfo(
        name="CHiME-6 Challenge",
        description="Dinner party conversations in challenging acoustic conditions",
        url="https://chimechallenge.github.io/chime6/",
        license="CC-BY 4.0",
        size_gb=8.0,
        duration_hours=20.0,
        speakers=4,
        language="English",
        domain="dinner_party",
        download_type="lhotse",
        processing_required=["diarization", "vad", "vap_labels"],
        notes="Good for robustness testing due to challenging acoustic conditions."
    ),
    
    "voxconverse": DatasetInfo(
        name="VoxConverse",
        description="In-the-wild conversations from YouTube videos",
        url="https://www.robots.ox.ac.uk/~vgg/data/voxconverse/",
        license="CC-BY 4.0",
        size_gb=5.0,
        duration_hours=50.0,
        speakers=2,
        language="English",
        domain="conversation",
        download_type="lhotse",
        processing_required=["diarization", "vad", "vap_labels"],
        notes="Good for domain generalization and real-world scenarios."
    ),
    
    "musan": DatasetInfo(
        name="MUSAN",
        description="Music, Speech, and Noise dataset for augmentation",
        url="https://www.openslr.org/17/",
        license="CC-BY 4.0",
        size_gb=2.0,
        duration_hours=100.0,
        speakers=0,  # Not speaker-specific
        language="multilingual",
        domain="noise",
        download_type="direct",
        processing_required=["categorization"],
        notes="Used for noise augmentation during training."
    ),
    
    "rirs": DatasetInfo(
        name="RIRS Noises",
        description="Room impulse responses for reverberation simulation",
        url="https://www.openslr.org/28/",
        license="CC-BY 4.0",
        size_gb=0.5,
        duration_hours=0,  # Not time-based
        speakers=0,
        language="N/A",
        domain="reverberation",
        download_type="direct",
        processing_required=["categorization"],
        notes="Used for reverberation augmentation during training."
    )
}

def get_dataset_paths(data_root: Path) -> Dict[str, Path]:
    """Get paths for all datasets relative to data root"""
    return {
        name: data_root / name
        for name in DATASET_INFO.keys()
    }

def get_total_duration() -> float:
    """Get total duration of all speech datasets in hours"""
    return sum(
        info.duration_hours 
        for info in DATASET_INFO.values() 
        if info.duration_hours > 0
    )

def get_total_size() -> float:
    """Get total size of all datasets in GB"""
    return sum(info.size_gb for info in DATASET_INFO.values())

def print_dataset_summary():
    """Print a summary of all datasets"""
    print("="*80)
    print("DATASET SUMMARY FOR VAP TURN DETECTOR")
    print("="*80)
    
    total_duration = get_total_duration()
    total_size = get_total_size()
    
    for name, info in DATASET_INFO.items():
        print(f"\n{info.name}")
        print(f"  Description: {info.description}")
        print(f"  Duration: {info.duration_hours:.1f} hours")
        print(f"  Size: {info.size_gb:.1f} GB")
        print(f"  Speakers: {info.speakers}")
        print(f"  Language: {info.language}")
        print(f"  Domain: {info.domain}")
        print(f"  License: {info.license}")
        print(f"  Download: {info.download_type}")
        print(f"  Processing: {', '.join(info.processing_required)}")
        if info.notes:
            print(f"  Notes: {info.notes}")
    
    print(f"\n{'='*80}")
    print(f"TOTAL: {total_duration:.1f} hours, {total_size:.1f} GB")
    print("="*80)

if __name__ == "__main__":
    print_dataset_summary() 