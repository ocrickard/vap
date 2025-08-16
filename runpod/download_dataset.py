#!/usr/bin/env python3
"""
Dataset Download Script for RunPod

This script downloads and prepares the LibriSpeech dataset directly on RunPod,
avoiding the need to commit large audio files to Git.
"""

import os
import sys
import logging
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset configuration
DATASET_CONFIG = {
    'librispeech_dev_clean': {
        'url': 'https://www.openslr.org/resources/12/dev-clean.tar.gz',
        'filename': 'dev-clean.tar.gz',
        'expected_size_mb': 337,  # ~337 MB
        'md5_hash': '76f87d09068c6eac535be5d3a1b5d8d1',  # Verify integrity
        'extract_path': 'LibriSpeech',
        'target_path': 'data/realtime_dataset/LibriSpeech/dev-clean'
    }
}

def download_file(url, filename, expected_size_mb=None):
    """Download a file with progress bar and size verification"""
    logger.info(f"📥 Downloading {filename} from {url}")
    
    # Create target directory (only if filename has a path)
    dirname = os.path.dirname(filename)
    if dirname:  # Only create directory if there's a path component
        os.makedirs(dirname, exist_ok=True)
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)
    
    # Verify file size
    actual_size_mb = os.path.getsize(filename) / (1024 * 1024)
    logger.info(f"✅ Downloaded: {filename} ({actual_size_mb:.1f} MB)")
    
    if expected_size_mb and abs(actual_size_mb - expected_size_mb) > 10:
        logger.warning(f"⚠️  File size differs from expected: {actual_size_mb:.1f} MB vs {expected_size_mb} MB")
    
    return filename

def verify_md5(filename, expected_hash):
    """Verify file integrity using MD5 hash"""
    logger.info(f"🔍 Verifying file integrity...")
    
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    actual_hash = hash_md5.hexdigest()
    
    if actual_hash == expected_hash:
        logger.info("✅ MD5 hash verification passed")
        return True
    else:
        logger.error(f"❌ MD5 hash verification failed!")
        logger.error(f"   Expected: {expected_hash}")
        logger.error(f"   Actual:   {actual_hash}")
        return False

def extract_tar(filename, extract_path):
    """Extract tar.gz file with progress"""
    logger.info(f"📦 Extracting {filename} to {extract_path}")
    
    # Create extract directory
    os.makedirs(extract_path, exist_ok=True)
    
    # Extract with progress
    with tarfile.open(filename, 'r:gz') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, extract_path)
    
    logger.info(f"✅ Extraction complete: {extract_path}")

def setup_dataset_structure():
    """Create the proper dataset directory structure"""
    logger.info("🏗️  Setting up dataset directory structure...")
    
    # Create directories
    directories = [
        'data/realtime_dataset',
        'data/realtime_dataset/LibriSpeech',
        'data/realtime_dataset/LibriSpeech/dev-clean',
        'checkpoints/optimized',
        'results',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"   ✅ Created: {directory}")

def create_manifest_from_downloaded_data():
    """Create manifest.json from the downloaded LibriSpeech data"""
    logger.info("📝 Creating manifest from downloaded data...")
    
    librispeech_path = Path("LibriSpeech/dev-clean")
    manifest_path = Path("data/realtime_dataset/manifest.json")
    
    if not librispeech_path.exists():
        logger.error("❌ LibriSpeech data not found. Please download first.")
        return False
    
    # Find all audio files
    audio_files = list(librispeech_path.rglob("*.flac"))
    logger.info(f"   Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        logger.error("❌ No audio files found in downloaded data")
        return False
    
    # Create manifest entries
    manifest_entries = []
    for audio_file in audio_files:
        # Extract speaker and chapter info from path
        parts = audio_file.parts
        if len(parts) >= 4:
            speaker_id = parts[-3]  # speaker_id
            chapter_id = parts[-2]  # chapter_id
            filename = parts[-1]    # filename.flac
            
            # Calculate relative path from data root
            relative_path = str(audio_file.relative_to(Path("LibriSpeech")))
            
            entry = {
                "audio_file": relative_path,
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "filename": filename,
                "duration": None,  # Will be calculated during training
                "split": "train" if "dev-clean" in str(audio_file) else "val"
            }
            manifest_entries.append(entry)
    
    # Split into train/val (80/20)
    import random
    random.shuffle(manifest_entries)
    
    split_point = int(len(manifest_entries) * 0.8)
    train_entries = manifest_entries[:split_point]
    val_entries = manifest_entries[split_point:]
    
    # Update split labels
    for entry in train_entries:
        entry["split"] = "train"
    for entry in val_entries:
        entry["split"] = "val"
    
    # Create manifest
    manifest = {
        "dataset_name": "LibriSpeech-dev-clean",
        "total_files": len(manifest_entries),
        "train_files": len(train_entries),
        "val_files": len(val_entries),
        "sample_rate": 16000,
        "audio_format": "flac",
        "entries": manifest_entries
    }
    
    # Save manifest
    import json
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"✅ Manifest created: {manifest_path}")
    logger.info(f"   Train files: {len(train_entries)}")
    logger.info(f"   Val files: {len(val_entries)}")
    
    return True

def cleanup_download_files():
    """Clean up temporary download files"""
    logger.info("🧹 Cleaning up temporary files...")
    
    files_to_remove = [
        'dev-clean.tar.gz',
        'LibriSpeech'  # Remove extracted directory after moving
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
            logger.info(f"   ✅ Removed: {file_path}")

def main():
    """Main function to download and setup dataset"""
    logger.info("🚀 RunPod Dataset Setup - LibriSpeech")
    logger.info("="*50)
    
    try:
        # 1. Setup directory structure
        setup_dataset_structure()
        
        # 2. Download dataset
        config = DATASET_CONFIG['librispeech_dev_clean']
        filename = download_file(
            config['url'], 
            config['filename'],
            config['expected_size_mb']
        )
        
        # 3. Verify integrity
        if not verify_md5(filename, config['md5_hash']):
            logger.error("❌ File integrity check failed. Please retry download.")
            return False
        
        # 4. Extract dataset
        extract_tar(filename, config['extract_path'])
        
        # 5. Move to proper location
        import shutil
        target_path = Path(config['target_path'])
        if target_path.exists():
            shutil.rmtree(target_path)
        
        shutil.move(config['extract_path'], target_path)
        logger.info(f"✅ Dataset moved to: {target_path}")
        
        # 6. Create manifest
        if not create_manifest_from_downloaded_data():
            return False
        
        # 7. Cleanup
        cleanup_download_files()
        
        logger.info("\n🎉 Dataset setup complete!")
        logger.info(f"📊 Dataset location: {config['target_path']}")
        logger.info(f"📝 Manifest: data/realtime_dataset/manifest.json")
        logger.info(f"🚀 Ready for training!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 