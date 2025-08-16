#!/usr/bin/env python3
"""
Dataset Download and Preparation Script for VAP Turn Detector

This script downloads and prepares the following datasets:
- AMI Meeting Corpus (100h, CC-BY 4.0)
- CHiME-6 (dinner-party conversations)
- VoxConverse (in-the-wild conversations)
- MUSAN (noise augmentation)
- RIRS (reverberation augmentation)
"""

import os
import sys
import argparse
import subprocess
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vap.data.dataset_info import DATASET_INFO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self, data_dir: str, force_download: bool = False):
        self.data_dir = Path(data_dir)
        self.force_download = force_download
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_ami(self):
        """Download AMI Meeting Corpus"""
        logger.info("Setting up AMI Meeting Corpus...")
        
        ami_dir = self.data_dir / "ami"
        ami_dir.mkdir(exist_ok=True)
        
        # AMI requires manual download from LDC
        # https://catalog.ldc.upenn.edu/LDC2005T04
        logger.warning("AMI Meeting Corpus requires manual download from LDC")
        logger.info("Please visit: https://catalog.ldc.upenn.edu/LDC2005T04")
        logger.info("Download and extract to: %s", ami_dir)
        
        # Check if already downloaded
        if (ami_dir / "amicorpus").exists():
            logger.info("AMI corpus found at %s", ami_dir / "amicorpus")
            return True
        else:
            logger.error("AMI corpus not found. Please download manually.")
            return False
    
    def download_chime6(self):
        """Download CHiME-6 dataset"""
        logger.info("Setting up CHiME-6 dataset...")
        
        chime_dir = self.data_dir / "chime6"
        chime_dir.mkdir(exist_ok=True)
        
        # CHiME-6 is available via Lhotse
        try:
            import lhotse
            logger.info("Using Lhotse to download CHiME-6...")
            
            # This will download and prepare CHiME-6
            cmd = [
                "python", "-c",
                "import lhotse; lhotse.dataset.chime6.download_chime6('data/chime6')"
            ]
            subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
            
            logger.info("CHiME-6 setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup CHiME-6: {e}")
            return False
    
    def download_voxconverse(self):
        """Download VoxConverse dataset"""
        logger.info("Setting up VoxConverse dataset...")
        
        vox_dir = self.data_dir / "voxconverse"
        vox_dir.mkdir(exist_ok=True)
        
        # VoxConverse is available via Lhotse
        try:
            import lhotse
            logger.info("Using Lhotse to download VoxConverse...")
            
            cmd = [
                "python", "-c",
                "import lhotse; lhotse.dataset.voxconverse.download_voxconverse('data/voxconverse')"
            ]
            subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
            
            logger.info("VoxConverse setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup VoxConverse: {e}")
            return False
    
    def download_musan(self):
        """Download MUSAN noise dataset"""
        logger.info("Downloading MUSAN noise dataset...")
        
        musan_dir = self.data_dir / "musan"
        musan_dir.mkdir(exist_ok=True)
        
        musan_url = "https://www.openslr.org/resources/17/musan.tar.gz"
        musan_file = musan_dir / "musan.tar.gz"
        
        if musan_file.exists() and not self.force_download:
            logger.info("MUSAN already downloaded")
        else:
            logger.info("Downloading MUSAN from %s", musan_url)
            urllib.request.urlretrieve(musan_url, musan_file)
        
        # Extract
        if not (musan_dir / "music").exists():
            logger.info("Extracting MUSAN...")
            with tarfile.open(musan_file, 'r:gz') as tar:
                tar.extractall(musan_dir)
        
        logger.info("MUSAN setup complete")
        return True
    
    def download_rirs(self):
        """Download RIRS reverberation dataset"""
        logger.info("Downloading RIRS reverberation dataset...")
        
        rirs_dir = self.data_dir / "rirs"
        rirs_dir.mkdir(exist_ok=True)
        
        rirs_url = "https://www.openslr.org/resources/28/rirs_noises.zip"
        rirs_file = rirs_dir / "rirs_noises.zip"
        
        if rirs_file.exists() and not self.force_download:
            logger.info("RIRS already downloaded")
        else:
            logger.info("Downloading RIRS from %s", rirs_url)
            urllib.request.urlretrieve(rirs_url, rirs_file)
        
        # Extract
        if not (rirs_dir / "RIRS_NOISES").exists():
            logger.info("Extracting RIRS...")
            with zipfile.ZipFile(rirs_file, 'r') as zip_ref:
                zip_ref.extractall(rirs_dir)
        
        logger.info("RIRS setup complete")
        return True
    
    def create_manifests(self):
        """Create Lhotse manifests for all datasets"""
        logger.info("Creating Lhotse manifests...")
        
        manifests_dir = self.data_dir / "manifests"
        manifests_dir.mkdir(exist_ok=True)
        
        # This will be implemented in the data processing pipeline
        logger.info("Manifest creation will be handled by data processing pipeline")
        return True
    
    def run(self):
        """Download and prepare all datasets"""
        logger.info("Starting dataset download and preparation...")
        
        results = {
            'ami': self.download_ami(),
            'chime6': self.download_chime6(),
            'voxconverse': self.download_voxconverse(),
            'musan': self.download_musan(),
            'rirs': self.download_rirs(),
        }
        
        # Create manifests
        results['manifests'] = self.create_manifests()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("DATASET SETUP SUMMARY")
        logger.info("="*50)
        
        for dataset, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            logger.info(f"{dataset:15} : {status}")
        
        logger.info("="*50)
        
        if all(results.values()):
            logger.info("All datasets prepared successfully!")
        else:
            logger.warning("Some datasets failed to prepare. Check logs above.")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Download and prepare datasets for VAP Turn Detector")
    parser.add_argument("--data-dir", default="data", help="Directory to store datasets")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of existing files")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir, args.force_download)
    results = downloader.run()
    
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main() 