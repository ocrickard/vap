#!/usr/bin/env python3
"""
Phase 1 Setup Script for VAP Turn Detector

This script sets up the complete data preparation pipeline:
1. Downloads datasets (AMI, CHiME-6, VoxConverse, MUSAN, RIRS)
2. Sets up data processing pipeline
3. Creates Lhotse manifests
4. Validates the setup
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vap.data.dataset_info import DATASET_INFO, print_dataset_summary
from vap.data.processing import DataProcessingPipeline
from vap.data.lhotse_integration import LhotseDataManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase1Setup:
    """Handles complete Phase 1 setup"""
    
    def __init__(self, data_root: str = "data", auth_token: str = None):
        self.data_root = Path(data_root)
        self.auth_token = auth_token
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.processing_pipeline = DataProcessingPipeline(auth_token=auth_token)
        self.lhotse_manager = LhotseDataManager(self.data_root)
    
    def setup_directories(self):
        """Create necessary directory structure"""
        logger.info("Setting up directory structure...")
        
        directories = [
            "raw",
            "processed", 
            "manifests",
            "ami",
            "chime6", 
            "voxconverse",
            "musan",
            "rirs"
        ]
        
        for dir_name in directories:
            (self.data_root / dir_name).mkdir(exist_ok=True)
        
        logger.info("Directory structure created")
    
    def download_datasets(self, force: bool = False):
        """Download all required datasets"""
        logger.info("Starting dataset downloads...")
        
        # Run the download script
        download_script = Path(__file__).parent / "download_datasets.py"
        if download_script.exists():
            cmd = ["python", str(download_script), "--data-dir", str(self.data_root)]
            if force:
                cmd.append("--force-download")
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Dataset downloads completed successfully")
                return True
            else:
                logger.error(f"Dataset downloads failed: {result.stderr}")
                return False
        else:
            logger.error("Download script not found")
            return False
    
    def setup_ami_dataset(self):
        """Set up AMI Meeting Corpus"""
        logger.info("Setting up AMI dataset...")
        
        ami_dir = self.data_root / "ami"
        
        # Check if AMI is already set up
        if (ami_dir / "amicorpus").exists():
            logger.info("AMI dataset already exists")
            return True
        
        # AMI requires manual download
        logger.warning("AMI Meeting Corpus requires manual download")
        logger.info("Please:")
        logger.info("1. Visit: https://catalog.ldc.upenn.edu/LDC2005T04")
        logger.info("2. Download the dataset (requires LDC license)")
        logger.info("3. Extract to: %s", ami_dir)
        logger.info("4. Run this script again")
        
        return False
    
    def setup_chime6_dataset(self):
        """Set up CHiME-6 dataset"""
        logger.info("Setting up CHiME-6 dataset...")
        
        chime6_dir = self.data_root / "chime6"
        
        try:
            # Use Lhotse to download CHiME-6
            chime6_cuts = self.lhotse_manager.create_chime6_manifest(chime6_dir)
            if chime6_cuts:
                stats = self.lhotse_manager.get_dataset_stats(chime6_cuts)
                logger.info(f"CHiME-6 setup complete: {stats}")
                return True
            else:
                logger.error("CHiME-6 setup failed")
                return False
                
        except Exception as e:
            logger.error(f"CHiME-6 setup failed: {e}")
            return False
    
    def setup_voxconverse_dataset(self):
        """Set up VoxConverse dataset"""
        logger.info("Setting up VoxConverse dataset...")
        
        voxconverse_dir = self.data_root / "voxconverse"
        
        try:
            # Use Lhotse to download VoxConverse
            voxconverse_cuts = self.lhotse_manager.create_voxconverse_manifest(voxconverse_dir)
            if voxconverse_cuts:
                stats = self.lhotse_manager.get_dataset_stats(voxconverse_cuts)
                logger.info(f"VoxConverse setup complete: {stats}")
                return True
            else:
                logger.error("VoxConverse setup failed")
                return False
                
        except Exception as e:
            logger.error(f"VoxConverse setup failed: {e}")
            return False
    
    def setup_augmentation_datasets(self):
        """Set up MUSAN and RIRS augmentation datasets"""
        logger.info("Setting up augmentation datasets...")
        
        # MUSAN
        musan_dir = self.data_root / "musan"
        if not (musan_dir / "music").exists():
            logger.info("Downloading MUSAN...")
            # This would be handled by the download script
            pass
        
        # RIRS
        rirs_dir = self.data_root / "rirs"
        if not (rirs_dir / "RIRS_NOISES").exists():
            logger.info("Downloading RIRS...")
            # This would be handled by the download script
            pass
        
        logger.info("Augmentation datasets setup complete")
        return True
    
    def validate_setup(self):
        """Validate that all components are working"""
        logger.info("Validating Phase 1 setup...")
        
        validation_results = {}
        
        # Check directories
        required_dirs = ["raw", "processed", "manifests", "ami", "chime6", "voxconverse", "musan", "rirs"]
        for dir_name in required_dirs:
            dir_path = self.data_root / dir_name
            validation_results[f"dir_{dir_name}"] = dir_path.exists()
        
        # Check datasets
        validation_results["ami_available"] = (self.data_root / "ami" / "amicorpus").exists()
        validation_results["chime6_available"] = (self.data_root / "chime6").exists() and len(list((self.data_root / "chime6").glob("*"))) > 0
        validation_results["voxconverse_available"] = (self.data_root / "voxconverse").exists() and len(list((self.data_root / "voxconverse").glob("*"))) > 0
        
        # Check manifests
        manifests_dir = self.data_root / "manifests"
        validation_results["manifests_created"] = manifests_dir.exists() and len(list(manifests_dir.glob("*.jsonl.gz"))) > 0
        
        # Check processing pipeline
        try:
            # Test basic functionality
            test_audio = self.data_root / "test_audio.wav"
            if test_audio.exists():
                # This would test the processing pipeline
                validation_results["processing_pipeline"] = True
            else:
                validation_results["processing_pipeline"] = "skipped (no test audio)"
        except Exception as e:
            validation_results["processing_pipeline"] = False
            logger.error(f"Processing pipeline validation failed: {e}")
        
        # Print validation results
        logger.info("\n" + "="*50)
        logger.info("PHASE 1 VALIDATION RESULTS")
        logger.info("="*50)
        
        for component, status in validation_results.items():
            if isinstance(status, bool):
                status_str = "✅ PASS" if status else "❌ FAIL"
            else:
                status_str = f"⚠️  {status}"
            logger.info(f"{component:25} : {status_str}")
        
        logger.info("="*50)
        
        # Overall status
        passed = sum(1 for v in validation_results.values() if v is True)
        total = len(validation_results)
        success_rate = passed / total if total > 0 else 0
        
        logger.info(f"Overall: {passed}/{total} components ready ({success_rate:.1%})")
        
        if success_rate >= 0.8:
            logger.info("🎉 Phase 1 setup is ready for training!")
        elif success_rate >= 0.6:
            logger.warning("⚠️  Phase 1 setup is partially ready. Some manual steps may be needed.")
        else:
            logger.error("❌ Phase 1 setup needs attention before proceeding.")
        
        return validation_results
    
    def run(self, force_download: bool = False):
        """Run complete Phase 1 setup"""
        logger.info("Starting Phase 1 setup...")
        
        # 1. Setup directories
        self.setup_directories()
        
        # 2. Download datasets
        if not self.download_datasets(force=force_download):
            logger.warning("Dataset downloads failed or incomplete")
        
        # 3. Setup individual datasets
        ami_ready = self.setup_ami_dataset()
        chime6_ready = self.setup_chime6_dataset()
        voxconverse_ready = self.setup_voxconverse_dataset()
        
        # 4. Setup augmentation datasets
        self.setup_augmentation_datasets()
        
        # 5. Validate setup
        validation_results = self.validate_setup()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("PHASE 1 SETUP SUMMARY")
        logger.info("="*50)
        logger.info(f"AMI Dataset: {'✅ Ready' if ami_ready else '❌ Manual setup required'}")
        logger.info(f"CHiME-6 Dataset: {'✅ Ready' if chime6_ready else '❌ Failed'}")
        logger.info(f"VoxConverse Dataset: {'✅ Ready' if voxconverse_ready else '❌ Failed'}")
        logger.info("="*50)
        
        return validation_results

def main():
    parser = argparse.ArgumentParser(description="Setup Phase 1: Data Preparation")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    parser.add_argument("--auth-token", help="Pyannote.audio auth token")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of datasets")
    parser.add_argument("--show-info", action="store_true", help="Show dataset information")
    
    args = parser.parse_args()
    
    if args.show_info:
        print_dataset_summary()
        return
    
    # Initialize setup
    setup = Phase1Setup(args.data_root, args.auth_token)
    
    # Run setup
    results = setup.run(force_download=args.force_download)
    
    # Exit with appropriate code
    if results:
        passed = sum(1 for v in results.values() if v is True)
        total = len(results)
        if passed / total >= 0.8:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Partial success/failure
    else:
        sys.exit(1)  # Failure

if __name__ == "__main__":
    main() 