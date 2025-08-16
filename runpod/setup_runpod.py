#!/usr/bin/env python3
"""
RunPod Setup Script for VAP Phase 3 Training

This script helps prepare your local environment and provides instructions
for deploying to RunPod.
"""

import os
import sys
import yaml
from pathlib import Path

def check_local_environment():
    """Check if local environment is ready for RunPod deployment"""
    print("🔍 Checking local environment...")
    
    # Check if we're in the VAP project directory
    if not Path("vap").exists():
        print("❌ Not in VAP project directory. Please run this from the project root.")
        return False
    
    # Check if RunPod directory exists
    if not Path("runpod").exists():
        print("❌ RunPod directory not found. Please ensure runpod/ is created.")
        return False
    
    # Check required files
    required_files = [
        "runpod/runpod_config.yaml",
        "runpod/train_on_runpod.py",
        "runpod/start_training.sh",
        "runpod/DEPLOYMENT_GUIDE.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ Local environment is ready!")
    return True

def create_runpod_instructions():
    """Create personalized RunPod deployment instructions"""
    print("\n🚀 RUNPOD DEPLOYMENT INSTRUCTIONS")
    print("="*50)
    
    print("\n📋 Step 1: Prepare Your Local Repository")
    print("   • Add RunPod files to Git:")
    print("     git add runpod/")
    print("     git add configs/")
    print("     git add vap/")
    print("     git commit -m \"Add RunPod GPU training configuration\"")
    print("     git push origin main")
    
    print("\n📋 Step 2: Create RunPod Account")
    print("   • Go to https://runpod.io")
    print("   • Sign up for an account")
    print("   • Add payment method")
    
    print("\n🔥 Step 3: Deploy GPU Pod")
    print("   • Login to RunPod dashboard")
    print("   • Click 'Deploy' → 'GPU Pod'")
    print("   • Select configuration:")
    print("     - GPU: RTX 4090 (24GB VRAM) - $0.60/hour")
    print("     - CPU: 8 cores")
    print("     - RAM: 32GB")
    print("     - Storage: 50GB")
    print("     - Image: runpod/pytorch:2.1.1-py3.10-cuda12.1.0")
    
    print("\n🔗 Step 4: Connect to Pod")
    print("   • Wait for pod to start (2-3 minutes)")
    print("   • Click 'Connect' → 'SSH' or 'Terminal'")
    print("   • Navigate to /workspace directory")
    
    print("\n📥 Step 5: Setup Training Environment")
    print("   • Clone your VAP repository:")
    print("     git clone https://github.com/yourusername/vap.git")
    print("   • Navigate to vap directory:")
    print("     cd vap")
    print("   • Make startup script executable:")
    print("     chmod +x runpod/start_training.sh")
    print("   • Run startup script (automatically downloads dataset):")
    print("     ./runpod/start_training.sh")
    
    print("\n📊 Step 6: Monitor Training")
    print("   • Watch real-time progress in terminal")
    print("   • Monitor GPU usage: nvidia-smi")
    print("   • Access TensorBoard logs")
    print("   • Download results when complete")
    
    print("\n💡 Data Strategy: Hybrid Approach")
    print("   • Code & Config: Git repository (version controlled)")
    print("   • Dataset: Downloaded fresh on RunPod (reliable)")
    print("   • No additional costs, fresh data every time!")

def estimate_training_costs():
    """Estimate training costs on RunPod"""
    print("\n💰 TRAINING COST ESTIMATION")
    print("="*40)
    
    # Cost per hour for different GPU types
    gpu_costs = {
        "RTX 4090": 0.60,
        "RTX 3090": 0.50,
        "RTX 3080": 0.40,
        "A100": 2.00
    }
    
    # Training time estimates (hours)
    training_times = {
        "RTX 4090": (2, 8),
        "RTX 3090": (3, 10),
        "RTX 3080": (4, 12),
        "A100": (2, 6)
    }
    
    print("GPU Type          | Cost/Hour | Est. Time | Total Cost")
    print("-" * 55)
    
    for gpu_type, cost_per_hour in gpu_costs.items():
        min_time, max_time = training_times[gpu_type]
        min_cost = cost_per_hour * min_time
        max_cost = cost_per_hour * max_time
        
        print(f"{gpu_type:<16} | ${cost_per_hour:>8.2f} | {min_time:>2}-{max_time:<2}h     | ${min_cost:>5.2f}-${max_cost:<5.2f}")
    
    print(f"\n💡 Recommendation: RTX 4090 for best price/performance ratio")
    print(f"   Expected cost: $1.20 - $4.80 for complete training")

def check_dataset_preparation():
    """Check if dataset is ready for RunPod"""
    print("\n📊 DATASET PREPARATION CHECK")
    print("="*40)
    
    manifest_path = "data/realtime_dataset/manifest.json"
    audio_root = "data/realtime_dataset/LibriSpeech/dev-clean"
    
    if Path(manifest_path).exists():
        print(f"✅ Manifest file found: {manifest_path}")
    else:
        print(f"❌ Manifest file missing: {manifest_path}")
        print("   Run: python scripts/setup_phase1.py")
    
    if Path(audio_root).exists():
        print(f"✅ Audio data found: {audio_root}")
        
        # Count audio files
        audio_files = list(Path(audio_root).rglob("*.flac"))
        print(f"   Audio files: {len(audio_files)}")
        
        if len(audio_files) > 0:
            print("✅ Dataset is ready for RunPod deployment!")
        else:
            print("⚠️  No audio files found in dataset")
    else:
        print(f"❌ Audio data missing: {audio_root}")
        print("   Run: python scripts/setup_phase1.py")

def create_runpod_script():
    """Create a simple script to help with RunPod deployment"""
    script_content = """#!/bin/bash
# Quick RunPod Setup Script
# Run this on your RunPod instance

echo "🚀 Setting up VAP training on RunPod..."

# Update packages
apt-get update -y
apt-get install -y git

# Install Python dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning tensorboard tqdm pyyaml numpy

# Clone repository (replace with your actual repo URL)
cd /workspace
git clone https://github.com/yourusername/vap.git
cd vap

# Install VAP package
pip install -e .

# Create directories
mkdir -p /workspace/data/realtime_dataset
mkdir -p /workspace/checkpoints/optimized
mkdir -p /workspace/results
mkdir -p /workspace/logs

echo "✅ Setup complete! Now upload your dataset and run:"
echo "   python runpod/train_on_runpod.py"
"""
    
    script_path = "runpod/quick_setup.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"✅ Created quick setup script: {script_path}")

def main():
    """Main setup function"""
    print("🚀 VAP Phase 3 - RunPod Setup")
    print("="*40)
    
    # Check local environment
    if not check_local_environment():
        print("\n❌ Please fix the issues above before proceeding.")
        return
    
    # Check dataset preparation
    check_dataset_preparation()
    
    # Create quick setup script
    create_runpod_script()
    
    # Show deployment instructions
    create_runpod_instructions()
    
    # Show cost estimation
    estimate_training_costs()
    
    print("\n🎉 RunPod setup is complete!")
    print("\n📚 Next Steps:")
    print("1. Read runpod/DEPLOYMENT_GUIDE.md for detailed instructions")
    print("2. Sign up for RunPod account")
    print("3. Deploy GPU pod with RTX 4090")
    print("4. Upload your dataset")
    print("5. Run the training script")
    
    print(f"\n💡 Pro tip: Training will be 10-25x faster on RunPod!")
    print(f"   Local CPU: 25-50 hours → RunPod GPU: 2-8 hours")

if __name__ == "__main__":
    main() 