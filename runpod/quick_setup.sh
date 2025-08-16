#!/bin/bash
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
