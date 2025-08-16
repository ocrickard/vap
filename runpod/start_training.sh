#!/bin/bash
# RunPod Startup Script for VAP Phase 3 Training
# This script sets up the environment and starts GPU-accelerated training

set -e

echo "🚀 Starting VAP Phase 3 Training on RunPod..."
echo "=============================================="

# Update system packages
echo "📦 Updating system packages..."
apt-get update -y
apt-get install -y git wget curl

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning tensorboard tqdm pyyaml numpy requests

# Clone the repository (if not already present)
if [ ! -d "/workspace/vap" ]; then
    echo "📥 Cloning VAP repository..."
    cd /workspace
    git clone https://github.com/yourusername/vap.git
    cd vap
else
    echo "📁 VAP repository already exists"
    cd /workspace/vap
fi

# Install VAP package
echo "🔧 Installing VAP package..."
pip install -e .

# Create necessary directories
echo "📁 Creating workspace directories..."
mkdir -p /workspace/data/realtime_dataset
mkdir -p /workspace/checkpoints/optimized
mkdir -p /workspace/results
mkdir -p /workspace/logs

# Download and setup dataset
echo "📊 Setting up LibriSpeech dataset..."
if [ ! -f "/workspace/data/realtime_dataset/manifest.json" ]; then
    echo "   Downloading dataset (this may take a few minutes)..."
    python runpod/download_dataset.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Dataset setup complete!"
    else
        echo "❌ Dataset setup failed. Please check the logs."
        exit 1
    fi
else
    echo "✅ Dataset already exists"
fi

# Set environment variables for GPU training
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Display GPU information
echo "🔥 GPU Information:"
nvidia-smi

# Start training
echo "🚀 Starting GPU-accelerated training..."
cd /workspace/vap
python runpod/train_on_runpod.py

echo "✅ Training completed!" 