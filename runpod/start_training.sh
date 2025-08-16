#!/bin/bash
# Smart VAP Training Startup Script for RunPod
# Automatically detects environment and optimizes training

set -e

echo "🚀 VAP Smart Training Startup"
echo "=============================="

# Check if we're on RunPod
if [ -d "/workspace" ]; then
    echo "✅ RunPod environment detected"
    WORKSPACE="/workspace"
else
    echo "💻 Local environment detected"
    WORKSPACE="."
fi

cd $WORKSPACE

# Clone repository if not exists
if [ ! -d "vap" ]; then
    echo "📥 Cloning VAP repository..."
    git clone https://github.com/ocrickard/vap.git
    cd vap
else
    echo "📁 VAP repository already exists"
    cd vap
fi

# Make scripts executable
chmod +x runpod/*.sh
chmod +x runpod/*.py

# Create necessary directories
echo "🏗️  Setting up directory structure..."
mkdir -p data/realtime_dataset data/realtime_dataset/LibriSpeech data/realtime_dataset/LibriSpeech/dev-clean
mkdir -p checkpoints/optimized results logs

# Download dataset if not exists
if [ ! -f "data/realtime_dataset/manifest.json" ]; then
    echo "📥 Setting up LibriSpeech dataset..."
    python3 runpod/download_dataset.py
else
    echo "✅ Dataset already exists"
fi

# Set environment variables
export PYTHONPATH="$WORKSPACE/vap:$PYTHONPATH"

echo ""
echo "🎉 SETUP COMPLETE!"
echo "=================="
echo "🚀 Start training with: python3 runpod/train_on_runpod.py"
echo ""
echo "The system will automatically:"
echo "  • Detect your environment (local vs RunPod)"
echo "  • Apply optimal settings for your hardware"
echo "  • Use consistent training logic across environments"
echo ""
echo "Ready to train! 🚀" 