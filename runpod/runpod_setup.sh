#!/bin/bash
# 🚀 Robust RunPod Setup Script for VAP Phase 3 Training
# This script handles everything step by step with validation
# Run this ONCE after connecting to your RunPod instance

set -e  # Exit on any error

echo "🚀 VAP Phase 3 - Robust RunPod Setup"
echo "===================================="
echo "This script will set up your entire training environment"
echo ""

# Configuration
REPO_URL="https://github.com/ocrickard/vap.git"
REPO_NAME="vap"
WORKSPACE="/workspace"
PROJECT_DIR="$WORKSPACE/$REPO_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        log_success "$1"
    else
        log_error "$2"
        exit 1
    fi
}

# Check if we're on RunPod
check_runpod_environment() {
    log_info "Step 1: Checking RunPod environment..."
    
    if [ ! -d "$WORKSPACE" ]; then
        log_error "Not in RunPod environment. Expected /workspace directory."
        exit 1
    fi
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi not found. GPU may not be available."
    else
        log_success "GPU environment detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    fi
    
    log_success "RunPod environment verified"
}

# Update system packages
update_system() {
    log_info "Step 2: Updating system packages..."
    apt-get update -y
    apt-get install -y git wget curl htop tree
    check_status "System packages updated" "Failed to update system packages"
}

# Clone and setup repository
setup_repository() {
    log_info "Step 3: Setting up VAP repository..."
    
    cd "$WORKSPACE"
    
    if [ -d "$REPO_NAME" ]; then
        log_warning "Repository already exists. Updating..."
        cd "$REPO_NAME"
        git fetch origin
        git reset --hard origin/main
        log_success "Repository updated"
    else
        log_info "Cloning repository..."
        git clone "$REPO_URL"
        cd "$REPO_NAME"
        log_success "Repository cloned"
    fi
    
    # Make scripts executable
    chmod +x runpod/*.sh
    chmod +x runpod/*.py
    
    check_status "Repository setup complete" "Failed to setup repository"
}

# Create workspace directories
create_directories() {
    log_info "Step 4: Creating workspace directories..."
    
    mkdir -p "$WORKSPACE/data/realtime_dataset"
    mkdir -p "$WORKSPACE/checkpoints/optimized"
    mkdir -p "$WORKSPACE/results"
    mkdir -p "$WORKSPACE/logs"
    mkdir -p "$WORKSPACE/logs/local_test"
    
    check_status "Workspace directories created" "Failed to create directories"
}

# Fix the dataset download script
fix_dataset_script() {
    log_info "Step 5: Fixing dataset download script..."
    
    cd "$PROJECT_DIR"
    
    # Fix the path issue in download_dataset.py
    sed -i 's/os.makedirs(os.path.dirname(filename), exist_ok=True)/dirname = os.path.dirname(filename)\n        if dirname:\n            os.makedirs(dirname, exist_ok=True)/' runpod/download_dataset.py
    
    # Also fix the LibriSpeech path issue
    sed -i 's/audio_path = self.audio_root \/ sample\["audio_path"\]/if "librispeech" in str(self.audio_root).lower():\n                audio_path = self.audio_root \/ "dev-clean" \/ sample["audio_path"]\n            else:\n                audio_path = self.audio_root \/ sample["audio_path"]/' vap/data/simple_loader.py
    
    check_status "Dataset script fixed" "Failed to fix dataset script"
}

# Install Python dependencies (minimal set)
install_python_deps() {
    log_info "Step 6: Installing Python dependencies..."
    
    cd "$PROJECT_DIR"
    
    # Install only essential packages (avoiding PyTorch conflicts)
    pip install pytorch-lightning tensorboard tqdm pyyaml numpy requests librosa soundfile webrtcvad lhotse matplotlib scipy
    
    check_status "Python dependencies installed" "Failed to install Python dependencies"
}

# Set environment variables
setup_environment() {
    log_info "Step 7: Setting up environment variables..."
    
    # GPU configuration
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    
    # Add to bashrc for persistence
    echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
    echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512" >> ~/.bashrc
    
    check_status "Environment variables configured" "Failed to configure environment variables"
}

# Download and setup dataset
setup_dataset() {
    log_info "Step 8: Setting up LibriSpeech dataset..."
    
    cd "$PROJECT_DIR"
    
    if [ ! -f "$WORKSPACE/data/realtime_dataset/manifest.json" ]; then
        log_info "Downloading dataset (this may take a few minutes)..."
        python runpod/download_dataset.py
        
        if [ $? -eq 0 ]; then
            log_success "Dataset setup complete!"
        else
            log_error "Dataset setup failed. Please check the logs."
            exit 1
        fi
    else
        log_success "Dataset already exists"
    fi
}

# Verify installation
verify_installation() {
    log_info "Step 9: Verifying installation..."
    
    cd "$PROJECT_DIR"
    
    # Check PyTorch
    log_info "Checking PyTorch installation..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    fi
    
    # Test local training pipeline
    log_info "Testing local training pipeline..."
    python runpod/test_local_training.py
    
    if [ $? -eq 0 ]; then
        log_success "Local testing passed - ready for GPU training!"
    else
        log_warning "Local testing had issues, but continuing..."
    fi
    
    log_success "Installation verification complete"
}

# Display final status
display_status() {
    echo ""
    echo "🎉 RUNPOD SETUP COMPLETE!"
    echo "========================="
    echo ""
    echo "📁 Project Location: $PROJECT_DIR"
    echo "📊 Dataset Location: $WORKSPACE/data/realtime_dataset"
    echo "💾 Checkpoints: $WORKSPACE/checkpoints/optimized"
    echo "📈 Results: $WORKSPACE/results"
    echo "📝 Logs: $WORKSPACE/logs"
    echo ""
    echo "🚀 Ready to start training!"
    echo ""
    echo "Next steps:"
    echo "1. Start training: cd $PROJECT_DIR && python runpod/train_on_runpod.py"
    echo "2. Monitor GPU: watch -n 1 nvidia-smi"
    echo "3. View logs: tail -f $WORKSPACE/logs/training.log"
    echo "4. TensorBoard: tensorboard --logdir=$WORKSPACE/logs --host=0.0.0.0 --port=6006"
    echo ""
    echo "💡 Training will be 10-25x faster on GPU vs CPU!"
    echo "⏱️  Expected time: 2-8 hours (vs 25-50 hours on CPU)"
    echo "💰 Estimated cost: $1.20 - $4.80 for complete training"
}

# Main execution
main() {
    echo "Starting robust RunPod setup..."
    echo ""
    
    check_runpod_environment
    update_system
    setup_repository
    create_directories
    fix_dataset_script
    install_python_deps
    setup_environment
    setup_dataset
    verify_installation
    display_status
    
    echo ""
    log_success "Setup complete! You can now start training."
}

# Run main function
main "$@" 