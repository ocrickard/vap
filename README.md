# VAP Turn Detector

A Voice Activity Projection (VAP) based turn detection model for real-time conversation analysis, inspired by LiveKit's turn detector architecture.

## 🚀 Quick Start

### Option 1: Run Complete Pipeline (Phases 1-2)
```bash
# Clone and setup
git clone <repository-url>
cd vap
python -m venv vap_env
source vap_env/bin/activate  # On Windows: vap_env\Scripts\activate
pip install -r requirements.txt

# Run complete pipeline (setup, train baseline, evaluate)
python scripts/run_pipeline.py
```

### Option 2: Run Phase 3 Optimization (Local CPU)
```bash
# After completing Phases 1-2, run optimization locally
python scripts/run_phase3.py
```

### Option 3: Run Phase 3 Optimization (RunPod GPU) 🚀🔥
```bash
# For 10-25x faster training, use RunPod GPU acceleration
python runpod/setup_runpod.py  # Setup instructions
# Follow the deployment guide: runpod/DEPLOYMENT_GUIDE.md
```

**✨ NEW: Enhanced Progress Indicators**
- **Real-time Training Dashboard**: Live progress bars, epoch tracking, and performance metrics
- **Smart Status Updates**: Batch-by-batch progress with loss and accuracy monitoring
- **Training Analytics**: Epoch summaries, trend analysis, and time estimates
- **Performance Tracking**: Best validation accuracy monitoring and improvement alerts

**🚀 NEW: RunPod GPU Acceleration**
- **10-25x Faster Training**: GPU acceleration vs local CPU
- **Cost Effective**: $1.20-$4.80 for complete training
- **Professional Infrastructure**: RTX 4090 with 24GB VRAM
- **Easy Deployment**: One-click GPU pod deployment

### Option 3: Step-by-Step
```bash
# 1. Setup data
python scripts/setup_phase1.py

# 2. Train baseline model
python scripts/train_baseline.py

# 3. Evaluate baseline
python scripts/evaluate_baseline.py

# 4. Train optimized model (with progress indicators)
python scripts/train_optimized.py

# 5. Evaluate optimized model
python scripts/evaluate_optimized.py
```

## 📊 Current Status

**Phase 3 In Progress** 🚧 - Model Optimization Underway!

- **Model**: 128-dim, 2-layer, 4-head transformer (3x capacity increase)
- **Training**: Enhanced with real VAP labels and audio augmentation
- **Performance**: Targeting >25% accuracy improvement over baseline
- **Architecture**: Multi-task learning with 5 prediction heads
- **Data**: Real audio processing + comprehensive augmentation pipeline
- **Status**: Phase 3 infrastructure complete, ready for training

## 🎯 Enhanced Training Experience

### ✨ Real-Time Progress Monitoring
The Phase 3 optimization includes a comprehensive training dashboard that provides:

- **Live Progress Bars**: Visual progress indicators for each epoch and batch
- **Performance Metrics**: Real-time loss and accuracy tracking
- **Epoch Summaries**: Detailed statistics after each training epoch
- **Trend Analysis**: Performance improvement tracking and alerts
- **Time Estimates**: Accurate predictions for training completion

### 📈 Training Dashboard Features
```
🚀 EPOCH 1/50
===========================================================
Epoch 1: 45%|████████████████▌         | 439/975 [02:15<02:45, 3.24batch/s]

📊 EPOCH 1 SUMMARY
--------------------------------------------------
⏱️  Duration: 135.2s
📈 Train Loss: 2.8476
📉 Val Loss: 2.2341
🎯 Val Accuracy: 0.1876
🎉 NEW BEST VALIDATION ACCURACY: 0.1876

📈 TRAINING PROGRESS SUMMARY
----------------------------------------
Completed Epochs: 1/50
Progress: 2.0%
Average Epoch Time: 135.2s
Total Training Time: 2.3 minutes
Estimated Time Remaining: 110.2 minutes
Best Validation Accuracy: 0.1876
Recent Trend: ↗️ Improving
```

### 🔍 Smart Monitoring
- **Batch Progress**: Updates every 50 training batches and 20 validation batches
- **Performance Alerts**: Immediate notification of new best accuracies
- **Trend Detection**: Automatic analysis of performance improvements
- **Resource Tracking**: Memory usage and training efficiency monitoring

## 🏗️ Architecture

### Core Components
- **Audio Encoder**: Log-Mel spectrogram + CNN downsampling
- **Cross-Attention Transformer**: Speaker interaction modeling
- **Multi-task Heads**: VAP patterns, EoT, backchannel, overlap, VAD

### Model Variants
- **Baseline**: 64-dim, 1-layer, 2-heads (Phase 2 - Complete)
- **Optimized**: 128-dim, 2-layers, 4-heads (Phase 3 - In Progress)
- **Standard**: 256-dim, 4-layers, 8-heads (Phase 4 - Planned)

## 📁 Project Structure

```
vap/
├── vap/                    # Core package
│   ├── models/            # Model definitions
│   ├── tasks/             # Training tasks
│   ├── data/              # Data processing + augmentation
│   └── streaming/         # Real-time inference
├── scripts/               # Executable scripts
│   ├── run_pipeline.py    # Complete pipeline runner
│   ├── train_baseline.py  # Baseline training (Phase 2)
│   ├── train_optimized.py # Optimized training (Phase 3)
│   ├── evaluate_baseline.py # Baseline evaluation
│   ├── evaluate_optimized.py # Optimized evaluation
│   ├── run_phase3.py      # Phase 3 optimization pipeline
│   └── inference.py       # Inference demo
├── configs/               # Configuration files
│   ├── vap_light.yaml     # Baseline configuration
│   └── vap_optimized.yaml # Optimized configuration (Phase 3)
├── data/                  # Dataset storage
├── checkpoints/           # Trained models
├── results/               # Evaluation results
└── tests/                 # Unit tests
```

## 🔧 Scripts Overview

| Script | Purpose | Status |
|--------|---------|---------|
| `run_pipeline.py` | Complete pipeline runner (Phases 1-2) | ✅ Ready |
| `run_phase3.py` | Phase 3 optimization pipeline | ✅ Ready |
| `train_baseline.py` | Baseline model training | ✅ Ready |
| `train_optimized.py` | Optimized model training | ✅ Ready |
| `evaluate_baseline.py` | Baseline performance evaluation | ✅ Ready |
| `evaluate_optimized.py` | Optimized performance evaluation | ✅ Ready |
| `inference.py` | Inference demo | ✅ Ready |
| `setup_phase1.py` | Data preparation | ✅ Ready |
| `download_realtime_dataset.py` | LibriSpeech download | ✅ Ready |

## 📈 Performance Metrics

### Phase 2 Baseline Results (Complete)
- **VAP Pattern Accuracy**: 4.73%
- **EoT Accuracy**: 0.00%
- **Backchannel Accuracy**: 0.00%
- **Overlap Accuracy**: 0.00%
- **VAD Accuracy**: 50.00%
- **Overall Accuracy**: 10.95%

### Phase 3 Optimized Targets (In Progress)
- **Target Improvement**: >25% overall accuracy
- **Model Capacity**: 3x parameter increase (378K → ~1.5M)
- **Architecture**: Enhanced transformer (128-dim, 2-layers, 4-heads)
- **Training**: Real VAP labels + comprehensive augmentation

### Next Steps
1. **Phase 3**: Complete optimized training and evaluation
2. **Phase 4**: Performance benchmarking and hyperparameter tuning
3. **Phase 5**: Production deployment and optimization

## 🎯 Use Cases

- **Real-time Turn Detection**: Live conversation analysis
- **Meeting Transcription**: Speaker diarization enhancement
- **Voice Assistants**: Turn-taking behavior modeling
- **Research**: Conversation dynamics analysis

## 🛠️ Development

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- PyTorch Lightning
- Audio processing libraries

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_model.py
```

### Adding New Features
1. Implement in `vap/` package
2. Add tests in `tests/`
3. Update scripts as needed
4. Document in README

## 📚 References

- **VAP Architecture**: Voice Activity Projection for turn detection
- **LiveKit Turn Detector**: Production turn detection system
- **LibriSpeech**: Open-source speech dataset
- **PyTorch Lightning**: Training framework

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request

## 📄 License

[Add your license here]

---

**Last Updated**: August 16, 2025  
**Status**: Phase 3 In Progress - Model Optimization Underway  
**Next Milestone**: Phase 4 - Performance Benchmarking 