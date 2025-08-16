# VAP Turn Detector - Project Structure

This document provides an overview of the project structure and implementation phases.

## Current Implementation (Phase 0)

### Core Architecture
- **`vap/models/`** - Model architectures
  - `vap_model.py` - Main VAP turn detector
  - `encoders.py` - Audio encoders and transformers
  - `heads.py` - Prediction heads for VAP patterns and turn-taking
- **`vap/streaming/`** - Real-time inference
  - `streaming_detector.py` - Streaming turn detector
  - `ring_buffer.py` - Audio ring buffer for streaming
- **`vap/tasks/`** - Training components
  - `turn_detection_task.py` - PyTorch Lightning training task

### Configuration & Scripts
- **`configs/`** - Training configurations
  - `vap_light.yaml` - Lightweight model configuration
- **`scripts/`** - Utility scripts
  - `train.py` - Main training script
  - `inference.py` - Inference script
  - `setup_test.py` - Project verification script

### Testing
- **`tests/`** - Unit tests
  - `test_model.py` - Basic model functionality tests

## Implementation Phases

### Phase 0: Project Setup & Scaffolding ✅
- [x] Core model architecture
- [x] Streaming inference pipeline
- [x] Training framework
- [x] Basic configuration
- [x] Project structure

### Phase 1: Dataset Preparation & Labeling (Next)
- [ ] AMI Meeting Corpus integration
- [ ] CHiME-6 dataset support
- [ ] VoxConverse integration
- [ ] Lhotse manifest generation
- [ ] Audio augmentation pipeline
- [ ] Label generation for VAP patterns

### Phase 2: Baseline Training
- [ ] Data loading and preprocessing
- [ ] Training pipeline validation
- [ ] Baseline model training
- [ ] Initial evaluation metrics

### Phase 3: Robustness & Optimization
- [ ] Noise and reverberation augmentation
- [ ] Multi-corpus training
- [ ] Model optimization
- [ ] Performance benchmarking

### Phase 4: Production Deployment
- [ ] Model quantization
- [ ] Streaming optimization
- [ ] Documentation
- [ ] Example applications

## Key Features

### Model Architecture
- **Audio-domain processing**: Log-Mel spectrograms with neural codec support
- **Cross-attention transformer**: Efficient speaker interaction modeling
- **Multi-task learning**: VAP patterns, EoT, backchannel, overlap, VAD
- **Small footprint**: 3-8M parameters, CPU/GPU friendly

### Streaming Inference
- **Real-time processing**: 20ms hop size
- **Ring buffer management**: Efficient audio buffering
- **State tracking**: Turn-taking state management
- **Performance monitoring**: Inference time tracking

### Training Framework
- **PyTorch Lightning**: Clean training loops
- **Multi-dataset support**: Flexible data loading
- **Comprehensive metrics**: EoT F1, hold/shift accuracy, overlap detection
- **Configurable loss weighting**: Balanced multi-task learning

## Usage Examples

### Basic Model Usage
```python
from vap.models import VAPTurnDetector

model = VAPTurnDetector()
outputs = model(audio_a, audio_b)
```

### Streaming Inference
```python
from vap.streaming import StreamingTurnDetector

detector = StreamingTurnDetector(model)
detector.add_audio(chunk_a, chunk_b)
predictions = detector.step()
```

### Training
```bash
python scripts/train.py --config configs/vap_light.yaml
```

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run setup test**: `python scripts/setup_test.py`
3. **Verify model**: `python tests/test_model.py`
4. **Begin Phase 1**: Dataset preparation and labeling

## Dependencies

- **Core**: PyTorch, PyTorch Lightning, TorchAudio
- **Audio**: Librosa, SoundFile, WebRTC VAD
- **Data**: Lhotse, Pyannote.audio (optional)
- **Utilities**: NumPy, SciPy, Matplotlib, WandB

## Architecture Notes

The model follows the VAP (Voice Activity Projection) paradigm:
- Predicts future speech activity patterns over 2-second horizon
- Uses cross-attention between speaker representations
- Outputs discrete VAP patterns plus auxiliary turn-taking predictions
- Designed for low-latency streaming inference

This approach enables early turn-taking decisions before silence stabilizes, beating traditional VAD-based heuristics on latency. 