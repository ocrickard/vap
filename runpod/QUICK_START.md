# 🚀 Quick Start: Smart VAP Training

**Train VAP models seamlessly on both local and RunPod environments!**

## ⚡ **5-Minute Setup**

### **1. Local Development** 💻
```bash
# Clone and setup
git clone https://github.com/ocrickard/vap.git
cd vap
python -m venv vap_env
source vap_env/bin/activate  # On Windows: vap_env\Scripts\activate
pip install -r requirements.txt

# Start training (automatically optimized for CPU)
python runpod/train_on_runpod.py
```

### **2. RunPod Deployment** 🚀
```bash
# On RunPod, clone and run
cd /workspace
git clone https://github.com/ocrickard/vap.git
cd vap
chmod +x runpod/start_training.sh
./runpod/start_training.sh

# Start training (automatically optimized for GPU)
python runpod/train_on_runpod.py
```

**That's it!** 🎉 The same script works everywhere.

## 🔥 **What Happens Next**

1. **✅ Environment Detection**: Automatically detects local vs RunPod
2. **📥 Dataset Setup**: Downloads LibriSpeech (337MB, ~2 minutes)
3. **⚙️ Smart Configuration**: Applies optimal settings for your hardware
4. **🚀 Training Starts**: CPU-optimized locally, GPU-accelerated on RunPod

## 💰 **Cost Breakdown (RunPod)**

- **RTX 4090**: $0.60/hour
- **Total Training**: 2-8 hours
- **Total Cost**: $1.20 - $4.80
- **Speed Improvement**: 10-25x faster

## 📊 **Expected Timeline**

| Environment | Setup | Training | Total |
|-------------|-------|----------|-------|
| **Local CPU** | 5 min | 25-50 hours | 1-2 days |
| **RunPod GPU** | 5 min | 2-8 hours | 2-8 hours |

## ✨ **Smart Features**

### **Automatic Environment Detection**
- **Local**: CPU optimization, smaller batch sizes, fewer workers
- **RunPod**: GPU acceleration, larger batch sizes, more workers

### **Consistent Training Logic**
- **Same Script**: `runpod/train_on_runpod.py` works everywhere
- **Same Model**: Consistent architecture and training approach
- **Same Results**: Reproducible across environments

### **Intelligent Configuration**
- **Hardware Detection**: Automatically detects GPU/CPU
- **Path Handling**: Relative paths work on both local and cloud
- **Progress Tracking**: Environment-appropriate progress indicators

## 🚨 **Troubleshooting**

**Local Issues?**
- Check virtual environment: `source vap_env/bin/activate`
- Verify dependencies: `pip install -r requirements.txt`
- Check dataset: `python runpod/download_dataset.py`

**RunPod Issues?**
- Check GPU: `nvidia-smi`
- Verify paths: `ls -la /workspace`
- Check logs: `tail -f /workspace/logs/training.log`

## 📚 **Documentation**

- **Smart Configuration**: `runpod/runpod_config.yaml`
- **Dataset Setup**: `runpod/download_dataset.py`
- **Startup Script**: `runpod/start_training.sh`

## 🎯 **Success Checklist**

- [ ] Environment detected correctly (local vs RunPod)
- [ ] Dataset downloaded and manifest created
- [ ] Training started with appropriate optimization
- [ ] Progress bars showing real-time updates
- [ ] Checkpoints saving to correct location

---

## 🆕 **Why Smart Training?**

**Before**: Different scripts for local vs RunPod, inconsistent approaches  
**Now**: Single script, automatic optimization, consistent results

### **Benefits**
- ✅ **No More Confusion**: Same command everywhere
- ✅ **Automatic Optimization**: Best settings for your hardware
- ✅ **Consistent Results**: Same training logic across environments
- ✅ **Easy Deployment**: One script to rule them all

---

**Ready to train?** 🚀

Your VAP Phase 3 training will be **10-25x faster** on RunPod compared to local CPU training, and the same script works seamlessly in both environments! 