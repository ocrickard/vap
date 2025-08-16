# 🚀 Quick Start: Deploy VAP Training to RunPod

**Get GPU-accelerated training running in 5 minutes!**

## ⚡ **5-Minute Deployment**

### **1. Commit RunPod Files (1 min)**
```bash
git add runpod/
git add configs/
git add vap/
git commit -m "Add RunPod GPU training configuration"
git push origin main
```

### **2. Deploy RunPod Instance (2 min)**
1. Go to [runpod.io](https://runpod.io)
2. Click "Deploy" → "GPU Pod"
3. Select: **RTX 4090** (24GB VRAM, $0.60/hour)
4. Click "Deploy"

### **3. Start Training (2 min)**
```bash
# On RunPod, clone and run
cd /workspace
git clone https://github.com/yourusername/vap.git
cd vap
chmod +x runpod/start_training.sh
./runpod/start_training.sh
```

**That's it!** 🎉 Training starts automatically.

## 🔥 **What Happens Next**

1. **✅ Dependencies Install** (PyTorch, PyTorch Lightning, etc.)
2. **📥 Dataset Downloads** (LibriSpeech, 337MB, ~2 minutes)
3. **🏗️ Environment Setup** (directories, manifest generation)
4. **🚀 GPU Training Starts** (10-25x faster than CPU)

## 💰 **Cost Breakdown**

- **RTX 4090**: $0.60/hour
- **Total Training**: 2-8 hours
- **Total Cost**: $1.20 - $4.80
- **Speed Improvement**: 10-25x faster

## 📊 **Expected Timeline**

| Phase | Duration | What Happens |
|-------|----------|--------------|
| **Setup** | 5-10 min | Dependencies + dataset download |
| **Training** | 2-8 hours | GPU-accelerated model training |
| **Results** | Immediate | Checkpoints + logs saved |

## 🚨 **Troubleshooting**

**Pod won't start?**
- Check RunPod status page
- Try different GPU type
- Verify payment method

**Dataset download fails?**
- Check internet connection
- Run `python runpod/download_dataset.py` manually
- Check available disk space

**Training crashes?**
- Check GPU memory: `nvidia-smi`
- Reduce batch size in `runpod_config.yaml`
- Check logs in `/workspace/logs/`

## 📚 **Full Documentation**

- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **Data Strategy**: `DATA_DEPLOYMENT_STRATEGY.md`
- **Configuration**: `runpod_config.yaml`

## 🎯 **Success Checklist**

- [ ] RunPod instance running with RTX 4090
- [ ] Repository cloned successfully
- [ ] Dataset downloaded (337MB)
- [ ] Training started with GPU acceleration
- [ ] Progress bars showing real-time updates
- [ ] Checkpoints saving to `/workspace/checkpoints/`

---

**Ready to deploy?** 🚀

Your VAP Phase 3 training will be **10-25x faster** on RunPod compared to local CPU training! 