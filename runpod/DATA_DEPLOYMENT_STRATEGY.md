# 📊 Data Deployment Strategy for RunPod

This document outlines the best approaches for getting your VAP training data to RunPod efficiently and reliably.

## 🎯 **Recommended Strategy: Hybrid Approach**

We use a **hybrid approach** that combines the best of multiple methods:

1. **Code & Configuration**: Git repository (small, version controlled)
2. **Dataset**: Direct download on RunPod (large, fresh, reliable)
3. **Metadata**: Generated on RunPod (manifest, splits, etc.)

## 🚀 **Deployment Options Comparison**

### **Option 1: Git LFS (Large File Storage)**
```bash
# Setup Git LFS
git lfs install
git lfs track "*.flac"
git lfs track "data/realtime_dataset/**"

# Commit and push
git add .gitattributes
git add data/realtime_dataset/
git commit -m "Add LibriSpeech dataset via Git LFS"
git push origin main
```

**Pros:**
- ✅ Version controlled
- ✅ Easy to update
- ✅ Professional workflow
- ✅ Works with any Git host

**Cons:**
- ❌ GitHub LFS: 1GB/month free, then $5/GB
- ❌ GitLab LFS: 10GB free, then $4/GB
- ❌ Setup complexity
- ❌ Large repository size

**Cost for LibriSpeech (337MB):**
- GitHub: Free (under 1GB/month limit)
- GitLab: Free (under 10GB limit)

### **Option 2: Direct Download on RunPod (RECOMMENDED)**
```bash
# On RunPod, download directly
cd /workspace
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz
mv LibriSpeech data/realtime_dataset/
```

**Pros:**
- ✅ No additional costs
- ✅ Fast download on RunPod's network
- ✅ Fresh data every time
- ✅ No Git complexity
- ✅ Reliable external URLs

**Cons:**
- ❌ Downloads every time you create a new pod
- ❌ No version control
- ❌ Depends on external URLs being available

### **Option 3: Hybrid Approach (BEST CHOICE)**
```bash
# 1. Keep code and config in Git (small)
git add runpod/
git add configs/
git add vap/
git commit -m "Add RunPod configuration and training code"

# 2. Download dataset on RunPod (large)
python runpod/download_dataset.py  # Automated download + setup
```

**Pros:**
- ✅ Best of both worlds
- ✅ Code version controlled
- ✅ Data fresh and reliable
- ✅ Cost effective
- ✅ Automated setup

**Cons:**
- ❌ Slightly more complex initial setup
- ❌ Requires internet connection on RunPod

## 🔧 **Implementation: Our Hybrid Solution**

### **What Goes in Git (Small Files)**
```
vap/
├── runpod/                    # RunPod configuration and scripts
│   ├── runpod_config.yaml    # Training configuration
│   ├── train_on_runpod.py    # Training script
│   ├── start_training.sh     # Startup script
│   ├── download_dataset.py   # Dataset downloader
│   └── DEPLOYMENT_GUIDE.md   # This guide
├── configs/                   # Model configurations
├── vap/                      # Core VAP code
└── scripts/                  # Training scripts
```

**Total Size: ~1-5 MB** (easily fits in any Git repository)

### **What Gets Downloaded on RunPod (Large Files)**
```
data/realtime_dataset/
├── LibriSpeech/
│   └── dev-clean/            # 337 MB of audio files
│       ├── speaker_id/
│       └── chapter_id/
└── manifest.json             # Generated automatically
```

**Total Size: ~337 MB** (downloaded fresh each time)

## 📥 **Step-by-Step Deployment**

### **Step 1: Prepare Your Local Repository**
```bash
# 1. Add RunPod files to Git
git add runpod/
git add configs/
git add vap/
git add scripts/

# 2. Commit and push
git commit -m "Add RunPod GPU training configuration"
git push origin main
```

### **Step 2: Deploy to RunPod**
```bash
# 1. Create RunPod instance
# 2. Connect via SSH/Terminal
# 3. Clone your repository
cd /workspace
git clone https://github.com/yourusername/vap.git
cd vap

# 4. Run automated setup
chmod +x runpod/start_training.sh
./runpod/start_training.sh
```

### **Step 3: Automated Dataset Setup**
The `start_training.sh` script automatically:
1. ✅ Installs dependencies
2. ✅ Downloads LibriSpeech dataset
3. ✅ Creates proper directory structure
4. ✅ Generates manifest.json
5. ✅ Starts GPU training

## 💰 **Cost Analysis**

### **Git LFS Approach**
| Platform | Free Limit | Cost for 337MB | Annual Cost |
|----------|------------|----------------|-------------|
| GitHub | 1GB/month | Free | $0 |
| GitLab | 10GB | Free | $0 |
| Self-hosted | Unlimited | $0 | $0 |

### **Direct Download Approach**
| Cost Component | Amount | Notes |
|----------------|--------|-------|
| Dataset Download | $0 | Free from OpenSLR |
| RunPod Storage | $0 | Included in pod cost |
| Bandwidth | $0 | Included in pod cost |
| **Total** | **$0** | No additional costs |

### **Hybrid Approach (Our Choice)**
| Cost Component | Amount | Notes |
|----------------|--------|-------|
| Git Storage | $0 | Small files only |
| Dataset Download | $0 | Free from OpenSLR |
| RunPod Storage | $0 | Included in pod cost |
| **Total** | **$0** | Most cost-effective |

## 🔄 **Workflow for Multiple Training Runs**

### **First Time Setup**
```bash
# 1. Deploy RunPod instance
# 2. Clone repository
# 3. Run setup script (downloads dataset)
# 4. Train model
# 5. Download results
# 6. Terminate pod
```

### **Subsequent Runs**
```bash
# 1. Deploy new RunPod instance
# 2. Clone repository (same code, fresh dataset)
# 3. Run setup script (downloads fresh dataset)
# 4. Train with different parameters
# 5. Download results
# 6. Terminate pod
```

## 🚨 **Troubleshooting**

### **Dataset Download Issues**
```bash
# Check internet connection
ping google.com

# Verify URL availability
curl -I https://www.openslr.org/resources/12/dev-clean.tar.gz

# Manual download if needed
wget --no-check-certificate https://www.openslr.org/resources/12/dev-clean.tar.gz
```

### **Storage Issues**
```bash
# Check available space
df -h

# Clean up if needed
rm -rf /tmp/*
rm -rf ~/.cache/*
```

### **Git Issues**
```bash
# If repository is too large
git clone --depth 1 https://github.com/yourusername/vap.git

# Or use sparse checkout
git clone --filter=blob:none --sparse https://github.com/yourusername/vap.git
cd vap
git sparse-checkout set runpod/ configs/ vap/ scripts/
```

## 📊 **Performance Comparison**

| Method | Setup Time | Storage Cost | Reliability | Complexity |
|--------|------------|--------------|-------------|------------|
| **Git LFS** | 5-10 min | $0-5/month | High | Medium |
| **Direct Download** | 2-5 min | $0 | High | Low |
| **Hybrid (Our Choice)** | 3-7 min | $0 | High | Low |

## 🎯 **Recommendation Summary**

**Use the Hybrid Approach** because it provides:

1. **✅ Zero Additional Costs**: No LFS fees, no storage costs
2. **✅ High Reliability**: Fresh data every time, no corruption
3. **✅ Easy Maintenance**: Simple Git workflow, automated setup
4. **✅ Scalability**: Works for any dataset size
5. **✅ Professional Quality**: Version controlled code, fresh data

## 🚀 **Next Steps**

1. **Commit RunPod files to Git**:
   ```bash
   git add runpod/
   git commit -m "Add RunPod GPU training configuration"
   git push origin main
   ```

2. **Deploy to RunPod**:
   - Follow `DEPLOYMENT_GUIDE.md`
   - Use `start_training.sh` for automated setup

3. **Start Training**:
   - Dataset downloads automatically
   - GPU training starts immediately
   - Results saved to cloud storage

This approach gives you the best of all worlds: **version controlled code**, **fresh reliable data**, and **zero additional costs**! 🎉 