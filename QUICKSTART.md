# FunGen Rewrite - Quick Start Guide

Get up and running with FunGen in 5 minutes!

---

## Prerequisites

- Python 3.11+
- FFmpeg installed
- For GPU: CUDA 12.1+ with RTX 3090
- For CPU: Raspberry Pi 4/5 or any modern CPU

---

## Installation (3 steps)

### Step 1: Install Dependencies

**For GPU (RTX 3090):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**For CPU (Raspberry Pi):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 2: Download YOLO Model

```bash
mkdir -p models
# Download YOLO11n (nano - fastest)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt -O models/yolo11n.pt
```

### Step 3: Install Package

```bash
pip install -e .
```

---

## Usage (Choose One)

### Option 1: GUI Mode (Interactive)

```bash
python main.py
```

### Option 2: CLI Mode (Single Video)

```bash
python main.py --cli video.mp4 -o output.funscript
```

### Option 3: Batch Processing

```bash
python main.py --cli --batch videos/ -o output/
```

---

## Example: Process Your First Video

```bash
# 1. Place your video in the current directory
# 2. Run with improved tracker
python main.py --cli my_video.mp4 --tracker improved

# Output will be saved to: output/my_video.funscript
```

---

## Configuration

### Hardware Profiles (Auto-detected)

- **Raspberry Pi**: CPU mode, 5+ FPS
- **RTX 3090**: GPU mode, 100+ FPS, TensorRT FP16

### Override Profile

```bash
# Force RTX 3090 mode
python main.py --profile prod_rtx3090

# Force Pi mode
python main.py --profile dev_pi
```

### Adjust Performance

```bash
# Increase batch size (more VRAM, faster)
python main.py --cli video.mp4 --batch-size 16

# Increase workers (parallel processing)
python main.py --cli --batch videos/ --workers 8
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```bash
python main.py --cli video.mp4 --batch-size 4
```

### Issue: TensorRT Not Found

**Solution:**
```bash
python main.py --cli video.mp4 --no-tensorrt
```

### Issue: Slow on Raspberry Pi

**Solution:**
```bash
python main.py --cli video.mp4 --profile dev_pi --device cpu
```

---

## What's Next?

- Read full documentation: [README.md](README.md)
- Understand architecture: [docs/architecture.md](docs/architecture.md)
- Run tests: `pytest tests/`
- Join development: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Performance Expectations

| Hardware | Resolution | FPS | VRAM |
|----------|-----------|-----|------|
| RTX 3090 | 1080p | 100-120 FPS | 18GB |
| RTX 3090 | 4K | 65-75 FPS | 20GB |
| RTX 3090 | 8K | 60-65 FPS | 22GB |
| Pi 5 | 1080p | 5-8 FPS | 0GB |

---

## Support

- GitHub Issues: https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/issues
- Documentation: [README.md](README.md)
- Integration Report: [INTEGRATION_REPORT.md](INTEGRATION_REPORT.md)

---

**Ready to process videos at 100+ FPS? Let's go!** 🚀
