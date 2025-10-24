# FunGen Rewrite - System Architecture Document

**Version:** 1.0
**Date:** 2025-10-24
**Author:** project-architect agent
**Target Platform:** Raspberry Pi (dev) + RTX 3090 (prod)

---

## Executive Summary

This document defines the architecture for rewriting FunGen with:
- **100+ FPS** tracking on RTX 3090 (current: 60-110 FPS)
- **Modular design** with zero code duplication
- **Cross-platform support** (Pi CPU mode for dev, RTX 3090 for production)
- **Modern UI** with real-time agent progress visualization
- **Advanced tracking** surpassing Enhanced Axis Projection

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Module Specifications](#module-specifications)
4. [Performance Optimization Strategy](#performance-optimization-strategy)
5. [Cross-Platform Design](#cross-platform-design)
6. [Data Flow](#data-flow)
7. [API Contracts](#api-contracts)
8. [Testing Strategy](#testing-strategy)

---

## 1. System Overview

### 1.1 Design Principles

1. **Separation of Concerns**: Video, ML, tracking, and UI are independent modules
2. **Hardware Abstraction**: Single codebase for CPU (Pi) and GPU (RTX 3090)
3. **Pipeline Parallelism**: Decode, detect, track, and encode in parallel
4. **Memory Efficiency**: Stream processing, <20GB VRAM usage
5. **Zero Duplication**: Shared utilities, single source of truth

### 1.2 Key Performance Targets

| Metric | Current (FunGen) | Target (Rewrite) | Strategy |
|--------|------------------|------------------|----------|
| 1080p FPS | 60-110 | 100+ | TensorRT FP16, batching |
| 8K FPS | N/A | 60+ | Multi-GPU, PyNvVideoCodec |
| VRAM Usage | Variable | <20GB | Quantization, streaming |
| Test Coverage | ~0% | 80%+ | Unit + integration tests |
| CPU Mode | N/A | Functional | Conditional GPU imports |

### 1.3 Technology Stack

**Core:**
- Python 3.11+ (type hints mandatory)
- PyTorch 2.x with CUDA 12.8 / ROCm (optional)
- TensorRT 10.x (FP16 quantization)
- FFmpeg 7.x / PyNvVideoCodec 2.0

**ML/Tracking:**
- Ultralytics YOLO11 (.engine models)
- ByteTrack (baseline, speed-focused)
- BoT-SORT (accuracy-focused with ReID)
- Optical Flow (cv2.cuda.FarnebackOpticalFlow)
- Kalman Filter (cv2.KalmanFilter)

**UI:**
- tkinter with sv_ttk theme
- matplotlib for real-time graphs
- Threading for non-blocking updates

**Development:**
- pytest (testing), Black (formatting), mypy (type checking)
- Google-style docstrings

---

## 2. Core Architecture

### 2.1 Module Hierarchy

```
elo_elo_320/
├── core/                    # Business logic (agent: video-specialist, ml-specialist)
│   ├── video_pipeline.py    # FFmpeg wrapper, frame streaming
│   ├── model_manager.py     # YOLO model loading, TensorRT optimization
│   ├── batch_processor.py   # Multi-video queue manager
│   └── config.py            # Global configuration, hardware detection
├── trackers/                # Tracking algorithms (agent: tracker-dev-1, tracker-dev-2)
│   ├── base_tracker.py      # Abstract base class with common interfaces
│   ├── bytetrack.py         # Fast baseline (50ms latency target)
│   ├── botsort.py           # Accuracy + ReID (100ms latency target)
│   ├── hybrid_tracker.py    # ByteTrack + OpticalFlow + Kalman + ReID
│   └── trackers_legacy/     # FunGen trackers (reference only)
├── utils/                   # Shared utilities (agent: cross-platform-dev)
│   ├── hardware.py          # GPU detection, CUDA/ROCm/CPU abstraction
│   ├── metrics.py           # FPS counter, VRAM monitor
│   ├── funscript.py         # .funscript I/O, validation
│   └── logger.py            # Structured logging
├── ui/                      # GUI (agent: ui-architect, ui-enhancer)
│   ├── main_window.py       # Primary tkinter window
│   ├── agent_dashboard.py   # Real-time agent progress visualization
│   ├── settings_panel.py    # Configuration UI
│   └── components/          # Reusable widgets (progress bars, tooltips)
├── tests/                   # Testing (agent: test-engineer-1, test-engineer-2)
│   ├── unit/                # 80%+ coverage target
│   ├── integration/         # End-to-end pipeline tests
│   └── benchmarks/          # FPS, VRAM, accuracy benchmarks
├── docs/                    # Documentation
│   ├── architecture.md      # This file
│   └── agent_assignments.json
├── progress/                # Agent progress tracking
├── communication/           # Inter-agent message passing
└── output/                  # Generated funscripts
```

### 2.2 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         UI Layer (tkinter)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│  │ Main Window │  │ Agent        │  │ Settings Panel     │     │
│  │             │←→│ Dashboard    │←→│                    │     │
│  └─────────────┘  └──────────────┘  └────────────────────┘     │
└────────────────────────────┬────────────────────────────────────┘
                             │ Threading (non-blocking)
┌────────────────────────────┼────────────────────────────────────┐
│                    Core Processing Layer                         │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────┐        │
│  │ Video        │→→│ Model      │→→│ Batch Processor  │        │
│  │ Pipeline     │  │ Manager    │  │ (Queue Manager)  │        │
│  └──────────────┘  └────────────┘  └──────────────────┘        │
│         ↓                ↓                    ↓                  │
│  ┌──────────────────────────────────────────────────┐           │
│  │         Hardware Abstraction (CUDA/ROCm/CPU)     │           │
│  └──────────────────────────────────────────────────┘           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                    Tracking Layer                                │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────────┐        │
│  │ ByteTrack    │  │ BoT-SORT   │  │ Hybrid Tracker   │        │
│  │ (Fast)       │  │ (Accurate) │  │ (Production)     │        │
│  └──────────────┘  └────────────┘  └──────────────────┘        │
│         ↓                ↓                    ↓                  │
│  ┌──────────────────────────────────────────────────┐           │
│  │      Optical Flow + Kalman Filter + ReID         │           │
│  └──────────────────────────────────────────────────┘           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                      Funscript Output
```

---

## 3. Module Specifications

### 3.1 Core Modules

#### 3.1.1 Video Pipeline (`core/video_pipeline.py`)

**Responsibility:** Decode video files efficiently, stream frames to tracking pipeline

**Key Features:**
- FFmpeg-based decoding with hardware acceleration (NVDEC for NVIDIA)
- PyNvVideoCodec 2.0 for multi-GPU decode, frame sampling, threaded decoding
- Frame buffer management (circular buffer, max 60 frames in memory)
- VR support: SBS Fisheye/Equirectangular 180° detection
- Preprocessing: crop, resize, normalize

**API:**
```python
class VideoPipeline:
    def __init__(self, video_path: str, hw_accel: bool = True):
        """Initialize video decoder with optional hardware acceleration."""

    def get_metadata(self) -> VideoMetadata:
        """Return resolution, fps, duration, VR format."""

    def stream_frames(self, batch_size: int = 8) -> Iterator[FrameBatch]:
        """Yield batches of frames (numpy arrays) for processing."""

    def seek(self, frame_num: int) -> None:
        """Jump to specific frame (for multi-pass processing)."""
```

**Performance Target:** 200+ FPS decode (1080p), 60+ FPS (8K)

#### 3.1.2 Model Manager (`core/model_manager.py`)

**Responsibility:** Load YOLO models, optimize with TensorRT, manage VRAM

**Key Features:**
- Auto-detect best model format (.engine > .onnx > .pt)
- TensorRT FP16 quantization (40% latency reduction: 22ms → 13ms)
- Multi-GPU support (MPS for concurrent inference)
- Lazy loading (load on first use)
- Model warmup (pre-compile CUDA kernels)

**API:**
```python
class ModelManager:
    def __init__(self, model_dir: Path, device: str = "auto"):
        """Initialize with model directory, auto-detect GPU."""

    def load_model(self, model_name: str, optimize: bool = True) -> YOLOModel:
        """Load model, apply TensorRT optimization if enabled."""

    def predict_batch(self, frames: FrameBatch) -> List[Detection]:
        """Run inference on batch, return bounding boxes."""

    def get_vram_usage(self) -> float:
        """Return current VRAM usage in GB."""
```

**Performance Target:** <20GB VRAM, 100+ FPS inference (1080p, FP16)

#### 3.1.3 Batch Processor (`core/batch_processor.py`)

**Responsibility:** Manage multi-video queues, parallel processing

**Key Features:**
- Queue-based video processing (asyncio or multiprocessing)
- CLI mode: spawn N processes (optimal N = 3-6 for RTX 3090)
- GUI mode: thread pool with progress callbacks
- Crash recovery (checkpoint system)

**API:**
```python
class BatchProcessor:
    def __init__(self, num_workers: int = "auto"):
        """Initialize with auto-detected optimal worker count."""

    def add_video(self, video_path: Path, settings: ProcessingSettings) -> JobID:
        """Add video to queue, return job ID."""

    def process(self, callback: Callable[[Progress], None]) -> None:
        """Start processing, call callback for progress updates."""

    def cancel_job(self, job_id: JobID) -> None:
        """Cancel specific job."""
```

**Performance Target:** 160-190 FPS (8K VR, 3-6 parallel processes)

---

### 3.2 Tracker Modules

#### 3.2.1 Base Tracker (`trackers/base_tracker.py`)

**Responsibility:** Abstract base class defining tracker interface

**API:**
```python
class BaseTracker(ABC):
    @abstractmethod
    def initialize(self, detections: List[Detection]) -> None:
        """Initialize tracker with first frame detections."""

    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Track]:
        """Update with new detections, return tracked objects."""

    @abstractmethod
    def get_funscript_data(self) -> FunscriptData:
        """Convert tracks to funscript positions (0-100)."""
```

#### 3.2.2 ByteTrack (`trackers/bytetrack.py`)

**Responsibility:** Fast baseline tracker for real-time performance

**Key Features:**
- Simple association (IoU matching)
- Low/high confidence detection handling
- Kalman filter for motion prediction
- Target latency: 50ms per frame

**Performance Target:** 100+ FPS (1080p), minimal VRAM

#### 3.2.3 BoT-SORT (`trackers/botsort.py`)

**Responsibility:** High-accuracy tracker with re-identification

**Key Features:**
- Camera motion compensation
- Appearance-based ReID (ResNet50 embeddings)
- Extended Kalman filter
- Target latency: 100ms per frame

**Performance Target:** 60+ FPS (1080p), +2GB VRAM for ReID

#### 3.2.4 Hybrid Tracker (`trackers/hybrid_tracker.py`)

**Responsibility:** Production tracker combining multiple techniques

**Key Features:**
- ByteTrack foundation (fast association)
- Optical flow for inter-frame motion (cv2.cuda.FarnebackOpticalFlow)
- Kalman filter for smooth predictions
- Optional ReID for long-term tracking
- Adaptive algorithm selection based on scene complexity

**Algorithm Flow:**
1. YOLO detection → bounding boxes
2. ByteTrack → initial track IDs
3. Optical flow → refine positions between frames
4. Kalman filter → smooth trajectories, predict occlusions
5. ReID → recover lost tracks after occlusion

**Performance Target:** 80+ FPS (1080p), adaptive VRAM usage

---

### 3.3 Utility Modules

#### 3.3.1 Hardware Abstraction (`utils/hardware.py`)

**Responsibility:** Detect GPU, provide unified API for CUDA/ROCm/CPU

**API:**
```python
def detect_hardware() -> HardwareInfo:
    """Return GPU type, VRAM, CUDA/ROCm version."""

def get_device(prefer_gpu: bool = True) -> str:
    """Return 'cuda', 'rocm', or 'cpu'."""

def optimize_batch_size(model_vram: float, available_vram: float) -> int:
    """Calculate optimal batch size for given VRAM."""
```

#### 3.3.2 Metrics (`utils/metrics.py`)

**Responsibility:** Real-time performance monitoring

**API:**
```python
class PerformanceMonitor:
    def start_frame(self) -> None:
        """Mark frame processing start."""

    def end_frame(self) -> None:
        """Mark frame processing end, update FPS."""

    def get_fps(self) -> float:
        """Return rolling average FPS (last 30 frames)."""

    def get_vram_usage(self) -> float:
        """Return current VRAM usage."""
```

---

### 3.4 UI Modules

#### 3.4.1 Main Window (`ui/main_window.py`)

**Responsibility:** Primary application window

**Features:**
- Video selection (file/folder browser)
- Tracker selection dropdown
- Settings access
- Start/stop/pause controls
- Real-time FPS/VRAM display

#### 3.4.2 Agent Dashboard (`ui/agent_dashboard.py`)

**Responsibility:** Visualize agent progress (unique feature)

**Features:**
- Read from `progress/*.json` every 2 seconds
- Display agent status (pending/in_progress/completed)
- Progress bars for each agent (0-100%)
- Error/warning indicators
- Expandable logs

**Design:**
```
┌─────────────────────────────────────────────┐
│ Agent Dashboard                              │
├─────────────────────────────────────────────┤
│ [■■■■■■■■■■] video-specialist (100%)        │
│ [■■■■■■□□□□] ml-specialist (60%) WORKING... │
│ [□□□□□□□□□□] tracker-dev-1 (0%) PENDING     │
│ [■■■■■■■■■■] test-engineer-1 (100%) ✓       │
└─────────────────────────────────────────────┘
```

---

## 4. Performance Optimization Strategy

### 4.1 TensorRT FP16 Optimization

**Implementation:**
1. Train YOLO model in FP32 (PyTorch)
2. Export to ONNX format
3. Convert to TensorRT engine with FP16 precision:
   ```python
   from ultralytics import YOLO
   model = YOLO("yolo11n.pt")
   model.export(format="engine", half=True)  # FP16
   ```
4. Expected speedup: 40% (22ms → 13ms per frame)

**VRAM Savings:** 50% (model size halved)

### 4.2 Batching Strategy

**Current FunGen:** Frame-by-frame processing (batch_size=1)
**Target:** Dynamic batching (batch_size=4-8)

**Implementation:**
```python
# Accumulate frames in buffer
buffer = []
for frame in video_pipeline.stream_frames():
    buffer.append(frame)
    if len(buffer) >= batch_size:
        detections = model.predict_batch(buffer)
        buffer.clear()
```

**Expected Speedup:** 30-50% (GPU utilization increases)

### 4.3 Multi-GPU Decode with PyNvVideoCodec 2.0

**Implementation:**
```python
from PyNvVideoCodec import Decoder

decoder = Decoder(
    video_path,
    gpu_id=0,
    threaded=True,  # Background decoding
    target_fps=30   # Skip frames for faster processing
)

for frame_batch in decoder.decode_batch(batch_size=8):
    # Process batch
```

**Features:**
- Zero-latency threaded decoding
- Multi-GPU scaling
- Frame sampling (skip frames for speed)

### 4.4 Parallel Processing (CLI Mode)

**Strategy:** Spawn 3-6 processes for multi-video processing

**Implementation:**
```python
import multiprocessing as mp

def process_video(video_path, gpu_id):
    # Each process loads model on separate GPU context
    model = ModelManager(device=f"cuda:{gpu_id % num_gpus}")
    # Process video...

with mp.Pool(processes=6) as pool:
    pool.starmap(process_video, [(v, i) for i, v in enumerate(videos)])
```

**Expected Performance:** 160-190 FPS (8K VR, RTX 3090)

**VRAM Management:** Each process uses 3-4GB, total <20GB

### 4.5 Optical Flow GPU Acceleration

**Current FunGen:** CPU-based optical flow (slow)
**Target:** CUDA optical flow

**Implementation:**
```python
import cv2

# Create GPU optical flow object
gpu_flow = cv2.cuda.FarnebackOpticalFlow_create(
    numLevels=3,
    pyrScale=0.5,
    fastPyramids=True
)

# Upload frames to GPU
gpu_frame1 = cv2.cuda_GpuMat()
gpu_frame1.upload(frame1)

# Compute flow on GPU
gpu_flow_result = gpu_flow.calc(gpu_frame1, gpu_frame2, None)
flow = gpu_flow_result.download()
```

**Expected Speedup:** 5-10x vs CPU optical flow

---

## 5. Cross-Platform Design

### 5.1 Conditional GPU Imports

**Problem:** Pi doesn't have CUDA, RTX 3090 requires it
**Solution:** Conditional imports with graceful fallbacks

**Implementation:**
```python
# utils/hardware.py
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

# core/model_manager.py
if TORCH_AVAILABLE:
    from torch import inference_mode
else:
    from contextlib import contextmanager
    @contextmanager
    def inference_mode():
        yield
```

### 5.2 CPU Fallback Mode

**Features:**
- Use .onnx models with ONNX Runtime (CPU optimized)
- Reduce batch size to 1
- Disable optical flow (too slow on CPU)
- Use ByteTrack only (BoT-SORT ReID too slow)

**Performance Target (Pi):** 5-10 FPS (sufficient for development testing)

### 5.3 Configuration Profiles

**Profiles:**
- `dev_pi`: CPU mode, minimal features, logging enabled
- `prod_rtx3090`: GPU mode, all features, TensorRT FP16
- `debug`: CPU/GPU with extensive logging

**Implementation:**
```python
# core/config.py
class Config:
    PROFILE = os.getenv("FUNGEN_PROFILE", "auto")

    if PROFILE == "auto":
        PROFILE = "prod_rtx3090" if TORCH_AVAILABLE else "dev_pi"

    # Load profile-specific settings
    settings = load_profile(PROFILE)
```

---

## 6. Data Flow

### 6.1 Single Video Processing Flow

```
1. Video File
   ↓ (VideoPipeline.stream_frames)
2. Frame Batches (numpy arrays)
   ↓ (ModelManager.predict_batch)
3. YOLO Detections (bounding boxes)
   ↓ (Tracker.update)
4. Tracked Objects (IDs + positions)
   ↓ (Tracker.get_funscript_data)
5. Funscript Data (positions 0-100)
   ↓ (FunscriptWriter.save)
6. .funscript File
```

### 6.2 Multi-Video Processing Flow (CLI)

```
1. Video Directory
   ↓ (BatchProcessor.add_video)
2. Job Queue
   ↓ (spawn N processes)
3. Parallel Processing (N videos simultaneously)
   ↓ (each process: VideoPipeline → Model → Tracker)
4. Individual Results
   ↓ (merge)
5. Output Directory (.funscript files)
```

### 6.3 Inter-Agent Communication Flow

```
1. Agent updates progress/agent_name.json every 2 minutes
2. UI reads all progress/*.json files every 2 seconds
3. Agent Dashboard displays real-time status
4. If bug found:
   - Debugger writes to communication/{agent}_inbox.json
   - Agent reads inbox on next cycle
   - Agent applies fix, tests, writes to _outbox.json
   - Debugger confirms fix
```

---

## 7. API Contracts

### 7.1 Detection Format

```python
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str  # "penis", "hand", "mouth", etc.
```

### 7.2 Track Format

```python
@dataclass
class Track:
    track_id: int
    detections: List[Detection]  # History of detections
    positions: List[Tuple[int, int]]  # (x, y) center positions
    velocities: List[Tuple[float, float]]  # (vx, vy) from Kalman
    confidence: float  # Average confidence
```

### 7.3 Funscript Format

```json
{
  "version": "1.0",
  "inverted": false,
  "range": 90,
  "actions": [
    {"at": 0, "pos": 50},
    {"at": 500, "pos": 80},
    {"at": 1000, "pos": 20}
  ]
}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests (80%+ Coverage Target)

**Modules to Test:**
- `core/video_pipeline.py`: Test decode, seek, metadata
- `core/model_manager.py`: Test model loading, inference
- `trackers/*.py`: Test association logic, Kalman filter
- `utils/*.py`: Test hardware detection, metrics

**Tools:** pytest, pytest-cov, pytest-mock

### 8.2 Integration Tests

**Test Cases:**
1. End-to-end: Video → Funscript
2. Multi-video processing (CLI mode)
3. GPU/CPU fallback behavior
4. Error handling (corrupted video, OOM)

### 8.3 Benchmark Tests

**Metrics:**
- FPS (1080p, 8K)
- VRAM usage
- CPU usage
- Funscript quality (compare to FunGen)

**Target Hardware:**
- Raspberry Pi 4/5 (dev)
- RTX 3090 (prod)

---

## 9. Migration from FunGen

### 9.1 Preserved Features

- Funscript generation (.funscript format)
- VR video support (SBS Fisheye/Equirectangular)
- Batch processing (folders)
- GUI with progress tracking
- CLI mode for automation

### 9.2 Deprecated Features (Keep as Reference)

- FunGen trackers (move to `trackers_legacy/`)
- Filter plugins (Ultimate Autotune, etc.) - defer to Phase 2
- Manual ROI selection - defer to Phase 2

### 9.3 New Features

- Agent progress dashboard (unique to rewrite)
- Real-time FPS/VRAM monitoring
- TensorRT FP16 optimization
- Multi-GPU support
- Cross-platform (Pi + RTX 3090)

---

## 10. Implementation Phases

### Phase 1: Core Pipeline (Minutes 5-15)
- video-specialist: `core/video_pipeline.py`
- ml-specialist: `core/model_manager.py`
- cross-platform-dev: `utils/hardware.py`, `utils/metrics.py`

### Phase 2: Tracking (Minutes 10-20)
- tracker-dev-1: `trackers/bytetrack.py`
- tracker-dev-2: `trackers/hybrid_tracker.py` (ByteTrack + Optical Flow + Kalman)

### Phase 3: UI (Minutes 10-25)
- ui-architect: `ui/main_window.py`, `ui/agent_dashboard.py`
- ui-enhancer: Polish, tooltips, themes

### Phase 4: Testing (Minutes 15-30, parallel)
- test-engineer-1: Unit tests (80%+ coverage)
- test-engineer-2: Integration tests, benchmarks

### Phase 5: Integration (Minutes 20-30)
- integration-master: Combine modules, remove duplicates, final testing

---

## 11. Success Criteria

1. **Performance:**
   - ✓ 100+ FPS (1080p, RTX 3090)
   - ✓ 60+ FPS (8K, RTX 3090)
   - ✓ <20GB VRAM

2. **Quality:**
   - ✓ 80%+ test coverage
   - ✓ Type hints on all functions
   - ✓ Zero mypy errors

3. **Functionality:**
   - ✓ Works on Pi (CPU mode)
   - ✓ Works on RTX 3090 (GPU mode)
   - ✓ Agent dashboard functional
   - ✓ CLI + GUI modes

4. **Architecture:**
   - ✓ Zero code duplication
   - ✓ Modular design (can swap trackers)
   - ✓ Clear separation of concerns

---

## 12. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| TensorRT installation issues | High | High | Provide pre-built .engine models |
| CUDA OOM on RTX 3090 | Medium | High | Dynamic batch sizing, VRAM monitoring |
| Pi too slow for testing | Low | Medium | Mock GPU modules for unit tests |
| Tracker accuracy regression | Medium | High | Benchmark against FunGen, A/B testing |
| Agent coordination overhead | Low | Medium | Use simple JSON files, avoid complexity |

---

## Appendix A: FunGen Analysis

### A.1 Current Architecture (FunGen)

- **Monolithic design:** `application/logic/app_logic.py` (5000+ lines)
- **Tight coupling:** UI, video, ML, tracking all intertwined
- **No hardware abstraction:** GPU/CPU code mixed
- **Limited testing:** No unit tests

### A.2 Performance Bottlenecks (FunGen)

1. **Frame-by-frame processing:** No batching
2. **Python YOLO.track:** Not parallelizable
3. **CPU optical flow:** No GPU acceleration
4. **No TensorRT:** Using .pt models (FP32)

### A.3 Improvements in Rewrite

| Aspect | FunGen | Rewrite | Improvement |
|--------|--------|---------|-------------|
| Batching | No | Yes (4-8) | +30-50% FPS |
| TensorRT | No | Yes (FP16) | +40% FPS |
| Optical Flow | CPU | GPU | +5-10x |
| Multi-GPU | No | Yes | 160-190 FPS (parallel) |
| Code Structure | Monolithic | Modular | Maintainable |
| Testing | 0% | 80%+ | Reliable |

---

## Appendix B: Research Summary

### B.1 YOLO Optimization (2025)

- **YOLO26:** Latest model, optimized for edge devices
- **TensorRT FP16:** 40% latency reduction (22ms → 13ms)
- **INT8 Quantization:** Further speedup, minimal accuracy loss

### B.2 ByteTrack vs BoT-SORT

| Metric | ByteTrack | BoT-SORT |
|--------|-----------|----------|
| Speed | Fast (real-time) | Moderate |
| Accuracy | Good | Excellent |
| ReID | No | Yes |
| Use Case | Speed-critical | Accuracy-critical |

**Recommendation:** Use ByteTrack as baseline, BoT-SORT for high-accuracy mode

### B.3 PyNvVideoCodec 2.0 (2025)

- **Threaded decoding:** Zero-latency frame delivery
- **Multi-GPU:** Scale across GPUs
- **Frame sampling:** Skip frames for speed
- **5x speedup:** vs CPU decoding

### B.4 Kalman Filter + Optical Flow

- **Real-time:** 50ms per frame (640x480)
- **Accuracy:** Optical flow guides Kalman predictions
- **Occlusion Handling:** Kalman predicts during occlusion, optical flow recovers

---

## Appendix C: Hardware Specs

### C.1 Development Platform (Raspberry Pi)

- **CPU:** ARM64 (Cortex-A76)
- **RAM:** 8GB
- **GPU:** None (CPU-only mode)
- **Storage:** SD card (slow I/O)
- **Use Case:** Development, unit tests

### C.2 Production Platform (RTX 3090)

- **GPU:** NVIDIA RTX 3090
- **VRAM:** 24GB GDDR6X
- **CUDA Cores:** 10496
- **Tensor Cores:** 328 (3rd gen)
- **CPU:** AMD Ryzen 2990 (64 threads)
- **RAM:** 48GB
- **Use Case:** Production inference

---

## Document Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-24 | project-architect | Initial architecture document |

