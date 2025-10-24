# FunGen Rewrite - Requirements Summary

**Quick Reference Guide for Development Team**

---

## Mission Critical (P0) - 8 Components

### 1. Video Processing Pipeline
- **Target:** 100+ FPS (1080p), 60+ FPS (8K) on RTX 3090
- **Key Tech:** FFmpeg, OpenCV, GPU-accelerated decoding
- **Deliverable:** Streaming frame extraction with <20GB VRAM usage

### 2. YOLO Object Detection
- **Current:** Ultralytics 8.3.78
- **Models:** .pt, .onnx, .engine (TensorRT), .mlpackage (Core ML)
- **Detection:** Penis, hands, mouth tracking
- **Optimization:** FP16 TensorRT, model caching (.msgpack)

### 3. Motion Tracking System
- **Architecture:** Dynamic tracker discovery, modular plugins
- **Categories:** Live, Live+Intervention, Offline, Community
- **Must Exceed:** Enhanced Axis Projection Tracker (current best: ~110 FPS)
- **Implementations:** ByteTrack baseline, Advanced (ByteTrack+OpticalFlow+Kalman+ReID)

### 4. Funscript Generation
- **Outputs:** .funscript, .roll.funscript, .fgp, _t1_raw.funscript
- **Pipeline:** Tracking → Motion Analysis → Position Calculation → Post-processing
- **Caching:** .msgpack (Stage 1), _stage2_overlay.msgpack (Stage 2)

### 5. Filter Plugin System
- **Critical Filters:** Ultimate Autotune (7-stage), Speed Limiter, Smooth (SG)
- **Architecture:** Chainable pipeline, parameter configuration
- **CLI Support:** --funscript-mode --filter for batch processing

### 6. GUI Application
- **Framework:** tkinter + sv_ttk theme + GPU rendering (moderngl)
- **New Feature:** Real-time agent progress visualization
- **Components:** Video Display, Control Panel, Navigation, Info Graphs
- **Performance:** GPU-accelerated video, automatic DPI scaling

### 7. CLI Mode
- **Usage:** `python main.py [path] --mode [tracker] [options]`
- **Features:** Batch processing, recursive folders, parallel instances
- **Performance:** 160-190 FPS aggregate (3-6 parallel instances)

### 8. Cross-Platform Compatibility
- **Dev:** Raspberry Pi 4/5 (ARM64, CPU-only) for logic testing
- **Prod:** RTX 3090 (24GB VRAM) for performance
- **Key:** Conditional GPU imports (torch.cuda.is_available())

---

## High Priority (P1) - 8 Features

1. **Video Format Detection** - Auto-detect VR/2D, SBS, Fisheye/Equirectangular
2. **Chapter/Segmentation** - Stage 2 scene detection (BJ/HJ chapters)
3. **Project Files (.fgp)** - Settings persistence, checkpoint resume
4. **Update System** - GitHub API integration, version selection
5. **TensorRT Compiler** - .pt → .engine conversion GUI
6. **File Management** - Output organization, generated file manager
7. **Keyboard Shortcuts** - Full keyboard-centric workflow
8. **Dependency Management** - Auto-install missing packages on startup

---

## Nice-to-Have (P2) - 5 Features

1. **Visualization/Debug** - Overlay bounding boxes, trajectories, heatmaps
2. **Device Integration** - Real-time funscript playback to hardware
3. **Calibration System** - Video offset, position range, speed tuning
4. **Energy Saver** - GPU power management, idle throttling
5. **Advanced Filters** - Anti-jerk, beat marker (visual/audio)

---

## Key FunGen Strengths to Preserve

### 1. Modular Architecture
- Plugin system for trackers and filters
- Dynamic discovery and registration
- Zero hardcoded tracker references

### 2. Multi-Stage Processing
- Stage 1: YOLO detection (cached)
- Stage 2: Tracking/segmentation (cached)
- Stage 3: Funscript generation
- Iterative refinement without re-detection

### 3. Professional GUI
- Single-window integrated interface
- Real-time video + overlay visualization
- Keyboard-centric workflow
- GPU-accelerated rendering

### 4. Batch Processing Excellence
- CLI automation support
- Recursive folder processing
- Parallel instance capability (3-6x speedup)
- Smart cache reuse

### 5. Cross-Platform Model Support
- Multiple formats (.pt/.onnx/.engine/.mlpackage)
- Automatic hardware detection
- TensorRT optimization for NVIDIA

---

## Performance Benchmarks

### Current (FunGen)
- Single process: 60-110 FPS (8K VR)
- Parallel (3-6 instances): 160-190 FPS aggregate
- Processing: 20-30 min per 8K VR video

### Target (Rewrite)
| Resolution | Target FPS | Hardware | VRAM |
|------------|------------|----------|------|
| 1080p | 100+ | RTX 3090 | <20GB |
| 4K | 80+ | RTX 3090 | <20GB |
| 8K | 60+ | RTX 3090 | <20GB |

### Quality Targets
- 80%+ test coverage
- Zero code duplication
- Type hints mandatory
- Black + isort + mypy compliance

---

## Critical Technical Decisions

### 1. Tracker Implementation
**Must improve upon:** Enhanced Axis Projection Tracker
- Multi-scale analysis
- Temporal coherence
- Adaptive thresholding

**New approach:**
- ByteTrack as baseline (fast, reliable)
- Advanced: ByteTrack + OpticalFlow + Kalman + ReID

### 2. GPU Optimization
**NVIDIA (Primary):**
- CUDA 12.8
- TensorRT FP16 engines
- 20xx/30xx/40xx support (standard)
- 50xx support (special requirements)

**Other Platforms:**
- AMD: ROCm (Linux only)
- Apple: Core ML (.mlpackage)
- CPU: Fallback mode

### 3. UI Framework
**Keep:** tkinter + sv_ttk
- Lightweight, cross-platform
- GPU rendering via moderngl
- Professional appearance

**Add:** Real-time agent progress
- Multi-agent status display
- Task completion indicators
- FPS/VRAM monitoring

---

## Tracker Ecosystem

### Live Trackers (Batch-Compatible)
- **Oscillation Detector** (2 modes: Current, Legacy)
- **YOLO ROI Tracker** (auto ROI detection)
- **Relative Distance Tracker** (high-performance)
- **Hybrid Intelligence Tracker** (multi-modal)

### Offline Trackers (Multi-Stage)
- **Contact Analysis (2-Stage)** - Contact detection only
- **Mixed Processing (3-Stage)** - Best accuracy, selective tracking
- **Optical Flow Analysis (3-Stage)** - Motion-based

### Experimental Trackers
- **Enhanced Axis Projection** - Current best performer
- **Working Axis Projection** - Simplified reliable version

### Live + Intervention (Not Batch-Compatible)
- **User ROI Tracker** - Manual ROI selection
- **DOT Marker** - Track colored point

---

## Filter Pipeline

### Essential Filters
1. **Ultimate Autotune** - Default 7-stage enhancement (can disable)
2. **Speed Limiter** - Hardware compatibility + vibrations
3. **Smooth (SG)** - Savitzky-Golay polynomial smoothing

### Simplification Filters
4. **Simplify (RDP)** - Remove redundant points
5. **Keyframes** - Extract peaks/valleys only
6. **Resample** - Regular interval resampling

### Transformation Filters
7. **Amplify** - Scale around center point
8. **Invert** - Flip positions (0↔100)
9. **Clamp** - Force to specific value
10. **Threshold Clamp** - Binary clamping

---

## Architecture Patterns

### Module Organization
```
core/
  video/          # Video processing pipeline
  ml/             # YOLO model management
  tracking/       # Tracker implementations

trackers/
  bytetrack/      # Baseline tracker
  advanced/       # ByteTrack+OpticalFlow+Kalman+ReID

ui/
  main_window.py  # Primary GUI
  video_player.py # GPU-accelerated display
  progress.py     # Agent progress visualization

utils/
  cross_platform.py  # Pi/RTX compatibility
```

### Inter-Agent Communication
**Shared Files:**
- docs/architecture.md - Module interfaces
- docs/agent_assignments.json - Work distribution
- progress/*.json - Individual agent status

**Progress Format:**
```json
{
  "agent": "tracker-dev-1",
  "progress": 75,
  "status": "working",
  "current_task": "Implementing ByteTrack core loop",
  "timestamp": "2025-10-24T19:45:00Z"
}
```

---

## Dependencies Matrix

### Core (All Platforms)
```
numpy, opencv-python, scipy, scikit-learn
ultralytics==8.3.78 (YOLO)
msgpack, pillow, imageio
tkinter, imgui, glfw, pyopengl, moderngl
tqdm, colorama, send2trash
```

### GPU-Specific
**NVIDIA 20xx-40xx:**
```
torch==2.8.0+cu128
torchvision==0.23.0+cu128
tensorrt
```

**NVIDIA 50xx:** Special requirements file

**AMD Linux:** ROCm requirements

**CPU Only:** CPU-specific PyTorch

### External Tools
- FFmpeg (required)
- FFprobe (required)
- Git (optional, for updates)

---

## Risk Mitigation

### High Risk: Performance Target
**Mitigation:**
- Benchmark suite for continuous monitoring
- Incremental optimization approach
- Preserve existing trackers as fallback

### High Risk: Tracking Quality
**Mitigation:**
- A/B testing framework
- Quality metrics (smoothness, accuracy, naturalness)
- User feedback integration

### Medium Risk: Cross-Platform
**Mitigation:**
- Conditional GPU imports everywhere
- CPU fallback for all GPU operations
- Test on both Pi and RTX 3090

### Medium Risk: Agent Coordination
**Mitigation:**
- Clear file-based communication protocol
- JSON progress updates every 2 minutes
- Inbox/outbox for bug fixes

---

## Success Metrics

### Performance
- [x] Documented: 100+ FPS target (1080p)
- [x] Documented: 60+ FPS target (8K)
- [ ] Implemented: <20GB VRAM usage
- [ ] Tested: 80%+ unit test coverage

### Functionality
- [x] Documented: All P0 features identified
- [x] Documented: All P1 features identified
- [ ] Implemented: CLI batch processing
- [ ] Implemented: Real-time agent progress UI

### Quality
- [ ] Code: Zero duplication
- [ ] Code: Type hints on all functions
- [ ] Code: Google-style docstrings
- [ ] Code: Black + isort + mypy pass

### Platform
- [ ] Tested: Raspberry Pi (CPU mode)
- [ ] Tested: RTX 3090 (GPU mode)
- [ ] Verified: Conditional imports working
- [ ] Verified: Cross-platform tests passing

---

## Quick CLI Reference

```bash
# GUI mode (no arguments)
python main.py

# Single video
python main.py "/path/to/video.mp4"

# Batch processing (recursive)
python main.py "/path/to/folder" --mode 3-stage --recursive

# Force reprocessing
python main.py "/path/to/folder" --mode oscillation --overwrite

# Dual-axis generation
python main.py "/path/to/video.mp4" --mode 3-stage --generate-roll

# Funscript filtering only
python main.py "/path/to/video.funscript" --funscript-mode --filter ultimate-autotune

# Performance mode (no autotune, no copy)
python main.py "/path/to/video.mp4" --mode yolo --no-autotune --no-copy

# Recursive with legacy oscillation detector
python main.py "/path/to/folder" --mode oscillation --od-mode legacy -r
```

---

## Next Steps for Development Team

### Immediate (Minutes 5-10)
1. **project-architect** - Create docs/architecture.md with module interfaces
2. **project-architect** - Create docs/agent_assignments.json with work distribution

### Core Development (Minutes 5-25, parallel)
1. **video-specialist** - Video processing pipeline
2. **ml-specialist** - YOLO model manager with TensorRT
3. **tracker-dev-1** - ByteTrack implementation
4. **tracker-dev-2** - Advanced tracker (ByteTrack+OpticalFlow+Kalman+ReID)
5. **ui-architect** - Core UI structure (tkinter + sv_ttk)
6. **ui-enhancer** - Agent progress visualization
7. **cross-platform-dev** - Pi/RTX compatibility layer

### Quality Assurance (Minutes 15-30, continuous)
1. **test-engineer-1** - Unit tests (80%+ coverage, Pi CPU testing)
2. **test-engineer-2** - Integration tests, benchmarks
3. **integration-master** - Combine work, remove duplicates
4. **code-quality** - Black, isort, mypy, docstrings

### Debug Loop (Minutes 15-30, on-demand)
1. **gpu-debugger** - CUDA errors → fixes → wait → re-test
2. **python-debugger** - General bugs → fixes → notify agents

---

**Requirements Analysis Complete**
**Total Features Documented:** 21 (8 P0, 8 P1, 5 P2)
**Total Trackers Analyzed:** 10+ implementations
**Total Filters Analyzed:** 11 filter types
**Performance Target:** 100+ FPS (up from 60-110 FPS)
