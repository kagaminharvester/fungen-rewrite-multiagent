# FunGen Rewrite - Comprehensive Requirements Analysis

**Document Version:** 1.0
**Date:** 2025-10-24
**Analyst:** requirements-analyst
**Source:** https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator

---

## Executive Summary

FunGen is a Python-based AI-powered tool that generates Funscript files from VR and 2D POV videos. The rewrite aims to achieve 100+ FPS tracking performance on RTX 3090, modern UI with real-time agent progress visualization, and cross-platform compatibility between Raspberry Pi (development) and RTX 3090 (production).

**Current Performance:** ~60-110 FPS (8K video, RTX 3090)
**Target Performance:** 100+ FPS (1080p), 60+ FPS (8K)
**Target VRAM:** <20GB usage
**Test Coverage:** 80%+

---

## Priority Classification

- **P0 (Critical):** Must-have features for core functionality
- **P1 (High):** Important features that enhance primary use cases
- **P2 (Nice-to-have):** Quality of life improvements and advanced features

---

## P0 - Critical Features

### 1. Core Video Processing Pipeline

**Priority:** P0
**Current Implementation:** FFmpeg + OpenCV
**Requirements:**
- Video preprocessing and standardization (_preprocessed.mkv format)
- Frame extraction and streaming pipeline
- Support for multiple video codecs (H.264, H.265, VP9)
- Efficient frame buffering for high FPS processing
- GPU-accelerated video decoding when available
- Batch frame extraction optimization

**Performance Targets:**
- 100+ FPS frame processing (1080p, RTX 3090)
- 60+ FPS frame processing (8K, RTX 3090)
- <20GB VRAM usage during processing

**Dependencies:**
- FFmpeg (external)
- FFprobe (external)
- OpenCV (opencv-python~=4.10.0.84)
- numpy

---

### 2. YOLO Object Detection System

**Priority:** P0
**Current Implementation:** Ultralytics YOLO 8.3.78
**Requirements:**

**Model Management:**
- Support multiple model formats:
  - .pt (PyTorch) - Base format, CUDA/ROCm required
  - .onnx (ONNX Runtime) - CPU-optimized
  - .engine (TensorRT) - NVIDIA GPU optimized, significant performance gains
  - .mlpackage (Core ML) - macOS optimized
- Automatic model selection based on hardware detection
- Dynamic model downloading on first startup
- TensorRT engine compilation (.pt → .engine) for NVIDIA GPUs
- Model versioning and update system

**Detection Classes:**
- Penis detection and tracking
- Hand detection and tracking
- Mouth detection and tracking
- Additional body part detection as needed

**Performance:**
- Stage 1: YOLO detection frame-by-frame
- Raw detection data caching (.msgpack format)
- Reusable detection cache to avoid re-processing
- FP16 optimization for TensorRT models

**GPU Support:**
- NVIDIA 20xx, 30xx, 40xx series (CUDA 12.8)
- NVIDIA 50xx series (special requirements)
- AMD GPUs (ROCm, Linux only)
- CPU fallback mode
- Apple Silicon (Core ML)

**Dependencies:**
- ultralytics==8.3.78
- torch (CUDA/ROCm/CPU variants)
- tensorrt (for .engine models)
- onnxruntime (for .onnx models)

---

### 3. Motion Tracking System

**Priority:** P0
**Current Implementation:** Modular tracker discovery system
**Requirements:**

**Architecture:**
- Dynamic tracker discovery and registration system
- Modular tracker plugin architecture
- Category-based organization (Live, Live+Intervention, Offline, Community)
- CLI alias mapping for batch processing
- Tracker metadata system (name, description, capabilities)

**Core Tracker Categories:**

#### Live Trackers (Real-time Processing)
- No user intervention required
- Batch processing compatible
- Real-time analysis capability
- Examples: Oscillation Detector, YOLO ROI Tracker, Relative Distance Tracker

#### Live + Intervention Trackers
- Require user setup (ROI selection, calibration)
- Real-time processing after setup
- Not batch-compatible
- Examples: User ROI Tracker, DOT Marker

#### Offline Trackers (Multi-stage Processing)
- Stage 1: YOLO detection (object detection)
- Stage 2: Tracking and segmentation (action identification)
- Stage 3: Funscript generation (motion analysis)
- Higher accuracy than live trackers
- Batch processing compatible
- Examples: Contact Analysis (2-Stage), Mixed Processing (3-Stage), Optical Flow Analysis (3-Stage)

**Key Tracker Implementations to Preserve:**

1. **Enhanced Axis Projection Tracker** (P0)
   - Production-grade motion tracking
   - Multi-scale analysis
   - Temporal coherence
   - Adaptive thresholding
   - Current best performer - must be improved upon

2. **Oscillation Detector** (P0)
   - Two modes: Current (experimental) and Legacy (proven)
   - Hybrid approach combining timing precision and amplification
   - Cohesion analysis and signal conditioning

3. **Mixed Processing 3-Stage** (P0)
   - Hybrid approach using Stage 2 signals
   - Selective live ROI tracking for BJ/HJ chapters
   - Best accuracy for complex scenes

**Tracking Capabilities:**
- Single-axis (up/down) tracking
- Dual-axis tracking (up/down + roll/twist)
- VR video support (SBS - Side by Side)
  - Fisheye projection (180°)
  - Equirectangular projection (180°)
  - Automatic FOV detection from filename (_FISHEYE190, _MKX200, _LR_180)
- 2D POV video support
- Automatic video format detection
- Manual format override in UI

**Performance Requirements:**
- Must exceed current Enhanced Axis Projection Tracker performance
- Target: 100+ FPS tracking (1080p)
- ByteTrack as baseline implementation
- Advanced tracker: ByteTrack + OpticalFlow + Kalman + ReID

---

### 4. Funscript Generation

**Priority:** P0
**Requirements:**

**Output Formats:**
- .funscript (primary axis - up/down motion)
- .roll.funscript (secondary axis - roll/twist motion)
- .fgp (FunGen Project file - settings, chapters, paths)
- _t1_raw.funscript (raw unprocessed output)

**Generation Pipeline:**
1. Tracking data collection
2. Motion analysis and temporal smoothing
3. Position value calculation (0-100 range)
4. Timestamp synchronization
5. Raw funscript output
6. Post-processing filters (optional)

**Data Caching:**
- .msgpack (Stage 1: Raw YOLO detections)
- _stage2_overlay.msgpack (Stage 2: Tracking and segmentation)
- Cache reuse for iterative refinement
- Debug visualization support

---

### 5. Filter Plugin System

**Priority:** P0
**Current Implementation:** Modular filter pipeline
**Requirements:**

**Core Filters (Must Preserve):**

1. **Ultimate Autotune** (P0)
   - 7-stage enhancement pipeline
   - Comprehensive motion optimization
   - Default post-processing filter
   - Can be disabled via CLI (--no-autotune)

2. **Speed Limiter** (P0)
   - Hardware device compatibility
   - Speed constraint enforcement
   - Vibration generation for transitions

3. **Smooth (Savitzky-Golay)** (P0)
   - Polynomial smoothing filter
   - Window size and order parameters
   - Noise reduction without losing peaks

4. **Autotune SG** (P1)
   - Automatic parameter optimization
   - Finds optimal Savitzky-Golay settings

5. **Simplify (RDP)** (P1)
   - Ramer-Douglas-Peucker algorithm
   - Removes redundant points
   - File size reduction

6. **Keyframes** (P1)
   - Extracts significant peaks and valleys
   - Simplifies to essential movements

7. **Resample** (P1)
   - Regular interval resampling
   - Preserves peak timing

8. **Amplify** (P2)
   - Amplifies/reduces position values
   - Center point adjustment

9. **Invert** (P2)
   - Position value inversion (0↔100)

10. **Clamp** (P2)
    - Clamps positions to specific value

11. **Threshold Clamp** (P2)
    - Binary clamping based on thresholds

**Filter Pipeline:**
- Chainable filter system
- Filter order matters
- Parameter configuration per filter
- CLI support for batch filtering (--funscript-mode --filter)

---

### 6. GUI Application

**Priority:** P0
**Current Implementation:** tkinter + custom components
**Requirements:**

**Core UI Framework:**
- tkinter as base framework
- sv_ttk theme for modern appearance
- Custom GPU-accelerated rendering for video display
- Automatic DPI scaling detection and adjustment
- Cross-platform support (Windows, macOS, Linux)

**Main UI Components:**

1. **Splash Screen** (P0)
   - Startup loading screen
   - Initialization progress display
   - Git version information display

2. **Video Display** (P0)
   - Real-time video playback
   - Frame-accurate seeking
   - Overlay visualization (detections, tracking data)
   - VR video display modes (SBS split view)
   - Fullscreen mode support
   - GPU-accelerated rendering (OpenGL via moderngl)

3. **Control Panel** (P0)
   - Video loading and management
   - Tracker selection dropdown
   - Processing mode selection
   - Start/Stop/Pause controls
   - Progress bars and status display
   - Settings configuration

4. **Video Navigation** (P0)
   - Timeline scrubber
   - Frame-by-frame navigation
   - Chapter markers
   - Keyboard shortcuts (Spacebar, Arrow keys, etc.)
   - Playback speed control

5. **Info Graphs** (P1)
   - Real-time tracking visualization
   - Funscript position graph
   - Speed/acceleration graphs
   - Detection confidence visualization

6. **Dynamic Tracker UI** (P1)
   - Tracker-specific parameter controls
   - Real-time parameter adjustment
   - Preset management

7. **Device Control** (P2)
   - Hardware device integration
   - Real-time funscript playback to device
   - Connection management

**Real-time Progress Visualization:**
- Agent progress tracking (NEW requirement)
- Multi-agent status display
- Task completion indicators
- Error/warning notifications
- FPS counter display
- VRAM usage monitor

**Themes:**
- Multiple theme support
- Light/Dark mode toggle
- Custom color schemes
- High-DPI display support

**Dependencies:**
- tkinter (Python standard library)
- imgui (for advanced UI elements)
- glfw~=2.8.0
- pyopengl~=3.1.7
- moderngl~=5.11.1
- pillow~=11.1.0
- imageio~=2.36.1

---

### 7. CLI Mode

**Priority:** P0
**Current Implementation:** Argparse-based
**Requirements:**

**Command Structure:**
```bash
python main.py [input_path] [options]
```

**Core Arguments:**
- `input_path` - Video file or folder path (required for CLI mode)
- `--mode` - Processing mode selection (dynamic from tracker discovery)
- `--od-mode` - Oscillation detector mode (current/legacy)
- `--overwrite` - Force reprocessing of existing funscripts
- `--no-autotune` - Disable Ultimate Autotune post-processing
- `--no-copy` - Don't copy funscript next to video file
- `--generate-roll` - Generate secondary axis (.roll.funscript)
- `--recursive` / `-r` - Recursive folder processing

**Funscript Filtering Mode:**
- `--funscript-mode` - Process existing funscripts instead of videos
- `--filter` - Filter to apply (ultimate-autotune, rdp-simplify, savgol-filter, etc.)

**Batch Processing:**
- Folder recursion support
- Skip files with existing funscripts (unless --overwrite)
- Parallel processing capability (multiple instances)
- Progress logging to console

**Performance:**
- Tested: 160-190 FPS with parallel processing (3-6 instances)
- 20-30 minutes for 8K VR videos (complete pipeline)
- Hardware-dependent optimization

---

### 8. Cross-Platform Compatibility

**Priority:** P0
**Requirements:**

**Development Platform:**
- Raspberry Pi 4/5 (ARM64)
- CPU-only mode
- Logic testing and unit tests
- Conditional GPU imports (torch.cuda.is_available())

**Production Platform:**
- AMD Ryzen 2990 + RTX 3090
- CUDA 12.8 support
- 24GB VRAM
- 48GB RAM
- TensorRT optimization

**Platform-Specific Features:**
- GPU detection and capability reporting
- Automatic PyTorch variant installation (CUDA/ROCm/CPU)
- Platform-specific multiprocessing (spawn method)
- Conditional GPU-accelerated video rendering
- DPI scaling for different displays

**GPU Support Matrix:**
- NVIDIA 20xx/30xx/40xx: CUDA 12.8 + TensorRT
- NVIDIA 50xx: Special requirements (cuda.50series.requirements.txt)
- AMD (Linux only): ROCm support
- Apple Silicon: Core ML support
- CPU fallback: All platforms

---

## P1 - High Priority Features

### 9. Video Format Detection

**Priority:** P1
**Requirements:**

**Automatic Detection:**
- VR vs 2D detection
- SBS (Side by Side) format detection
- Projection type detection:
  - Fisheye (180°)
  - Equirectangular (180°)
- FOV detection from filename patterns:
  - _FISHEYE190
  - _MKX200
  - _LR_180
  - Other standard patterns

**Manual Override:**
- UI settings for manual format selection
- Per-video format configuration
- Format validation and warnings

---

### 10. Chapter/Segmentation System

**Priority:** P1
**Requirements:**

**Stage 2 Processing:**
- Automatic scene segmentation
- Action chapter identification:
  - BJ (Blowjob) chapters
  - HJ (Handjob) chapters
  - Contact vs non-contact detection
- Temporal segmentation data
- Chapter metadata storage

**Use Cases:**
- Mixed processing mode (selective tracking per chapter)
- Visualization in UI timeline
- Jump-to-chapter navigation
- Per-chapter filter application

---

### 11. Project File System (.fgp)

**Priority:** P1
**Requirements:**

**FunGen Project Format:**
- Settings persistence
- Chapter data storage
- File path references
- Processing metadata
- Version information

**Use Cases:**
- Resume processing from checkpoint
- Iterative refinement workflow
- Settings sharing/templates
- Debug and analysis

---

### 12. Update System

**Priority:** P1
**Current Implementation:** GitHub API integration
**Requirements:**

**Features:**
- Version selection from commits
- Changelog display
- Update notifications
- Branch switching (main)
- GitHub token support for rate limit avoidance

**Token System:**
- Optional GitHub Personal Access Token
- 5,000 requests/hour (vs 60 without token)
- Scopes: public_repo, read:user
- Local storage (github_token.ini)
- Token validation and testing

---

### 13. TensorRT Model Compilation

**Priority:** P1
**Requirements:**

**Compiler Interface:**
- GUI window for compilation
- .pt → .engine conversion
- Batch size optimization
- FP16 precision mode
- Progress tracking
- Generated model validation

**CLI Support:**
- "Generate TensorRT.bat" script
- Automated compilation workflow

---

### 14. File Management

**Priority:** P1
**Requirements:**

**Output Directory Management:**
- Configurable output folder
- Subfolder per video
- File naming conventions
- Generated file tracking

**Generated File Manager:**
- UI window for file management
- File deletion with trash support (send2trash)
- File organization tools
- Storage usage display

**File Types:**
- _preprocessed.mkv
- .msgpack (Stage 1 cache)
- _stage2_overlay.msgpack
- _t1_raw.funscript
- .funscript (final output)
- .roll.funscript (secondary axis)
- .fgp (project file)

---

### 15. Keyboard Shortcuts

**Priority:** P1
**Requirements:**

**Navigation:**
- Spacebar: Play/Pause
- Arrow Keys: Frame navigation (Left/Right), Speed control (Up/Down)
- Home/End: Jump to start/end
- Number keys: Jump to percentage

**Processing:**
- Ctrl+O: Open video
- Ctrl+S: Save funscript
- Ctrl+G: Generate funscript
- Ctrl+P: Process video

**UI:**
- F11: Fullscreen toggle
- Ctrl+K: Keyboard shortcuts dialog
- Ctrl+Q: Quit application

**Customization:**
- User-configurable shortcuts
- Shortcut conflict detection
- Help dialog with all shortcuts

---

### 16. Dependency Management

**Priority:** P1
**Current Implementation:** Automatic installation system
**Requirements:**

**Bootstrap Dependencies:**
- packaging
- requests
- tqdm
- send2trash

**Automatic Installation:**
- Check on startup
- Install missing dependencies
- Requirements file support:
  - core.requirements.txt (base dependencies)
  - cuda.requirements.txt (NVIDIA 20xx-40xx)
  - cuda.50series.requirements.txt (NVIDIA 50xx)
  - cpu.requirements.txt (CPU-only)
  - rocm.requirements.txt (AMD Linux)

**Installer Scripts:**
- install.bat (Windows)
- install.sh (Linux/macOS)
- install.py (Python installer)
- Automatic Miniconda installation
- FFmpeg installation
- Git installation

---

## P2 - Nice-to-Have Features

### 17. Visualization and Debug Tools

**Priority:** P2
**Requirements:**

**Overlay Visualization:**
- YOLO detection bounding boxes
- Tracking IDs and trajectories
- ROI rectangles
- Optical flow vectors
- Confidence scores

**Debug Overlays:**
- Stage 2 segmentation data
- Chapter boundaries
- Motion heatmaps
- Performance metrics (FPS, latency)

**Export Options:**
- Debug video with overlays
- Frame-by-frame image export
- Tracking data CSV export

---

### 18. Device Integration

**Priority:** P2
**Requirements:**

**Device Control UI:**
- Device connection management
- Real-time funscript playback
- Connection status display
- Device list and selection

**Supported Protocols:**
- Buttplug.io integration
- Direct serial communication
- Network-based devices

---

### 19. Calibration System

**Priority:** P2
**Requirements:**

**Video Calibration:**
- Offset adjustment (time sync)
- Position range calibration
- Speed calibration

**Device Calibration:**
- Position mapping
- Speed limits
- Device-specific profiles

---

### 20. Energy Saver Mode

**Priority:** P2
**Requirements:**

**Power Management:**
- Reduce processing during idle
- GPU power state management
- Background processing throttling

---

### 21. Advanced Filters

**Priority:** P2
**Requirements:**

**Anti-Jerk Filter:**
- Reduce sudden movements
- Smooth acceleration curves

**Beat Marker:**
- Visual brightness detection
- Audio beat detection
- Metronome generation

---

## Performance Requirements

### Frame Processing Performance

| Resolution | Target FPS | Hardware | VRAM Usage |
|------------|------------|----------|------------|
| 1080p | 100+ FPS | RTX 3090 | <20GB |
| 4K | 80+ FPS | RTX 3090 | <20GB |
| 8K | 60+ FPS | RTX 3090 | <20GB |

**Current Baseline:**
- Single process: 60-110 FPS (8K VR)
- Parallel (3-6 instances): 160-190 FPS aggregate
- Processing time: 20-30 minutes per 8K VR video

### Memory Requirements

**Production (RTX 3090):**
- VRAM: 24GB available, target <20GB usage
- RAM: 48GB available
- Model loading: 3-6 instances for parallel processing

**Development (Raspberry Pi):**
- CPU-only mode
- Memory-efficient algorithms
- Conditional GPU code

### Code Quality Requirements

- **Test Coverage:** 80%+ unit test coverage
- **Type Hints:** Python 3.11+ with mandatory type hints
- **Documentation:** Google-style docstrings
- **Code Formatting:** Black (line-length=100), isort, mypy
- **No Code Duplication:** Zero duplicate code across modules

---

## FunGen Strengths to Preserve

### 1. Modular Architecture

**Why it's good:**
- Tracker discovery system allows runtime plugin addition
- Filter pipeline is chainable and extensible
- Separation of concerns (detection, tracking, generation, filtering)

**Preserve:**
- Plugin architecture pattern
- Dynamic discovery system
- Registry-based module loading

### 2. Multi-Stage Processing Pipeline

**Why it's good:**
- Stage 1 (YOLO) results are cached and reusable
- Stage 2 (tracking) can be refined without re-detection
- Iterative improvement workflow

**Preserve:**
- 3-stage architecture pattern
- Caching system (.msgpack format)
- Checkpoint/resume capability

### 3. GUI Design Philosophy

**Why it's good:**
- Real-time video visualization with overlays
- Integrated processing and preview
- Keyboard-centric workflow
- Professional appearance (sv_ttk theme)

**Preserve:**
- Single-window integrated interface
- Real-time feedback and visualization
- Keyboard shortcut system
- GPU-accelerated video rendering

### 4. Batch Processing Capability

**Why it's good:**
- CLI mode for automation
- Recursive folder processing
- Skip existing files optimization
- Parallel instance support

**Preserve:**
- CLI argument structure
- Batch processing logic
- Cache reuse strategy

### 5. Cross-Platform Model Support

**Why it's good:**
- .pt/.onnx/.engine/.mlpackage support
- Automatic model selection
- Hardware-optimized inference

**Preserve:**
- Multi-format model system
- Automatic hardware detection
- TensorRT compilation workflow

---

## Technical Dependencies

### Core Dependencies (core.requirements.txt)

```
numpy                      # Numerical computing
imgui                      # Advanced UI elements
ultralytics==8.3.78        # YOLO object detection
glfw~=2.8.0                # OpenGL windowing
pyopengl~=3.1.7            # OpenGL bindings
moderngl~=5.11.1           # Modern OpenGL
imageio~=2.36.1            # Image I/O
tqdm~=4.67.1               # Progress bars
colorama~=0.4.6            # Colored terminal output
opencv-python~=4.10.0.84   # Computer vision
scipy~=1.15.1              # Scientific computing
scikit-learn>=1.0.0        # Machine learning utilities
simplification~=0.7.13     # RDP algorithm
msgpack~=1.1.0             # Binary serialization
pillow~=11.1.0             # Image processing
orjson~=3.10.15            # Fast JSON
send2trash~=1.8.3          # Safe file deletion
aiosqlite                  # Async SQLite
```

### GPU Dependencies

**NVIDIA CUDA (20xx-40xx series):**
```
torch==2.8.0+cu128
torchvision==0.23.0+cu128
torchaudio==2.8.0+cu128
tensorrt                   # TensorRT optimization
sympy>=1.13.3
```

**NVIDIA 50xx Series:** (cuda.50series.requirements.txt)

**AMD ROCm (Linux only):** (rocm.requirements.txt)

**CPU Fallback:** (cpu.requirements.txt)

### External Tools

- **FFmpeg** - Video processing (required)
- **FFprobe** - Video metadata extraction (required)
- **Git** - Version control and updates (optional)

---

## Architecture Insights from FunGen

### 1. Application Logic Structure

**app_logic.py (125KB):**
- Core ApplicationLogic class
- Coordinates all subsystems
- State management
- Event handling orchestration

**app_stage_processor.py (139KB):**
- Multi-stage processing coordination
- Stage 1: YOLO detection
- Stage 2: Tracking and segmentation
- Stage 3: Funscript generation
- Progress reporting and cancellation

**app_funscript_processor.py (89KB):**
- Funscript generation logic
- Filter pipeline execution
- Post-processing coordination

**app_file_manager.py (45KB):**
- Output directory management
- File naming and organization
- Cache file handling

### 2. UI Component Architecture

**app_gui.py (131KB):**
- Main GUI class
- Window management
- Component coordination

**video_display_ui.py (132KB):**
- GPU-accelerated video rendering
- Overlay system
- VR projection handling

**control_panel_ui.py (252KB):**
- Main control interface
- Settings management
- Processing controls

**video_navigation_ui.py (94KB):**
- Timeline scrubber
- Chapter navigation
- Playback controls

**info_graphs_ui.py (73KB):**
- Real-time graph plotting
- Performance metrics display

### 3. Tracker System Architecture

**Dynamic Discovery System:**
- tracker_registry - Central tracker registration
- TrackerMetadata - Tracker description and capabilities
- TrackerCategory - Live/Offline/Community classification
- CLI alias mapping for batch mode

**Tracker Interface:**
- Standardized tracker API
- Stage definitions for multi-stage trackers
- Property system for capabilities
- Batch compatibility flags

### 4. Filter System Architecture

**Modular Filter Pipeline:**
- Filter registry and discovery
- Chainable filter execution
- Parameter configuration
- CLI filter support

---

## Risk Analysis

### High-Risk Areas

1. **Performance Target Achievement**
   - Risk: May not reach 100+ FPS on all hardware
   - Mitigation: Incremental optimization, benchmark suite, fallback modes

2. **Tracking Algorithm Improvement**
   - Risk: New trackers may not exceed Enhanced Axis Projection performance
   - Mitigation: Preserve existing trackers, A/B testing framework, quality metrics

3. **Cross-Platform Compatibility**
   - Risk: GPU code may not work on Raspberry Pi
   - Mitigation: Conditional imports, CPU fallback, extensive testing

### Medium-Risk Areas

1. **YOLO Model Compatibility**
   - Risk: Future Ultralytics updates may break compatibility
   - Mitigation: Pin version (8.3.78), testing before upgrades

2. **TensorRT Optimization**
   - Risk: TensorRT engines are GPU-specific
   - Mitigation: Per-GPU compilation, model format fallback

3. **UI Complexity**
   - Risk: Real-time agent visualization may impact performance
   - Mitigation: Async updates, update throttling, optional display

---

## Success Criteria

### Performance Metrics
- [ ] 100+ FPS processing at 1080p (RTX 3090)
- [ ] 60+ FPS processing at 8K (RTX 3090)
- [ ] <20GB VRAM usage during processing
- [ ] 80%+ unit test coverage

### Functionality Metrics
- [ ] All P0 features implemented and working
- [ ] CLI mode supports batch processing
- [ ] GUI displays real-time agent progress
- [ ] Tracking quality meets or exceeds Enhanced Axis Projection

### Quality Metrics
- [ ] Zero code duplication
- [ ] All code has type hints
- [ ] Google-style docstrings on all functions
- [ ] Black + isort + mypy pass on all files

### Platform Metrics
- [ ] Works on Raspberry Pi (CPU mode)
- [ ] Works on RTX 3090 (GPU mode)
- [ ] Conditional GPU imports working
- [ ] Cross-platform tests passing

---

## Open Questions

1. **ByteTrack Implementation:** Should we use existing ByteTrack libraries or implement from scratch?
2. **Agent Communication:** Should agent progress updates use files, sockets, or shared memory?
3. **UI Framework:** Keep tkinter or migrate to Qt/wxPython for better cross-platform support?
4. **Test Strategy:** Unit tests on Pi, integration tests on RTX 3090, or both on both?

---

## Appendix A: CLI Mode Examples

```bash
# GUI mode
python main.py

# Single video processing
python main.py "/path/to/video.mp4"

# Batch processing with specific mode
python main.py "/path/to/folder" --mode 3-stage --recursive --overwrite

# Funscript filtering
python main.py "/path/to/video.funscript" --funscript-mode --filter ultimate-autotune

# Performance mode (no autotune, no copy)
python main.py "/path/to/video.mp4" --mode oscillation --no-autotune --no-copy

# Dual-axis generation
python main.py "/path/to/video.mp4" --mode 3-stage --generate-roll
```

---

## Appendix B: Tracker Categories

### Live Trackers
- Hybrid Intelligence Tracker
- Oscillation Detector (Experimental 2)
- Oscillation Detector (Legacy)
- Relative Distance Tracker
- YOLO ROI Tracker

### Live + Intervention Trackers
- User ROI Tracker
- DOT Marker (Manual Point)

### Offline Trackers
- Contact Analysis (2-Stage)
- Mixed Processing (3-Stage)
- Optical Flow Analysis (3-Stage)

### Experimental Trackers
- Enhanced Axis Projection Tracker (current best)
- Working Axis Projection Tracker
- Beat Marker (Visual/Audio)

### Community Trackers
- Community Example Tracker (template)

---

## Appendix C: Filter Types

### Motion Enhancement
- Ultimate Autotune (7-stage pipeline)
- Autotune SG (automatic Savitzky-Golay)
- Smooth (SG) (Savitzky-Golay filter)

### Simplification
- Simplify (RDP) (Ramer-Douglas-Peucker)
- Keyframes (peak/valley extraction)
- Resample (regular interval)

### Device Compatibility
- Speed Limiter (hardware constraints + vibrations)

### Transformation
- Amplify (scale around center)
- Invert (0↔100)
- Clamp (specific value)
- Threshold Clamp (binary clamping)

---

**End of Requirements Document**
