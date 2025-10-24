# FunGen Rewrite - Multi-Agent Grand Plan

## Mission
Rewrite https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator with:
- 100+ FPS tracking on RTX 3090 (24GB VRAM, current: ~60-110 FPS)
- Modern UI with real-time agent progress visualization
- Cross-platform: Raspberry Pi (dev) ↔ RTX 3090 (prod)
- Better tracking algorithms than Enhanced Axis Projection
- Clean modular architecture with zero code duplication

## Agent Team (15 Specialists)

### Planning (Minutes 0-5)
- **project-architect**: Master coordinator, creates architecture
- **requirements-analyst**: Extracts FunGen features, prioritizes

### Core Development (Minutes 5-25, parallel execution)
- **video-specialist**: Video processing pipeline (FFmpeg, OpenCV, streaming)
- **ml-specialist**: YOLO model manager (TensorRT, FP16 optimization)
- **tracker-dev-1**: ByteTrack implementation (fast baseline)
- **tracker-dev-2**: Advanced tracker (ByteTrack+OpticalFlow+Kalman+ReID)
- **ui-architect**: Core UI structure (tkinter, sv_ttk theme)
- **ui-enhancer**: UI polish (animations, tooltips, shortcuts, themes)
- **cross-platform-dev**: Pi/RTX3090 compatibility layer

### Quality Assurance (Minutes 15-30, continuous)
- **test-engineer-1**: Unit tests (80%+ coverage, Pi CPU testing)
- **test-engineer-2**: Integration tests, benchmarks
- **integration-master**: Combines work, removes duplicates, final assembly
- **code-quality**: Black, isort, mypy, docstrings

### Debug Loop (Minutes 15-30, on-demand)
- **gpu-debugger**: CUDA errors → sends fixes → waits for agent response → re-tests
- **python-debugger**: General bugs → same fix loop → notifies affected agents

## Inter-Agent Communication Protocol

**Shared Files (all agents read):**
- CLAUDE.md (this file) - grand plan
- docs/architecture.md - module interfaces
- docs/agent_assignments.json - work distribution
- progress/*.json - individual agent progress

**Bug Fix Workflow:**
1. Test agent finds bug → reports to debugger (gpu/python)
2. Debugger analyzes → identifies responsible agent
3. Debugger writes fix → sends to communication/{agent}_inbox.json
4. Responsible agent reads inbox → applies fix → tests → sends to _outbox.json
5. Debugger reads outbox → applies updated code → re-tests
6. If still broken: loop back to step 3 with more details
7. If fixed: notify affected agents via communication/{agent}_inbox.json
8. Affected agents check integration → confirm no conflicts

**Progress Tracking:**
Every agent updates progress/{agent}.json every 2 minutes with:
```json
{
  "agent": "video-specialist",
  "progress": 65,
  "status": "working",
  "current_task": "Implementing batch frame extraction",
  "timestamp": "2025-10-24T19:45:00Z"
}
```

## Target Platform
- **Dev**: Raspberry Pi 4/5 (ARM64, CPU-only, test logic)
- **Prod**: AMD Ryzen 2990 + RTX 3090 (24GB VRAM, 48GB RAM)

## Performance Targets
- 100+ FPS (1080p video, RTX 3090, current best: ~110)
- 60+ FPS (8K video, RTX 3090)
- <20GB VRAM usage
- 80%+ test coverage
- Works on Pi (CPU mode) for development

## Code Standards
- Python 3.11+, type hints mandatory
- Google-style docstrings
- Black (line-length=100), isort, mypy
- No duplicate code across modules
- Conditional GPU imports (torch.cuda.is_available())

## FunGen Repository
https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator

## Key Features to Preserve
- Funscript generation (.funscript format)
- Multiple tracking algorithms
- VR video support (SBS)
- Batch processing
- GUI with progress tracking
- CLI mode for automation
