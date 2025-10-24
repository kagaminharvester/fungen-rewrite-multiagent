# FunGen Rewrite - Implementation Roadmap

**Created:** 2025-10-24
**Author:** project-architect
**Duration:** 30 minutes (5 phases)

---

## Overview

This roadmap outlines the 30-minute implementation plan for the FunGen rewrite, coordinating 15 specialist agents working in parallel phases.

---

## Phase 1: Planning (Minutes 0-5)

**Objective:** Establish architecture and requirements

**Agents:** 2 (sequential)

| Agent | Tasks | Deliverables | Status |
|-------|-------|--------------|--------|
| project-architect | Design architecture, research optimizations | architecture.md, agent_assignments.json | IN PROGRESS |
| requirements-analyst | Extract FunGen features, prioritize | requirements.md, feature_priority.json | PENDING |

**Key Milestones:**
- ✓ Architecture document complete (12 sections, 1000+ lines)
- ✓ Agent assignments defined (15 agents, 100+ tasks)
- ☐ Requirements extracted from FunGen
- ☐ Feature priorities established

---

## Phase 2: Core Development - Video & ML (Minutes 5-15)

**Objective:** Build video pipeline and model management

**Agents:** 3 (parallel)

### video-specialist

**Priority:** CRITICAL
**Duration:** 10 minutes
**Dependencies:** project-architect

**Tasks:**
1. Implement `VideoPipeline` class with FFmpeg integration (4 min)
2. Add PyNvVideoCodec 2.0 support for RTX 3090 (3 min)
3. Implement circular frame buffer (max 60 frames) (2 min)
4. Add VR format detection (SBS Fisheye/Equirectangular) (2 min)
5. Create `BatchProcessor` for multi-video queues (3 min)
6. Write unit tests (2 min)

**Deliverables:**
- `/home/pi/elo_elo_320/core/video_pipeline.py`
- `/home/pi/elo_elo_320/core/batch_processor.py`
- `/home/pi/elo_elo_320/tests/unit/test_video_pipeline.py`

**Performance Targets:**
- 200+ FPS decode (1080p)
- 60+ FPS decode (8K)
- <500MB memory usage

### ml-specialist

**Priority:** CRITICAL
**Duration:** 10 minutes
**Dependencies:** project-architect

**Tasks:**
1. Implement `ModelManager` class with auto-detection (3 min)
2. Add TensorRT FP16 optimization (3 min)
3. Implement batch inference (`predict_batch`) (2 min)
4. Add VRAM monitoring (`get_vram_usage`) (2 min)
5. Implement multi-GPU support with MPS (2 min)
6. Create `Config` class with profile support (2 min)
7. Write unit tests with mock models (2 min)

**Deliverables:**
- `/home/pi/elo_elo_320/core/model_manager.py`
- `/home/pi/elo_elo_320/core/config.py`
- `/home/pi/elo_elo_320/tests/unit/test_model_manager.py`

**Performance Targets:**
- 100+ FPS inference (1080p, FP16)
- <20GB VRAM usage
- <5 seconds model load time

### cross-platform-dev

**Priority:** HIGH
**Duration:** 10 minutes
**Dependencies:** project-architect, ml-specialist (partial)

**Tasks:**
1. Implement hardware detection (`detect_hardware`) (3 min)
2. Add conditional GPU imports with fallbacks (2 min)
3. Create `PerformanceMonitor` class (2 min)
4. Implement funscript I/O utilities (2 min)
5. Create structured logger (2 min)
6. Add CPU fallback mode for Pi (3 min)
7. Write unit tests for hardware detection (2 min)

**Deliverables:**
- `/home/pi/elo_elo_320/utils/hardware.py`
- `/home/pi/elo_elo_320/utils/metrics.py`
- `/home/pi/elo_elo_320/utils/funscript.py`
- `/home/pi/elo_elo_320/utils/logger.py`
- `/home/pi/elo_elo_320/tests/unit/test_hardware.py`

**Performance Targets:**
- Pi: 5+ FPS (CPU mode)
- RTX 3090: 100+ FPS (GPU mode)

---

## Phase 3: Core Development - Tracking (Minutes 10-20)

**Objective:** Implement tracking algorithms

**Agents:** 2 (parallel, TD2 depends on TD1)

### tracker-dev-1

**Priority:** HIGH
**Duration:** 10 minutes
**Dependencies:** project-architect, ml-specialist

**Tasks:**
1. Create `BaseTracker` abstract class (2 min)
2. Implement `ByteTrack` tracker (fast baseline) (4 min)
3. Add detection-to-track association logic (2 min)
4. Implement Kalman filter for motion prediction (3 min)
5. Add funscript conversion logic (2 min)
6. Write unit tests (2 min)

**Deliverables:**
- `/home/pi/elo_elo_320/trackers/base_tracker.py`
- `/home/pi/elo_elo_320/trackers/bytetrack.py`
- `/home/pi/elo_elo_320/tests/unit/test_bytetrack.py`

**Performance Targets:**
- <50ms latency per frame
- 70%+ MOTA (tracking accuracy)

### tracker-dev-2

**Priority:** HIGH
**Duration:** 10 minutes
**Dependencies:** project-architect, tracker-dev-1

**Tasks:**
1. Implement `BoT-SORT` tracker (accuracy-focused) (4 min)
2. Add ResNet50 ReID feature extraction (3 min)
3. Implement `HybridTracker` (production-grade) (5 min)
4. Add CUDA optical flow integration (3 min)
5. Implement adaptive algorithm selection (2 min)
6. Write unit tests (2 min)

**Deliverables:**
- `/home/pi/elo_elo_320/trackers/botsort.py`
- `/home/pi/elo_elo_320/trackers/hybrid_tracker.py`
- `/home/pi/elo_elo_320/tests/unit/test_hybrid_tracker.py`

**Performance Targets:**
- 80+ FPS (1080p, HybridTracker)
- 85%+ MOTA (tracking accuracy)
- 90%+ ReID accuracy

---

## Phase 4: Core Development - UI (Minutes 10-25)

**Objective:** Build user interface with agent dashboard

**Agents:** 2 (parallel, UE depends on UA)

### ui-architect

**Priority:** HIGH
**Duration:** 10 minutes
**Dependencies:** project-architect

**Tasks:**
1. Create `MainWindow` class with tkinter (3 min)
2. Implement `AgentDashboard` widget (unique feature) (4 min)
3. Add real-time FPS/VRAM display (2 min)
4. Create `SettingsPanel` with sv_ttk theme (3 min)
5. Implement threading for non-blocking UI (3 min)
6. Write UI unit tests (2 min)

**Deliverables:**
- `/home/pi/elo_elo_320/ui/main_window.py`
- `/home/pi/elo_elo_320/ui/agent_dashboard.py`
- `/home/pi/elo_elo_320/ui/settings_panel.py`
- `/home/pi/elo_elo_320/tests/unit/test_ui.py`

**Performance Targets:**
- <100ms UI update latency
- 2-second agent dashboard refresh rate

### ui-enhancer

**Priority:** MEDIUM
**Duration:** 10 minutes
**Dependencies:** ui-architect

**Tasks:**
1. Add tooltips to all buttons/fields (2 min)
2. Implement keyboard shortcuts (2 min)
3. Create reusable UI components (3 min)
4. Add light/dark theme support (3 min)
5. Implement smooth animations (2 min)
6. Create UI documentation (2 min)

**Deliverables:**
- `/home/pi/elo_elo_320/ui/components/`
- `/home/pi/elo_elo_320/ui/themes.py`
- `/home/pi/elo_elo_320/docs/ui_guide.md`

---

## Phase 5: Quality Assurance (Minutes 15-30)

**Objective:** Test, integrate, and polish

**Agents:** 4 (parallel)

### test-engineer-1

**Priority:** HIGH
**Duration:** 15 minutes
**Dependencies:** video-specialist, ml-specialist, tracker-dev-1

**Tasks:**
1. Set up pytest infrastructure (2 min)
2. Write unit tests for core/ modules (5 min)
3. Write unit tests for trackers/ modules (4 min)
4. Write unit tests for utils/ modules (3 min)
5. Add CPU-only tests for Pi compatibility (3 min)
6. Generate coverage report (1 min)

**Deliverables:**
- `/home/pi/elo_elo_320/tests/unit/` (complete)
- `/home/pi/elo_elo_320/tests/conftest.py`
- `/home/pi/elo_elo_320/docs/test_coverage_report.html`

**Success Criteria:**
- 80%+ test coverage
- All tests passing

### test-engineer-2

**Priority:** HIGH
**Duration:** 15 minutes
**Dependencies:** test-engineer-1, integration-master

**Tasks:**
1. Create end-to-end integration tests (4 min)
2. Add error handling tests (3 min)
3. Create benchmark suite (4 min)
4. Run benchmarks on RTX 3090 (3 min)
5. Run benchmarks on Pi (CPU mode) (2 min)
6. Document results (2 min)

**Deliverables:**
- `/home/pi/elo_elo_320/tests/integration/`
- `/home/pi/elo_elo_320/tests/benchmarks/`
- `/home/pi/elo_elo_320/docs/benchmark_results.md`

**Success Criteria:**
- 100+ FPS (1080p, RTX 3090)
- 60+ FPS (8K, RTX 3090)
- <20GB VRAM
- Pi functional (CPU mode)

### integration-master

**Priority:** CRITICAL
**Duration:** 10 minutes (starts minute 20)
**Dependencies:** All core development agents

**Tasks:**
1. Review all modules for code duplication (3 min)
2. Remove duplicates, consolidate utilities (3 min)
3. Create `main.py` entry point (3 min)
4. Test integration of all modules (4 min)
5. Fix integration bugs (4 min)
6. Document integration (2 min)

**Deliverables:**
- `/home/pi/elo_elo_320/main.py`
- `/home/pi/elo_elo_320/docs/integration_report.md`

**Success Criteria:**
- Zero code duplication
- All modules integrated
- main.py functional

### code-quality

**Priority:** HIGH
**Duration:** 10 minutes (starts minute 20)
**Dependencies:** integration-master

**Tasks:**
1. Run Black formatter (1 min)
2. Run isort on imports (1 min)
3. Run mypy type checking (3 min)
4. Add Google-style docstrings (5 min)
5. Set up pre-commit hooks (2 min)
6. Create pyproject.toml (1 min)
7. Document code quality metrics (2 min)

**Deliverables:**
- `/home/pi/elo_elo_320/.pre-commit-config.yaml`
- `/home/pi/elo_elo_320/pyproject.toml`
- `/home/pi/elo_elo_320/docs/code_quality_report.md`

**Success Criteria:**
- Zero mypy errors
- All functions documented
- Code formatted

---

## Phase 6: Debug Loop (Minutes 15-30, On-Demand)

**Objective:** Fix bugs found during testing

**Agents:** 2 (on-demand)

### gpu-debugger

**Priority:** HIGH
**Duration:** Variable (on-demand)
**Dependencies:** test-engineer-1, test-engineer-2

**Responsibilities:**
- Monitor test output for CUDA errors (OOM, invalid device, kernel errors)
- Analyze root cause (VRAM, batch size, model size)
- Write fixes, send to affected agents via `communication/{agent}_inbox.json`
- Wait for agent response in `communication/{agent}_outbox.json`
- Re-test, iterate if needed
- Notify affected agents when fixed

**Communication Protocol:**
1. Test agent finds bug → reports to gpu-debugger
2. gpu-debugger analyzes → identifies responsible agent
3. gpu-debugger writes fix → sends to inbox.json
4. Agent applies fix → tests → sends to outbox.json
5. gpu-debugger re-tests → iterates if broken
6. If fixed → notify affected agents

### python-debugger

**Priority:** HIGH
**Duration:** Variable (on-demand)
**Dependencies:** test-engineer-1, test-engineer-2

**Responsibilities:**
- Monitor test output for Python errors (exceptions, logic errors, edge cases)
- Analyze errors, identify responsible agent
- Write fixes, send to affected agents via inbox.json
- Wait for agent response in outbox.json
- Re-test, iterate if needed
- Notify affected agents when fixed

**Communication Protocol:** Same as gpu-debugger

---

## Timeline Visualization

```
Minute 0  ┌──────────────────┐
          │ project-architect│ (Planning)
Minute 5  ├──────────────────┴──────────────────────────────────┐
          │ video-specialist                                     │
          ├──────────────────────────────────────────────────────┤
          │ ml-specialist                                        │
          ├──────────────────────────────────────────────────────┤
          │ cross-platform-dev                                   │
Minute 10 ├──────────────────────────────────────────────────────┤
          │ tracker-dev-1                                        │
          ├──────────────────────────────────────────────────────┤
          │ tracker-dev-2                  (depends on TD1)      │
          ├──────────────────────────────────────────────────────┤
          │ ui-architect                                         │
          ├──────────────────────────────────────────────────────┤
          │ ui-enhancer                    (depends on UA)       │
Minute 15 ├──────────────────────────────────────────────────────┤
          │ test-engineer-1                                      │
          ├──────────────────────────────────────────────────────┤
          │ test-engineer-2                                      │
Minute 20 ├──────────────────────────────────────────────────────┤
          │ integration-master                                   │
          ├──────────────────────────────────────────────────────┤
          │ code-quality                                         │
Minute 25 ├──────────────────────────────────────────────────────┤
          │ Final testing and documentation                      │
Minute 30 └──────────────────────────────────────────────────────┘

Debug agents (gpu-debugger, python-debugger) run continuously from minute 15-30
```

---

## Critical Path

The critical path determines the minimum time to complete the project:

1. **project-architect** (5 min) →
2. **ml-specialist** (10 min) →
3. **tracker-dev-1** (10 min) →
4. **tracker-dev-2** (10 min, parallel with TE1) →
5. **integration-master** (10 min) →
6. **code-quality** (10 min)

**Total Critical Path:** 55 minutes

**With Parallelization:** 30 minutes (agents run concurrently)

---

## Dependencies Graph

```
project-architect
    ├── requirements-analyst
    ├── video-specialist
    ├── ml-specialist
    │   └── cross-platform-dev
    │       └── tracker-dev-1
    │           └── tracker-dev-2
    │               └── test-engineer-1
    │                   └── test-engineer-2
    │                       └── integration-master
    │                           └── code-quality
    └── ui-architect
        └── ui-enhancer

Debug agents (gpu-debugger, python-debugger) monitor all test agents
```

---

## Inter-Agent Communication

### Progress Tracking

Every agent updates `progress/{agent}.json` every 2 minutes:

```json
{
  "agent": "video-specialist",
  "progress": 65,
  "status": "working",
  "current_task": "Implementing batch frame extraction",
  "timestamp": "2025-10-24T19:45:00Z"
}
```

### Bug Fix Workflow

1. Test agent finds bug → reports to debugger (gpu/python)
2. Debugger analyzes → identifies responsible agent
3. Debugger writes fix → sends to `communication/{agent}_inbox.json`
4. Responsible agent reads inbox → applies fix → tests → sends to `_outbox.json`
5. Debugger reads outbox → applies updated code → re-tests
6. If still broken: loop back to step 3 with more details
7. If fixed: notify affected agents via `communication/{agent}_inbox.json`
8. Affected agents check integration → confirm no conflicts

---

## Success Criteria Checklist

### Performance
- [ ] 100+ FPS (1080p, RTX 3090)
- [ ] 60+ FPS (8K, RTX 3090)
- [ ] <20GB VRAM usage
- [ ] 5+ FPS (Pi CPU mode)

### Quality
- [ ] 80%+ test coverage
- [ ] Zero mypy errors
- [ ] All functions have Google-style docstrings
- [ ] Black formatting applied

### Functionality
- [ ] Works on Pi (CPU mode)
- [ ] Works on RTX 3090 (GPU mode)
- [ ] Agent dashboard functional
- [ ] CLI mode functional
- [ ] GUI mode functional

### Architecture
- [ ] Zero code duplication
- [ ] Modular design (can swap trackers)
- [ ] Clear separation of concerns
- [ ] Cross-platform compatibility

---

## Risk Mitigation Plan

| Risk | Probability | Impact | Mitigation | Owner |
|------|-------------|--------|------------|-------|
| TensorRT installation issues | High | High | Provide pre-built .engine models, fallback to ONNX | ml-specialist |
| CUDA OOM on RTX 3090 | Medium | High | Dynamic batch sizing, VRAM monitoring, gpu-debugger | ml-specialist, gpu-debugger |
| Pi too slow for testing | Low | Medium | Mock GPU modules for unit tests | cross-platform-dev |
| Tracker accuracy regression | Medium | High | Benchmark against FunGen, A/B testing | tracker-dev-2, test-engineer-2 |
| Agent coordination overhead | Low | Medium | Use simple JSON files, avoid complexity | project-architect |
| Integration conflicts | Medium | High | Clear API contracts, early integration testing | integration-master |
| Timeline overrun | Medium | High | Prioritize critical path, defer nice-to-have features | project-architect |

---

## Deliverables Summary

By the end of 30 minutes, the following deliverables should be complete:

### Documentation (docs/)
- [x] architecture.md (project-architect)
- [x] agent_assignments.json (project-architect)
- [x] implementation_roadmap.md (project-architect)
- [ ] requirements.md (requirements-analyst)
- [ ] feature_priority.json (requirements-analyst)
- [ ] ui_guide.md (ui-enhancer)
- [ ] test_coverage_report.html (test-engineer-1)
- [ ] benchmark_results.md (test-engineer-2)
- [ ] integration_report.md (integration-master)
- [ ] code_quality_report.md (code-quality)

### Core Modules (core/)
- [ ] video_pipeline.py (video-specialist)
- [ ] batch_processor.py (video-specialist)
- [ ] model_manager.py (ml-specialist)
- [ ] config.py (ml-specialist)

### Trackers (trackers/)
- [ ] base_tracker.py (tracker-dev-1)
- [ ] bytetrack.py (tracker-dev-1)
- [ ] botsort.py (tracker-dev-2)
- [ ] hybrid_tracker.py (tracker-dev-2)

### Utilities (utils/)
- [ ] hardware.py (cross-platform-dev)
- [ ] metrics.py (cross-platform-dev)
- [ ] funscript.py (cross-platform-dev)
- [ ] logger.py (cross-platform-dev)

### UI (ui/)
- [ ] main_window.py (ui-architect)
- [ ] agent_dashboard.py (ui-architect)
- [ ] settings_panel.py (ui-architect)
- [ ] components/ (ui-enhancer)
- [ ] themes.py (ui-enhancer)

### Tests (tests/)
- [ ] unit/ (test-engineer-1)
- [ ] integration/ (test-engineer-2)
- [ ] benchmarks/ (test-engineer-2)
- [ ] conftest.py (test-engineer-1)

### Configuration
- [ ] main.py (integration-master)
- [ ] .pre-commit-config.yaml (code-quality)
- [ ] pyproject.toml (code-quality)

---

## Next Steps for Agents

### Immediate (Minutes 5-10)
1. **requirements-analyst**: Start extracting FunGen features
2. **video-specialist**: Begin VideoPipeline implementation
3. **ml-specialist**: Start ModelManager with TensorRT support
4. **cross-platform-dev**: Implement hardware detection

### Parallel (Minutes 10-20)
5. **tracker-dev-1**: Implement ByteTrack
6. **tracker-dev-2**: Implement HybridTracker (after TD1 completes base)
7. **ui-architect**: Build main UI structure
8. **ui-enhancer**: Polish UI (after UA completes base)

### Final (Minutes 20-30)
9. **test-engineer-1**: Run unit tests, achieve 80%+ coverage
10. **test-engineer-2**: Run integration tests, benchmarks
11. **integration-master**: Combine all modules, remove duplicates
12. **code-quality**: Format, type-check, document

### On-Demand (Minutes 15-30)
13. **gpu-debugger**: Monitor for CUDA errors, fix as needed
14. **python-debugger**: Monitor for Python errors, fix as needed

---

## Final Notes

This roadmap is designed for maximum parallelization. Agents should:

1. **Read `docs/architecture.md`** for detailed specifications
2. **Read `docs/agent_assignments.json`** for task details
3. **Update `progress/{agent}.json`** every 2 minutes
4. **Monitor `communication/{agent}_inbox.json`** for bug fixes
5. **Follow API contracts** defined in architecture.md
6. **Run tests continuously** to catch issues early
7. **Communicate via JSON files** (no direct agent interaction)

**Success depends on:**
- Clear task boundaries (no overlap)
- Consistent API contracts (defined interfaces)
- Early integration (don't wait until end)
- Continuous testing (catch bugs early)
- Effective communication (JSON progress/inbox/outbox)

---

**Project Start:** 2025-10-24T19:59:00Z
**Expected Completion:** 2025-10-24T20:29:00Z
**Status:** Phase 1 (Planning) - 75% complete

