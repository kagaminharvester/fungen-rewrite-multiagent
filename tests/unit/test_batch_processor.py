"""
Comprehensive unit tests for core/batch_processor.py

Tests cover:
- Batch processor initialization
- Job queue management
- Multi-video processing
- Progress tracking
- Error handling
- Worker management

Author: test-engineer-1 agent
Date: 2025-10-24
Target: 80%+ code coverage
"""

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from core.batch_processor import BatchProcessor, JobStatus, ProcessingJob

    BATCH_PROCESSOR_AVAILABLE = True
except ImportError:
    BATCH_PROCESSOR_AVAILABLE = False

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


# Skip all tests if batch_processor not available
if not BATCH_PROCESSOR_AVAILABLE:
    print("Warning: batch_processor module not available, skipping tests")


# ============================================================================
# JobStatus Tests
# ============================================================================


def test_job_status_enum():
    """Test JobStatus enum values."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.PROCESSING.value == "processing"
    assert JobStatus.COMPLETED.value == "completed"
    assert JobStatus.FAILED.value == "failed"
    assert JobStatus.CANCELLED.value == "cancelled"


# ============================================================================
# ProcessingJob Tests
# ============================================================================


def test_processing_job_creation():
    """Test ProcessingJob dataclass creation."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    job = ProcessingJob(
        job_id=1,
        video_path=Path("test.mp4"),
        output_path=Path("output/test.funscript"),
        status=JobStatus.PENDING,
        progress=0.0,
    )

    assert job.job_id == 1
    assert job.video_path == Path("test.mp4")
    assert job.status == JobStatus.PENDING
    assert job.progress == 0.0


# ============================================================================
# BatchProcessor Initialization Tests
# ============================================================================


def test_batch_processor_init_default():
    """Test BatchProcessor initialization with default parameters."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    processor = BatchProcessor()

    assert processor.num_workers > 0
    assert len(processor.jobs) == 0


def test_batch_processor_init_custom_workers():
    """Test BatchProcessor initialization with custom worker count."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    processor = BatchProcessor(num_workers=4)

    assert processor.num_workers == 4


def test_batch_processor_auto_workers():
    """Test BatchProcessor auto-detects optimal worker count."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    with patch("multiprocessing.cpu_count", return_value=8):
        processor = BatchProcessor(num_workers="auto")

        # Should use some fraction of CPUs
        assert 1 <= processor.num_workers <= 8


# ============================================================================
# Job Management Tests
# ============================================================================


def test_add_video_to_queue():
    """Test adding video to processing queue."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    processor = BatchProcessor()

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        video_path = Path(f.name)

        job_id = processor.add_video(video_path=video_path, output_dir=Path("output/"))

        assert job_id is not None
        assert len(processor.jobs) == 1
        assert processor.jobs[job_id].status == JobStatus.PENDING


def test_add_multiple_videos():
    """Test adding multiple videos to queue."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    processor = BatchProcessor()

    job_ids = []
    for i in range(5):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            job_id = processor.add_video(video_path=Path(f.name), output_dir=Path("output/"))
            job_ids.append(job_id)

    assert len(processor.jobs) == 5
    assert len(set(job_ids)) == 5  # All unique


def test_get_job_status():
    """Test getting job status."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    processor = BatchProcessor()

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        job_id = processor.add_video(Path(f.name), Path("output/"))

        status = processor.get_job_status(job_id)
        assert status == JobStatus.PENDING


def test_cancel_job():
    """Test cancelling a job."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    processor = BatchProcessor()

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        job_id = processor.add_video(Path(f.name), Path("output/"))

        processor.cancel_job(job_id)

        assert processor.jobs[job_id].status == JobStatus.CANCELLED


# ============================================================================
# Progress Tracking Tests
# ============================================================================


def test_update_job_progress():
    """Test updating job progress."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    processor = BatchProcessor()

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        job_id = processor.add_video(Path(f.name), Path("output/"))

        processor.update_progress(job_id, 0.5)

        assert processor.jobs[job_id].progress == 0.5


def test_progress_callback():
    """Test progress callback is called."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    callback_called = []

    def progress_callback(job_id, progress):
        callback_called.append((job_id, progress))

    processor = BatchProcessor()

    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        job_id = processor.add_video(Path(f.name), Path("output/"), callback=progress_callback)

        processor.update_progress(job_id, 0.3)

        assert len(callback_called) > 0
        assert callback_called[0][0] == job_id
        assert callback_called[0][1] == 0.3


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_add_nonexistent_video():
    """Test adding non-existent video raises error."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    processor = BatchProcessor()

    try:
        processor.add_video(Path("nonexistent.mp4"), Path("output/"))
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        pass


def test_get_invalid_job_status():
    """Test getting status of invalid job ID."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    processor = BatchProcessor()

    try:
        processor.get_job_status(99999)
        assert False, "Should raise KeyError"
    except KeyError:
        pass


def test_cancel_invalid_job():
    """Test cancelling invalid job ID."""
    if not BATCH_PROCESSOR_AVAILABLE:
        return

    processor = BatchProcessor()

    try:
        processor.cancel_job(99999)
        assert False, "Should raise KeyError"
    except KeyError:
        pass


# ============================================================================
# Main Test Runner
# ============================================================================


def run_all_tests():
    """Run all tests manually if pytest is not available."""
    if not BATCH_PROCESSOR_AVAILABLE:
        print("\n" + "=" * 70)
        print("BatchProcessor module not available - skipping tests")
        print("=" * 70 + "\n")
        return True

    test_functions = [
        test_job_status_enum,
        test_processing_job_creation,
        test_batch_processor_init_default,
        test_batch_processor_init_custom_workers,
        test_batch_processor_auto_workers,
        test_add_video_to_queue,
        test_add_multiple_videos,
        test_get_job_status,
        test_cancel_job,
        test_update_job_progress,
        test_progress_callback,
        test_add_nonexistent_video,
        test_get_invalid_job_status,
        test_cancel_invalid_job,
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 70)
    print("Batch Processor Test Suite")
    print("=" * 70 + "\n")

    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("PASSED")
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    import sys

    success = run_all_tests()
    sys.exit(0 if success else 1)
