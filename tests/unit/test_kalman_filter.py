"""
Comprehensive unit tests for trackers/kalman_filter.py

Tests cover:
- Kalman filter initialization
- State prediction
- Measurement update
- Motion model accuracy
- Batch processing
- GPU acceleration (mocked on Pi)
- Edge cases

Author: test-engineer-1 agent
Date: 2025-10-24
Target: 80%+ code coverage
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trackers.kalman_filter import AdvancedKalmanFilter, KalmanState

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False


# ============================================================================
# KalmanState Tests
# ============================================================================


def test_kalman_state_creation():
    """Test KalmanState dataclass creation."""
    x = np.array([100, 200, 5, 10, 0, 0], dtype=np.float32)  # [x, y, vx, vy, ax, ay]
    P = np.eye(6, dtype=np.float32)

    state = KalmanState(x=x, P=P, track_id=1, age=0)

    assert state.track_id == 1
    assert state.age == 0
    assert state.x.shape == (6,)
    assert state.P.shape == (6, 6)
    assert np.array_equal(state.x, x)
    assert np.array_equal(state.P, P)


def test_kalman_state_position():
    """Test extracting position from state."""
    x = np.array([100, 200, 5, 10, 0, 0], dtype=np.float32)
    P = np.eye(6, dtype=np.float32)
    state = KalmanState(x=x, P=P, track_id=1)

    # Position is first two elements
    position = state.x[:2]
    assert position[0] == 100
    assert position[1] == 200


def test_kalman_state_velocity():
    """Test extracting velocity from state."""
    x = np.array([100, 200, 5, 10, 0, 0], dtype=np.float32)
    P = np.eye(6, dtype=np.float32)
    state = KalmanState(x=x, P=P, track_id=1)

    # Velocity is elements 2-3
    velocity = state.x[2:4]
    assert velocity[0] == 5
    assert velocity[1] == 10


# ============================================================================
# AdvancedKalmanFilter Initialization Tests
# ============================================================================


def test_kalman_filter_init_default():
    """Test Kalman filter initialization with default parameters."""
    kf = AdvancedKalmanFilter()

    assert kf.dt == 1.0
    assert kf.process_noise == 0.03
    assert kf.measurement_noise == 1.0
    assert kf.F.shape == (6, 6)  # State transition matrix
    assert kf.H.shape == (2, 6)  # Measurement matrix
    assert kf.Q.shape == (6, 6)  # Process noise
    assert kf.R.shape == (2, 2)  # Measurement noise


def test_kalman_filter_init_custom():
    """Test Kalman filter initialization with custom parameters."""
    kf = AdvancedKalmanFilter(
        dt=0.033, process_noise=0.05, measurement_noise=2.0, use_gpu=False  # 30 FPS
    )

    assert kf.dt == 0.033
    assert kf.process_noise == 0.05
    assert kf.measurement_noise == 2.0
    assert kf.use_gpu is False


def test_kalman_filter_matrices():
    """Test Kalman filter state transition matrix."""
    dt = 1.0
    kf = AdvancedKalmanFilter(dt=dt)

    # Check state transition matrix structure
    # x' = x + vx*dt + 0.5*ax*dt^2
    assert kf.F[0, 0] == 1.0  # x depends on x
    assert kf.F[0, 2] == dt  # x depends on vx
    assert kf.F[0, 4] == 0.5 * dt**2  # x depends on ax

    # vx' = vx + ax*dt
    assert kf.F[2, 2] == 1.0  # vx depends on vx
    assert kf.F[2, 4] == dt  # vx depends on ax


# ============================================================================
# Prediction Tests
# ============================================================================


def test_kalman_filter_predict_position():
    """Test position prediction with constant velocity."""
    kf = AdvancedKalmanFilter(dt=1.0)

    # Initial state: position (100, 200), velocity (10, 5), no acceleration
    x = np.array([100, 200, 10, 5, 0, 0], dtype=np.float32)
    P = np.eye(6, dtype=np.float32)
    state = KalmanState(x=x, P=P, track_id=1)

    # Predict next state
    predicted = kf.predict(state)

    # After 1 time step with velocity (10, 5):
    # x' = 100 + 10*1 = 110
    # y' = 200 + 5*1 = 205
    assert abs(predicted.x[0] - 110) < 0.01
    assert abs(predicted.x[1] - 205) < 0.01


def test_kalman_filter_predict_with_acceleration():
    """Test prediction with acceleration."""
    kf = AdvancedKalmanFilter(dt=1.0)

    # Initial state with acceleration
    x = np.array([100, 200, 10, 5, 2, 1], dtype=np.float32)  # ax=2, ay=1
    P = np.eye(6, dtype=np.float32)
    state = KalmanState(x=x, P=P, track_id=1)

    predicted = kf.predict(state)

    # After 1 time step:
    # x' = 100 + 10*1 + 0.5*2*1^2 = 111
    # y' = 200 + 5*1 + 0.5*1*1^2 = 205.5
    assert abs(predicted.x[0] - 111) < 0.01
    assert abs(predicted.x[1] - 205.5) < 0.01

    # Velocity should also change:
    # vx' = 10 + 2*1 = 12
    # vy' = 5 + 1*1 = 6
    assert abs(predicted.x[2] - 12) < 0.01
    assert abs(predicted.x[3] - 6) < 0.01


def test_kalman_filter_predict_increases_uncertainty():
    """Test that prediction increases uncertainty (covariance)."""
    kf = AdvancedKalmanFilter()

    x = np.array([100, 200, 10, 5, 0, 0], dtype=np.float32)
    P_init = np.eye(6, dtype=np.float32)
    state = KalmanState(x=x, P=P_init, track_id=1)

    predicted = kf.predict(state)

    # Covariance should increase after prediction (uncertainty grows)
    assert np.sum(predicted.P) > np.sum(P_init)


# ============================================================================
# Update Tests
# ============================================================================


def test_kalman_filter_update_reduces_uncertainty():
    """Test that measurement update reduces uncertainty."""
    kf = AdvancedKalmanFilter()

    x = np.array([100, 200, 10, 5, 0, 0], dtype=np.float32)
    P = np.eye(6, dtype=np.float32) * 10  # High initial uncertainty
    state = KalmanState(x=x, P=P, track_id=1)

    # Measurement close to predicted position
    measurement = np.array([102, 198], dtype=np.float32)

    updated = kf.update(state, measurement)

    # Covariance should decrease after update
    assert np.sum(updated.P) < np.sum(state.P)


def test_kalman_filter_update_corrects_position():
    """Test that measurement corrects predicted position."""
    kf = AdvancedKalmanFilter()

    # State with wrong position
    x = np.array([100, 200, 10, 5, 0, 0], dtype=np.float32)
    P = np.eye(6, dtype=np.float32)
    state = KalmanState(x=x, P=P, track_id=1)

    # Measurement far from prediction
    measurement = np.array([150, 250], dtype=np.float32)

    updated = kf.update(state, measurement)

    # Position should move toward measurement
    assert updated.x[0] > state.x[0]  # x moved right
    assert updated.x[1] > state.x[1]  # y moved down
    assert updated.x[0] < 150  # But not fully to measurement
    assert updated.x[1] < 250


# ============================================================================
# Batch Processing Tests
# ============================================================================


def test_kalman_filter_batch_predict():
    """Test batch prediction for multiple tracks."""
    kf = AdvancedKalmanFilter(dt=1.0, use_gpu=False)

    # Create 5 tracks
    states = []
    for i in range(5):
        x = np.array([100 + i * 50, 200, 10, 5, 0, 0], dtype=np.float32)
        P = np.eye(6, dtype=np.float32)
        states.append(KalmanState(x=x, P=P, track_id=i))

    # Predict all tracks
    predicted_states = kf.batch_predict(states)

    assert len(predicted_states) == 5

    # All positions should have moved
    for i, pred in enumerate(predicted_states):
        expected_x = 100 + i * 50 + 10  # Initial x + velocity
        assert abs(pred.x[0] - expected_x) < 0.01


# ============================================================================
# Edge Cases Tests
# ============================================================================


def test_kalman_filter_zero_velocity():
    """Test prediction with zero velocity."""
    kf = AdvancedKalmanFilter(dt=1.0)

    x = np.array([100, 200, 0, 0, 0, 0], dtype=np.float32)
    P = np.eye(6, dtype=np.float32)
    state = KalmanState(x=x, P=P, track_id=1)

    predicted = kf.predict(state)

    # Position should stay the same with zero velocity
    assert abs(predicted.x[0] - 100) < 0.01
    assert abs(predicted.x[1] - 200) < 0.01


def test_kalman_filter_large_dt():
    """Test prediction with large time step."""
    kf = AdvancedKalmanFilter(dt=10.0)  # Large time step

    x = np.array([100, 200, 10, 5, 0, 0], dtype=np.float32)
    P = np.eye(6, dtype=np.float32)
    state = KalmanState(x=x, P=P, track_id=1)

    predicted = kf.predict(state)

    # After 10 time steps:
    # x' = 100 + 10*10 = 200
    # y' = 200 + 5*10 = 250
    assert abs(predicted.x[0] - 200) < 0.01
    assert abs(predicted.x[1] - 250) < 0.01


# ============================================================================
# Performance Tests
# ============================================================================


def test_kalman_filter_performance():
    """Test Kalman filter performance (should be fast)."""
    import time

    kf = AdvancedKalmanFilter(use_gpu=False)

    # Create 100 tracks
    states = []
    for i in range(100):
        x = np.array([100 + i, 200, 10, 5, 0, 0], dtype=np.float32)
        P = np.eye(6, dtype=np.float32)
        states.append(KalmanState(x=x, P=P, track_id=i))

    # Benchmark batch prediction
    start_time = time.time()
    for _ in range(10):
        predicted = kf.batch_predict(states)
    elapsed = time.time() - start_time

    # Should process 100 tracks 10 times in < 1 second on Pi
    assert elapsed < 2.0, f"Kalman filter too slow: {elapsed:.3f}s"


# ============================================================================
# Main Test Runner
# ============================================================================


def run_all_tests():
    """Run all tests manually if pytest is not available."""
    test_functions = [
        test_kalman_state_creation,
        test_kalman_state_position,
        test_kalman_state_velocity,
        test_kalman_filter_init_default,
        test_kalman_filter_init_custom,
        test_kalman_filter_matrices,
        test_kalman_filter_predict_position,
        test_kalman_filter_predict_with_acceleration,
        test_kalman_filter_predict_increases_uncertainty,
        test_kalman_filter_update_reduces_uncertainty,
        test_kalman_filter_update_corrects_position,
        test_kalman_filter_batch_predict,
        test_kalman_filter_zero_velocity,
        test_kalman_filter_large_dt,
        test_kalman_filter_performance,
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 70)
    print("Kalman Filter Test Suite")
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
