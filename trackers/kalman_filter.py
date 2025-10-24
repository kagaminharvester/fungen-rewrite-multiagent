"""
Advanced Kalman Filter implementation with GPU acceleration support.

This module provides a high-performance Kalman filter for motion prediction in
multi-object tracking. It supports both CPU and GPU (CUDA) acceleration for
maximum performance on different platforms.

Author: tracker-dev-2 agent
Date: 2025-10-24
Target Platform: Raspberry Pi (dev) + RTX 3090 (prod)
Performance Target: <1ms per track prediction, GPU-accelerated batch processing
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class KalmanState:
    """State representation for a Kalman filter.

    Attributes:
        x: State vector [x, y, vx, vy, ax, ay] - position, velocity, acceleration
        P: Error covariance matrix
        track_id: Associated track ID
        age: Number of updates since initialization
    """

    x: np.ndarray  # State vector (6x1)
    P: np.ndarray  # Covariance matrix (6x6)
    track_id: int
    age: int = 0


class AdvancedKalmanFilter:
    """Advanced Kalman filter for 2D object tracking with GPU support.

    This implementation uses a 6-state model: [x, y, vx, vy, ax, ay]
    - x, y: Object center position
    - vx, vy: Velocity
    - ax, ay: Acceleration

    The filter supports:
    - Constant acceleration model
    - Adaptive process noise
    - GPU-accelerated batch prediction
    - Occlusion handling
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 0.03,
        measurement_noise: float = 1.0,
        use_gpu: bool = True,
    ):
        """Initialize Advanced Kalman Filter.

        Args:
            dt: Time step between frames (1.0 for frame-based)
            process_noise: Process noise covariance multiplier
            measurement_noise: Measurement noise covariance multiplier
            use_gpu: Whether to use GPU acceleration if available
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.use_gpu = use_gpu and TORCH_AVAILABLE

        # State transition matrix (constant acceleration model)
        # x' = x + vx*dt + 0.5*ax*dt^2
        # vx' = vx + ax*dt
        # ax' = ax (constant)
        self.F = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt**2, 0],
                [0, 1, 0, dt, 0, 0.5 * dt**2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Measurement matrix (we only observe position)
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float32)

        # Process noise covariance matrix
        self.Q = np.eye(6, dtype=np.float32) * process_noise

        # Measurement noise covariance matrix
        self.R = np.eye(2, dtype=np.float32) * measurement_noise

        # GPU tensors (initialized on first use)
        self.F_gpu: Optional[torch.Tensor] = None
        self.H_gpu: Optional[torch.Tensor] = None
        self.Q_gpu: Optional[torch.Tensor] = None
        self.R_gpu: Optional[torch.Tensor] = None

    def _init_gpu_tensors(self):
        """Initialize GPU tensors for batch processing."""
        if not self.use_gpu or self.F_gpu is not None:
            return

        device = torch.device("cuda")
        self.F_gpu = torch.from_numpy(self.F).to(device)
        self.H_gpu = torch.from_numpy(self.H).to(device)
        self.Q_gpu = torch.from_numpy(self.Q).to(device)
        self.R_gpu = torch.from_numpy(self.R).to(device)

    def initialize(self, position: Tuple[float, float], track_id: int) -> KalmanState:
        """Initialize a new Kalman state for a track.

        Args:
            position: Initial (x, y) position
            track_id: Track identifier

        Returns:
            Initialized KalmanState
        """
        x, y = position

        # Initial state: [x, y, 0, 0, 0, 0] (zero velocity and acceleration)
        state_vector = np.array([x, y, 0, 0, 0, 0], dtype=np.float32).reshape(6, 1)

        # Initial covariance (high uncertainty)
        P = np.eye(6, dtype=np.float32) * 10.0

        return KalmanState(x=state_vector, P=P, track_id=track_id, age=0)

    def predict(self, state: KalmanState) -> KalmanState:
        """Predict next state using motion model.

        Args:
            state: Current Kalman state

        Returns:
            Predicted Kalman state
        """
        # Predict state: x' = F * x
        x_pred = self.F @ state.x

        # Predict covariance: P' = F * P * F^T + Q
        P_pred = self.F @ state.P @ self.F.T + self.Q

        return KalmanState(x=x_pred, P=P_pred, track_id=state.track_id, age=state.age)

    def update(self, state: KalmanState, measurement: Tuple[float, float]) -> KalmanState:
        """Update state with new measurement (correction step).

        Args:
            state: Predicted Kalman state
            measurement: Observed (x, y) position

        Returns:
            Updated Kalman state
        """
        x, y = measurement
        z = np.array([x, y], dtype=np.float32).reshape(2, 1)

        # Innovation: y = z - H * x
        innovation = z - self.H @ state.x

        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ state.P @ self.H.T + self.R

        # Kalman gain: K = P * H^T * S^-1
        K = state.P @ self.H.T @ np.linalg.inv(S)

        # Update state: x' = x + K * y
        x_updated = state.x + K @ innovation

        # Update covariance: P' = (I - K * H) * P
        I = np.eye(6, dtype=np.float32)
        P_updated = (I - K @ self.H) @ state.P

        return KalmanState(x=x_updated, P=P_updated, track_id=state.track_id, age=state.age + 1)

    def predict_batch(self, states: List[KalmanState]) -> List[KalmanState]:
        """Predict next state for multiple tracks (GPU-accelerated if available).

        Args:
            states: List of current Kalman states

        Returns:
            List of predicted Kalman states
        """
        if not states:
            return []

        if self.use_gpu and len(states) > 5:
            return self._predict_batch_gpu(states)
        else:
            return [self.predict(state) for state in states]

    def _predict_batch_gpu(self, states: List[KalmanState]) -> List[KalmanState]:
        """GPU-accelerated batch prediction.

        Args:
            states: List of Kalman states

        Returns:
            List of predicted states
        """
        self._init_gpu_tensors()

        device = torch.device("cuda")

        # Stack state vectors and covariance matrices
        x_batch = torch.stack([torch.from_numpy(s.x).squeeze() for s in states]).to(device)
        P_batch = torch.stack([torch.from_numpy(s.P) for s in states]).to(device)

        # Batch predict: x' = F * x
        x_pred = torch.matmul(self.F_gpu, x_batch.unsqueeze(-1)).squeeze(-1)

        # Batch predict covariance: P' = F * P * F^T + Q
        P_pred = (
            torch.matmul(torch.matmul(self.F_gpu, P_batch), self.F_gpu.transpose(0, 1)) + self.Q_gpu
        )

        # Convert back to numpy
        x_pred_np = x_pred.cpu().numpy()
        P_pred_np = P_pred.cpu().numpy()

        # Create updated states
        predicted_states = []
        for i, state in enumerate(states):
            predicted_states.append(
                KalmanState(
                    x=x_pred_np[i].reshape(6, 1),
                    P=P_pred_np[i],
                    track_id=state.track_id,
                    age=state.age,
                )
            )

        return predicted_states

    def get_position(self, state: KalmanState) -> Tuple[float, float]:
        """Extract current position from state.

        Args:
            state: Kalman state

        Returns:
            (x, y) position tuple
        """
        return (float(state.x[0, 0]), float(state.x[1, 0]))

    def get_velocity(self, state: KalmanState) -> Tuple[float, float]:
        """Extract current velocity from state.

        Args:
            state: Kalman state

        Returns:
            (vx, vy) velocity tuple
        """
        return (float(state.x[2, 0]), float(state.x[3, 0]))

    def get_acceleration(self, state: KalmanState) -> Tuple[float, float]:
        """Extract current acceleration from state.

        Args:
            state: Kalman state

        Returns:
            (ax, ay) acceleration tuple
        """
        return (float(state.x[4, 0]), float(state.x[5, 0]))

    def predict_n_steps(self, state: KalmanState, n: int) -> List[Tuple[float, float]]:
        """Predict positions for next N steps (for occlusion handling).

        Args:
            state: Current Kalman state
            n: Number of steps to predict

        Returns:
            List of predicted (x, y) positions
        """
        predicted_positions = []
        current_state = state

        for _ in range(n):
            current_state = self.predict(current_state)
            predicted_positions.append(self.get_position(current_state))

        return predicted_positions

    def adaptive_process_noise(self, state: KalmanState, innovation_variance: float) -> None:
        """Adapt process noise based on innovation variance (adaptive Kalman).

        Args:
            state: Current Kalman state
            innovation_variance: Variance of recent innovations
        """
        # Increase process noise if innovation is high (model mismatch)
        if innovation_variance > 100.0:
            self.Q *= 1.1  # Increase by 10%
        elif innovation_variance < 10.0:
            self.Q *= 0.9  # Decrease by 10%

        # Clamp to reasonable bounds
        self.Q = np.clip(self.Q, 0.01, 1.0)


class KalmanFilterCV2:
    """Lightweight Kalman filter using OpenCV (fallback for systems without torch).

    This is a simpler 4-state model: [x, y, vx, vy] (constant velocity model).
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 0.03,
        measurement_noise: float = 1.0,
    ):
        """Initialize CV2 Kalman Filter.

        Args:
            dt: Time step between frames
            process_noise: Process noise covariance multiplier
            measurement_noise: Measurement noise covariance multiplier
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.filters = {}  # type: dict[int, Any]

    def _create_filter(self):
        """Create a new CV2 Kalman filter.

        Returns:
            Initialized cv2.KalmanFilter
        """
        if not CV2_AVAILABLE:
            return None

        kf = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurements

        # State transition matrix
        kf.transitionMatrix = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )

        # Measurement matrix
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

        # Process noise covariance
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise

        # Measurement noise covariance
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise

        # Error covariance
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        return kf

    def initialize(self, position: Tuple[float, float], track_id: int) -> None:
        """Initialize Kalman filter for a track.

        Args:
            position: Initial (x, y) position
            track_id: Track identifier
        """
        kf = self._create_filter()
        x, y = position
        kf.statePost = np.array([x, y, 0, 0], dtype=np.float32).reshape(4, 1)
        self.filters[track_id] = kf

    def predict(self, track_id: int) -> Optional[Tuple[float, float]]:
        """Predict next position for a track.

        Args:
            track_id: Track identifier

        Returns:
            Predicted (x, y) position or None if track not found
        """
        if track_id not in self.filters:
            return None

        prediction = self.filters[track_id].predict()
        return (float(prediction[0, 0]), float(prediction[1, 0]))

    def update(
        self, track_id: int, measurement: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Update filter with new measurement.

        Args:
            track_id: Track identifier
            measurement: Observed (x, y) position

        Returns:
            Corrected (x, y) position or None if track not found
        """
        if track_id not in self.filters:
            return None

        x, y = measurement
        measurement_vec = np.array([x, y], dtype=np.float32).reshape(2, 1)
        corrected = self.filters[track_id].correct(measurement_vec)

        return (float(corrected[0, 0]), float(corrected[1, 0]))

    def remove_track(self, track_id: int) -> None:
        """Remove Kalman filter for a track.

        Args:
            track_id: Track identifier
        """
        if track_id in self.filters:
            del self.filters[track_id]
