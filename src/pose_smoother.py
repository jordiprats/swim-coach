"""
Advanced pose smoothing: Savitzky–Golay + One Euro + Kalman + Occlusion handling.

This replaces the old simple moving-average + interpolation approach with
proper temporal filters that trade speed for significantly better accuracy.

Filters applied per-landmark:
  1. Savitzky–Golay (scipy) — preserves high-frequency biomechanics (stroke
     transitions) while removing per-frame pose-estimation jitter.
  2. One Euro filter — adaptive low-pass that reduces latency during fast
     movements (recovery) and smooths aggressively during slow movements (catch).
  3. Kalman filter (lightweight, per-coordinate) — used on torso landmarks
     (shoulders, hips) that should have very smooth trajectories.
  4. Occlusion interpolation — fills short gaps (< 5 frames) with cubic
     spline interpolation instead of linear, preserving movement curvature.

All filters operate on the full time series after detection, so we can use
non-causal (centred) windows — impossible in real-time.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.signal import savgol_filter
from collections import defaultdict

# ── Tunable parameters ──

# Savitzky–Golay: window length must be odd, polyorder ≤ window length
SG_WINDOW = 9          # Frames — 9 @ 30fps ≈ 300ms temporal context
SG_POLYORDER = 3       # Cubic fit — good for smooth joint trajectories

# One Euro filter parameters
ONE_EURO_BETA = 0.7    # Low-speed cutoff frequency (lower = smoother)
ONE_EURO_MIN_CUTOFF = 0.1  # Hz — minimum cutoff during slow motion
ONE_EURO_CUTOFF_AT_RATE = 1.0  # How much cutoff increases with speed

# Kalman filter parameters (per-coordinate)
KALMAN_PROCESS_NOISE = 1e-3   # How much we trust the motion model
KALMAN_MEASUREMENT_NOISE = 0.5  # How much we trust each measurement

# Maximum gap (in frames) for spline interpolation; longer gaps get linear
MAX_SPLINE_GAP = 5

# Landmarks that get Kalman filtering (torso — should be very smooth)
KALMAN_LANDMARKS = {'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'}


# ── Helper classes ──

class OneEuroFilter:
    """Lightweight 1D One Euro filter for a single coordinate signal."""

    def __init__(self, beta: float = ONE_EURO_BETA,
                 min_cutoff: float = ONE_EURO_MIN_CUTOFF,
                 cutoff_at_rate: float = ONE_EURO_CUTOFF_AT_RATE):
        self.beta = beta
        self.min_cutoff = min_cutoff
        self.cutoff_at_rate = cutoff_at_rate
        self.prev_low = None
        self.prev_dx = None
        self.dt = 1.0  # Assume uniform frame rate; caller can override

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply One Euro filter to a 1-D signal array."""
        if len(signal) < 2:
            return signal.copy()

        out = np.empty_like(signal)
        dx = 0.0
        low_pass = signal[0]

        for i in range(len(signal)):
            dt = self.dt
            if self.prev_low is not None:
                dx = (signal[i] - self.prev_low) / dt

            # Cutoff frequency: increases with movement speed
            cutoff = self.min_cutoff + self.beta * abs(dx)
            # Smoothing factor (exponential smoothing)
            tau = 1.0 / (2.0 * np.pi * cutoff)
            alpha = dt / (dt + tau)
            alpha = np.clip(alpha, 0.0, 1.0)

            low_pass = low_pass + alpha * (signal[i] - low_pass)
            out[i] = low_pass
            self.prev_low = signal[i]

        return out


class KalmanFilter1D:
    """Simple 1D Kalman filter for smoothing a coordinate over time."""

    def __init__(self, process_noise: float = KALMAN_PROCESS_NOISE,
                 measurement_noise: float = KALMAN_MEASUREMENT_NOISE):
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance
        self.x = None  # State estimate
        self.P = 1.0   # Estimate error covariance
        self.initialized = False

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply Kalman filter to a 1-D signal array."""
        if len(signal) < 2:
            return signal.copy()

        out = np.empty_like(signal)

        for i in range(len(signal)):
            z = signal[i]  # Measurement
            if not self.initialized:
                self.x = z
                self.P = 1.0
                self.initialized = True
                out[i] = z
                continue

            # Predict (assume constant velocity; we can extend later)
            # For now, just predict same as previous
            # State transition: x = x_prev (constant position model)
            self.P += self.Q  # Update covariance

            # Update (measurement)
            K = self.P / (self.P + self.R)  # Kalman gain
            self.x = self.x + K * (z - self.x)
            self.P = (1 - K) * self.P

            out[i] = self.x

        return out


# ── Main smoothing pipeline ──

def smooth_pose_data(pose_data: List[Dict]) -> List[Dict]:
    """
    Apply advanced temporal smoothing to pose data.

    Operates on the full time series, using non-causal filters.
    Modifies pose_data in-place and returns it.

    Steps:
      1. Extract per-landmark time series for all coordinates.
      2. Apply Savitzky–Golay to each coordinate time series.
      3. Apply One Euro filter (non-causal variant) for adaptive smoothing.
      4. Apply Kalman filter to torso landmarks.
      5. Cubic-spline interpolation for short occlusion gaps.
      6. Rebuild landmark dicts from filtered time series.
      7. Recalculate visibility based on temporal consistency.

    Returns the modified pose_data list.
    """
    if not pose_data:
        return pose_data

    print("\n[Precision] Applying advanced temporal smoothing...")
    print(f"  Savitzky–Golay: window={SG_WINDOW}, polyorder={SG_POLYORDER}")
    print(f"  One Euro filter: beta={ONE_EURO_BETA}")
    print(f"  Kalman filter on torso landmarks")

    # ── Step 1: Extract time series per landmark ──
    n_frames = len(pose_data)
    landmarks_of_interest = [
        'nose',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
    ]

    # Build arrays: for each landmark, we have (x, y, z) time series
    # We only smooth coordinates where visibility > 0.5 for most frames
    time_series = {}
    for lm in landmarks_of_interest:
        xs = []
        ys = []
        zs = []
        for fd in pose_data:
            pose = fd.get('pose')
            if pose is None:
                xs.append(np.nan)
                ys.append(np.nan)
                zs.append(np.nan)
            else:
                l = pose['landmarks'].get(lm)
                if l is None or l.get('visibility', 0) < 0.3:
                    xs.append(np.nan)
                    ys.append(np.nan)
                    zs.append(np.nan)
                else:
                    xs.append(l['x'])
                    ys.append(l['y'])
                    zs.append(l.get('z', 0))
        time_series[lm] = {
            'x': np.array(xs, dtype=float),
            'y': np.array(ys, dtype=float),
            'z': np.array(zs, dtype=float),
        }

    # ── Step 2: Savitzky–Golay smoothing ──
    for lm in landmarks_of_interest:
        for coord in ('x', 'y', 'z'):
            sig = time_series[lm][coord]
            # Count valid (non-NaN) entries
            valid = ~np.isnan(sig)
            if np.sum(valid) < SG_WINDOW:
                continue

            # Linear interpolation to fill NaNs before filtering
            filled = _fill_nan_linear(sig)

            # Apply Savitzky–Golay
            # Window must be odd and ≤ len(signal)
            w = min(SG_WINDOW, len(filled))
            if w % 2 == 0:
                w -= 1
            w = max(w, 3)  # Minimum window
            try:
                smoothed = savgol_filter(filled, w, SG_POLYORDER)
                # Replace only valid positions; keep original NaNs elsewhere
                smoothed[~valid] = np.nan
                time_series[lm][coord] = smoothed
            except Exception:
                # Fall back to original if SavGol fails
                pass

    # ── Step 3: One Euro filter ──
    # We apply a non-causal variant: forward One Euro, then backward.
    # This gives us the low-latency adaptive behaviour while using
    # non-causal smoothing (better for offline analysis).
    for lm in landmarks_of_interest:
        for coord in ('x', 'y', 'z'):
            sig = time_series[lm][coord]
            valid = ~np.isnan(sig)
            if np.sum(valid) < 3:
                continue

            # Forward-backward One Euro for zero-phase filtering
            filled = _fill_nan_linear(sig)

            # Forward pass
            oef = OneEuroFilter(beta=ONE_EURO_BETA)
            forward = oef.apply(filled)

            # Backward pass (reverse and apply again)
            oef_b = OneEuroFilter(beta=ONE_EURO_BETA * 0.5)  # Less aggressive backward
            backward = oef_b.apply(forward[::-1])
            backward = backward[::-1]

            # Average forward and backward (centred)
            smoothed = (forward + backward) / 2
            smoothed[~valid] = np.nan
            time_series[lm][coord] = smoothed

    # ── Step 4: Kalman filter on torso landmarks ──
    for lm in KALMAN_LANDMARKS:
        for coord in ('x', 'y'):
            sig = time_series[lm][coord]
            valid = ~np.isnan(sig)
            if np.sum(valid) < 5:
                continue

            filled = _fill_nan_linear(sig)
            kf = KalmanFilter1D()
            smoothed = kf.apply(filled)
            smoothed[~valid] = np.nan
            time_series[lm][coord] = smoothed

    # ── Step 5: Cubic spline interpolation for short occlusion gaps ──
    # After filtering, some gaps may have widened due to NaN handling.
    # Use cubic interpolation for gaps ≤ MAX_SPLINE_GAP.
    _spline_interpolate_gaps(time_series, landmarks_of_interest, pose_data)

    # ── Step 6: Rebuild landmark dicts from filtered time series ──
    for i, fd in enumerate(pose_data):
        pose = fd.get('pose')
        if pose is None:
            continue
        landmarks = pose.get('landmarks', {})
        if not landmarks:
            continue

        for lm in landmarks_of_interest:
            if lm not in landmarks:
                continue
            l = landmarks[lm]
            for coord in ('x', 'y', 'z'):
                val = time_series[lm][coord][i]
                if not np.isnan(val):
                    l[coord] = float(val)

    # ── Step 7: Update visibility based on temporal consistency ──
    _update_visibility_from_temporal_consistency(pose_data, landmarks_of_interest)

    # Count valid frames
    valid_count = sum(1 for fd in pose_data if fd['pose'] is not None)
    print(f"  Smoothing complete: {valid_count}/{len(pose_data)} frames with valid pose")
    return pose_data


# ── Helper functions ──

def _fill_nan_linear(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values with linear interpolation."""
    out = arr.copy()
    n = len(out)
    if n < 2:
        return out

    # Find NaN positions
    nan_mask = np.isnan(out)

    # Forward fill
    last_valid = None
    for i in range(n):
        if not nan_mask[i]:
            last_valid = out[i]
        elif last_valid is not None:
            out[i] = last_valid

    # Backward fill (for leading NaNs)
    last_valid = None
    for i in range(n - 1, -1, -1):
        if not nan_mask[i]:
            last_valid = out[i]
        elif last_valid is not None:
            out[i] = last_valid

    # Now linearly interpolate between valid endpoints
    # Find segments between non-NaN values
    i = 0
    while i < n:
        if not np.isnan(out[i]):
            # Find next valid
            j = i + 1
            while j < n and np.isnan(out[j]):
                j += 1
            if j < n:
                # Interpolate between i and j
                start_val = out[i]
                end_val = out[j]
                gap = j - i
                for k in range(i + 1, j):
                    t = (k - i) / gap
                    out[k] = start_val + t * (end_val - start_val)
            i = j
        else:
            i += 1

    return out


def _spline_interpolate_gaps(time_series: Dict, landmarks: List[str],
                              pose_data: List[Dict]):
    """
    Replace linear interpolation with cubic spline for gaps ≤ MAX_SPLINE_GAP.
    """
    from scipy.interpolate import CubicSpline  # lazy import

    n = len(pose_data)
    for lm in landmarks:
        for coord in ('x', 'y', 'z'):
            sig = time_series[lm][coord]
            valid = ~np.isnan(sig)
            valid_indices = np.where(valid)[0]

            if len(valid_indices) < 4:
                continue

            # Build cubic spline from valid points
            try:
                cs = CubicSpline(valid_indices, sig[valid_indices], axis=0,
                                 bc_type='natural')
            except Exception:
                continue

            # Evaluate at all indices, but only replace short gaps
            full = cs(np.arange(n))

            # Find NaN gaps in original
            nan_start = None
            for i in range(n):
                if np.isnan(sig[i]):
                    if nan_start is None:
                        nan_start = i
                elif nan_start is not None:
                    gap_len = i - nan_start
                    if gap_len <= MAX_SPLINE_GAP:
                        # Replace with spline values
                        for j in range(nan_start, i):
                            sig[j] = full[j]
                    nan_start = None
            if nan_start is not None:
                gap_len = n - nan_start
                if gap_len <= MAX_SPLINE_GAP:
                    for j in range(nan_start, n):
                        sig[j] = full[j]


def _update_visibility_from_temporal_consistency(
    pose_data: List[Dict],
    landmarks: List[str],
    max_std: float = 15.0  # Max std of pixel movement over 5-frame window
):
    """
    Down-weight visibility on landmarks whose temporal trajectory is
    suspiciously jittery (indicating a pose-estimation glitch rather than
    genuine movement).
    """
    n = len(pose_data)
    if n < 5:
        return

    for lm in landmarks:
        xs = []
        for fd in pose_data:
            pose = fd.get('pose')
            if pose is None:
                xs.append(np.nan)
            else:
                l = pose['landmarks'].get(lm)
                if l is None:
                    xs.append(np.nan)
                else:
                    xs.append(l.get('x', np.nan))

        xs_arr = np.array(xs, dtype=float)

        # Compute local std over 5-frame windows
        for i in range(2, n - 2):
            window = xs_arr[i - 2:i + 3]
            valid = window[~np.isnan(window)]
            if len(valid) < 3:
                continue
            local_std = np.std(valid)
            if local_std > max_std:
                # Mark this frame's landmark as low-visibility
                fd = pose_data[i]
                if fd['pose'] is not None:
                    l = fd['pose']['landmarks'].get(lm)
                    if l is not None:
                        # Reduce visibility proportionally
                        current_vis = l.get('visibility', 0.5)
                        penalty = min(1.0, local_std / max_std)
                        l['visibility'] = max(0.0, current_vis * (1.0 - penalty * 0.5))
