"""
Stroke cycle segmentation with a proper state machine.

Instead of analyzing all frames indiscriminately, this module detects
individual stroke cycles and segments each into phases:
  - Entry (hand enters water)
  - Catch (hand begins to press back)
  - Pull (arm pulls through)
  - Push (hand passes hip, final propulsion)
  - Recovery (arm lifted out of water)

This enables phase-specific metrics: "catch elbow angle" is different from
"pull elbow angle", and we can report per-cycle statistics (median, IQR)
instead of frame-by-frame noise.

Uses multiple signals simultaneously:
  - Hand velocity (wrist x,y derivatives)
  - Elbow angle trajectory
  - Shoulder rotation angle
  - Wrist relative to shoulder/hip position
  - Body rotation

The state machine is hand-crafted for freestyle; butterfly uses a simpler
dual-arm version.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# ── State definitions ──

STROKE_PHASES = [
    'RECOVERY',
    'ENTRY',
    'EXTENSION',
    'CATCH',
    'PULL',
    'PUSH',
    'EXIT',
]

# Transition matrix: (current_state, signal_conditions) -> next_state
# Conditions are evaluated per-frame

MIN_VISIBILITY = 0.5

# ── Detection parameters ──

# For peak detection in signals
MIN_PEAK_GAP_FRAMES = 10  # Minimum frames between stroke events
MIN_PEAK_DELTA = 15       # Minimum change in signal to count as peak (pixels)

# Phase thresholds (tuned for side-view swimming at ~30 fps)
CATCH_ELBOW_THRESHOLD = 120   # Max elbow angle during catch (degrees)
PULL_ELBOW_MAX = 160          # Max elbow angle during pull
RECOVERY_VELOCITY_THRESHOLD = 30  # Min wrist velocity for recovery (px/frame)


# ── Stroke Cycle Detection ──

class StrokeCycleDetector:
    """Detects stroke cycles and segments phases from pose data."""

    def __init__(self, stroke_type: str = 'freestyle'):
        self.stroke_type = stroke_type
        self.cycles = []  # List of cycle dicts
        self._reset()

    def _reset(self):
        """Reset internal state for a new video."""
        self.state = 'RECOVERY'
        self.cycle_start_frame = None
        self.cycle_count = 0
        self.cycles = []
        self.current_cycle = {
            'entry_frame': None,
            'catch_frame': None,
            'pull_start': None,
            'push_frame': None,
            'exit_frame': None,
            'recovery_start': None,
        }

    def detect_cycles(self, pose_data: List[Dict]) -> List[Dict]:
        """
        Detect stroke cycles in pose data.

        Returns list of cycle dicts with:
          - cycle_number: int
          - entry_frame: frame index in pose_data
          - catch_frame: frame index
          - pull_start: frame index
          - push_frame: frame index
          - exit_frame: frame index
          - recovery_start: frame index
          - duration_frames: total frames in cycle
          - phase_frames: dict of phase -> (start_frame, end_frame)
        """
        self._reset()

        if len(pose_data) < 10:
            return []

        print("\n[Precision] Detecting stroke cycles...")

        # Extract signals
        signals = self._extract_signals(pose_data)

        # Detect cycle boundaries from wrist x trajectory
        cycles_raw = self._detect_cycle_boundaries(signals, pose_data)

        if not cycles_raw:
            print("  No stroke cycles detected")
            return []

        # For each cycle, find phase transition frames
        for start, end in cycles_raw:
            cycle = self._segment_cycle(start, end, signals, pose_data)
            if cycle is not None:
                self.cycles.append(cycle)

        print(f"  Detected {len(self.cycles)} stroke cycles")

        # Assign cycle numbers
        for i, cycle in enumerate(self.cycles):
            cycle['cycle_number'] = i + 1
            if cycle.get('entry_frame') is not None and cycle.get('exit_frame') is not None:
                cycle['duration_frames'] = cycle['exit_frame'] - cycle['entry_frame']

        return self.cycles

    def _extract_signals(self, pose_data: List[Dict]) -> Dict:
        """Extract key signals from pose data for cycle detection."""
        n = len(pose_data)
        signals = {
            'wrist_x': np.full(n, np.nan),
            'wrist_y': np.full(n, np.nan),
            'wrist_velocity': np.full(n, np.nan),
            'elbow_angle': np.full(n, np.nan),
            'shoulder_rotation': np.full(n, np.nan),
            'wrist_to_shoulder_y': np.full(n, np.nan),
            'body_rotation': np.full(n, np.nan),
        }

        # Use left arm as primary signal (side view)
        for i, fd in enumerate(pose_data):
            pose = fd.get('pose')
            if pose is None:
                continue
            landmarks = pose.get('landmarks', {})
            if not landmarks:
                continue

            # Left wrist position
            lw = landmarks.get('left_wrist')
            if lw and lw.get('visibility', 0) >= MIN_VISIBILITY:
                signals['wrist_x'][i] = lw['x']
                signals['wrist_y'][i] = lw['y']

            # Elbow angle (shoulder-elbow-wrist)
            ls = landmarks.get('left_shoulder')
            le = landmarks.get('left_elbow')
            lw_ = landmarks.get('left_wrist')
            if (ls and le and lw_ and
                ls.get('visibility', 0) >= MIN_VISIBILITY and
                le.get('visibility', 0) >= MIN_VISIBILITY and
                lw_.get('visibility', 0) >= MIN_VISIBILITY):
                angle = self._calc_angle(ls, le, lw_)
                signals['elbow_angle'][i] = angle

            # Wrist relative to shoulder (vertical distance)
            if ls and lw and ls.get('visibility', 0) >= MIN_VISIBILITY:
                signals['wrist_to_shoulder_y'][i] = lw['y'] - ls['y']

            # Body rotation estimate
            left_s = landmarks.get('left_shoulder')
            right_s = landmarks.get('right_shoulder')
            left_h = landmarks.get('left_hip')
            right_h = landmarks.get('right_hip')
            if (left_s and right_s and left_h and right_h and
                left_s.get('visibility', 0) >= MIN_VISIBILITY and
                right_s.get('visibility', 0) >= MIN_VISIBILITY and
                left_h.get('visibility', 0) >= MIN_VISIBILITY and
                right_h.get('visibility', 0) >= MIN_VISIBILITY):
                sw = abs(left_s['x'] - right_s['x'])
                hw = abs(left_h['x'] - right_h['x'])
                avg = (sw + hw) / 2
                frame_w = fd['pose']['frame_shape'][1]
                rotation = 90 - (avg / frame_w * 180)
                signals['body_rotation'][i] = max(0, min(90, rotation))

        # Compute wrist velocity (smoothed derivative)
        from scipy.ndimage import gaussian_filter1d
        wx = signals['wrist_x'].copy()
        valid = ~np.isnan(wx)
        if np.sum(valid) > 3:
            # Fill NaNs for derivative
            filled = _fill_nan_linear(wx)
            # Gaussian smooth before derivative
            smoothed = gaussian_filter1d(filled, sigma=1.5, mode='nearest')
            # Central difference
            vel = np.gradient(smoothed)
            signals['wrist_velocity'] = vel

        return signals

    def _detect_cycle_boundaries(self, signals: Dict,
                                  pose_data: List[Dict]) -> List[Tuple[int, int]]:
        """
        Detect start/end frames of each stroke cycle.

        Uses the wrist x trajectory: each forward sweep (recovery) followed by
        backward sweep (pull) = one cycle.
        """
        wx = signals['wrist_x']
        n = len(wx)

        # Find direction changes in wrist x (smoothed)
        from scipy.ndimage import gaussian_filter1d
        valid = ~np.isnan(wx)
        if np.sum(valid) < 10:
            return []

        filled = _fill_nan_linear(wx)
        smoothed = gaussian_filter1d(filled, sigma=2.0, mode='nearest')

        # Detect peaks (local maxima) and troughs (local minima)
        # Peak = wrist furthest forward (recovery start)
        # Trough = wrist furthest back (pull end)

        # Use relative threshold based on overall signal range
        signal_range = np.max(smoothed) - np.min(smoothed)
        rel_threshold = max(MIN_PEAK_DELTA, signal_range * 0.15)  # 15% of range

        # Detect peaks by looking for sign change in gradient (positive -> non-positive)
        # and troughs (negative -> non-negative). This handles plateaus correctly.
        grad = np.gradient(smoothed)
        grad_sign = np.zeros_like(grad)
        grad_sign[grad > rel_threshold * 0.05] = 1
        grad_sign[grad < -rel_threshold * 0.05] = -1

        peaks = []
        troughs = []
        for i in range(1, len(grad_sign)):
            # Peak: positive gradient becomes zero or negative
            if grad_sign[i - 1] > 0 and grad_sign[i] <= 0:
                peaks.append(i - 1)
            # Trough: negative gradient becomes zero or positive
            if grad_sign[i - 1] < 0 and grad_sign[i] >= 0:
                troughs.append(i - 1)

        if len(peaks) < 2 or len(troughs) < 1:
            return []

        # Filter peaks with minimum gap
        filtered_peaks = [peaks[0]]
        for p in peaks[1:]:
            if p - filtered_peaks[-1] >= MIN_PEAK_GAP_FRAMES:
                filtered_peaks.append(p)

        if len(filtered_peaks) < 2:
            return []

        # Each pair of consecutive peaks = one cycle (recovery to recovery)
        cycles_raw = []
        for i in range(len(filtered_peaks) - 1):
            start = filtered_peaks[i]
            end = filtered_peaks[i + 1]
            cycles_raw.append((start, end))

        return cycles_raw

    def _segment_cycle(self, start_frame: int, end_frame: int,
                       signals: Dict, pose_data: List[Dict]) -> Optional[Dict]:
        """
        Segment a single stroke cycle into phases.

        Returns dict with frame indices for each phase transition.
        """
        cycle_frames = list(range(start_frame, end_frame + 1))
        if len(cycle_frames) < 5:
            return None

        cycle = {
            'cycle_number': None,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'entry_frame': None,
            'catch_frame': None,
            'pull_start': None,
            'push_frame': None,
            'exit_frame': None,
            'recovery_start': None,
            'duration_frames': end_frame - start_frame,
        }

        # Extract signals for this cycle
        elbow_angles = signals['elbow_angle'][start_frame:end_frame + 1]
        wrist_y = signals['wrist_y'][start_frame:end_frame + 1]
        wrist_vel = signals['wrist_velocity'][start_frame:end_frame + 1]
        wrist_to_shoulder = signals['wrist_to_shoulder_y'][start_frame:end_frame + 1]

        # --- Find RECOVERY start: wrist velocity peak (hand moving forward) ---
        # Recovery starts at the beginning of the cycle (peak of wrist x)
        cycle['recovery_start'] = 0  # Relative to cycle start

        # --- Find ENTRY: wrist y local maximum (hand lowest in frame, entering water) ---
        valid_y = ~np.isnan(wrist_y)
        if np.sum(valid_y) > 3:
            # Wrist y peak = hand at lowest point = entry
            local_max = self._find_local_max(wrist_y, min_gap=3)
            if local_max is not None:
                cycle['entry_frame'] = start_frame + local_max

        # --- Find CATCH: elbow angle reaches minimum after entry ---
        # After entry, elbow bends to catch position. Catch = minimum elbow angle
        # in the window between entry and mid-cycle.
        if cycle['entry_frame'] is not None:
            entry_rel = cycle['entry_frame'] - start_frame
        else:
            entry_rel = 0

        valid_elbow = ~np.isnan(elbow_angles)
        if np.sum(valid_elbow) > 3:
            # Look for minimum elbow angle after entry
            search_start = max(entry_rel, 0)
            search_end = min(len(elbow_angles), search_start + int(len(elbow_angles) * 0.6))
            window = elbow_angles[search_start:search_end]
            valid_window = ~np.isnan(window)
            if np.sum(valid_window) > 2:
                min_idx = search_start + np.argmin(window[valid_window]) if np.any(valid_window) else search_start
                cycle['catch_frame'] = start_frame + min_idx

        # --- Find PULL: elbow angle reaches maximum (arm extended) ---
        # Pull phase extends from catch to push. The end of pull = max elbow angle
        # (arm straightening) before hand passes hip.
        if cycle['catch_frame'] is not None:
            catch_rel = cycle['catch_frame'] - start_frame
        else:
            catch_rel = entry_rel

        valid_elbow2 = ~np.isnan(elbow_angles)
        if np.sum(valid_elbow2) > 3:
            # Look for maximum elbow angle after catch
            search_start = max(catch_rel + 3, 0)
            search_end = min(len(elbow_angles), len(elbow_angles) - 2)
            if search_end > search_start:
                window = elbow_angles[search_start:search_end]
                valid_win = ~np.isnan(window)
                if np.sum(valid_win) > 2:
                    max_idx = search_start + np.argmax(window[valid_win]) if np.any(valid_win) else search_start
                    cycle['push_frame'] = start_frame + max_idx

        # --- Find EXIT: wrist velocity minimum (hand changes direction) ---
        valid_vel = ~np.isnan(wrist_vel)
        if np.sum(valid_vel) > 3:
            # Exit is when wrist velocity crosses zero (end of pull, start recovery)
            # Find zero crossing after push
            if cycle['push_frame'] is not None:
                push_rel = cycle['push_frame'] - start_frame
            else:
                push_rel = len(wrist_vel) // 2

            # Look for velocity minimum (hand moving backward slows down)
            search_start = max(push_rel, 0)
            search_end = len(wrist_vel)
            window = wrist_vel[search_start:search_end]
            valid_win = ~np.isnan(window)
            if np.sum(valid_win) > 3:
                # Find where velocity crosses from negative to positive
                for i in range(1, len(window)):
                    if (window[i - 1] is not np.nan and window[i] is not np.nan and
                        window[i - 1] < 0 and window[i] >= 0):
                        cycle['exit_frame'] = start_frame + search_start + i
                        break

        # --- Recovery start is at the peak of wrist x (already the cycle start) ---
        cycle['recovery_start'] = start_frame

        # --- Compute phase frame ranges ---
        phases = {}
        if cycle['entry_frame'] is not None:
            phases['entry'] = (start_frame, cycle['entry_frame'])
        if cycle['catch_frame'] is not None:
            phases['catch'] = (cycle['entry_frame'] if cycle['entry_frame'] else start_frame,
                               cycle['catch_frame'])
        if cycle['push_frame'] is not None:
            phases['pull'] = (cycle['catch_frame'] if cycle['catch_frame'] else cycle['entry_frame'],
                              cycle['push_frame'])
        if cycle['exit_frame'] is not None:
            phases['push_exit'] = (cycle['push_frame'] if cycle['push_frame'] else cycle['catch_frame'],
                                   cycle['exit_frame'])
        phases['recovery'] = (cycle['exit_frame'] if cycle['exit_frame'] else end_frame, end_frame)

        cycle['phase_frames'] = phases

        return cycle

    def _find_local_max(self, signal: np.ndarray, min_gap: int = 3) -> Optional[int]:
        """Find index of first local maximum in signal (peak detection)."""
        n = len(signal)
        if n < 3:
            return None

        for i in range(1, n - 1):
            if (not np.isnan(signal[i]) and
                not np.isnan(signal[i - 1]) and
                not np.isnan(signal[i + 1]) and
                signal[i] > signal[i - 1] and signal[i] > signal[i + 1]):
                return i

        # Fall back to global maximum
        valid = ~np.isnan(signal)
        if np.any(valid):
            return np.argmax(signal[valid])
        return None

    @staticmethod
    def _calc_angle(p1: Dict, p2: Dict, p3: Dict) -> float:
        """Angle at p2 in degrees."""
        a = np.array([p1['x'], p1['y']])
        b = np.array([p2['x'], p2['y']])
        c = np.array([p3['x'], p3['y']])
        v1 = a - b
        v2 = c - b
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm < 1e-6:
            return 0.0
        cos = np.dot(v1, v2) / norm
        cos = np.clip(cos, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))


# ── Utility ──

def _fill_nan_linear(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values with linear interpolation (same as pose_smoother)."""
    out = arr.copy()
    n = len(out)
    if n < 2:
        return out

    # Forward fill
    last_valid = None
    for i in range(n):
        if not np.isnan(out[i]):
            last_valid = out[i]
        elif last_valid is not None:
            out[i] = last_valid

    # Backward fill
    last_valid = None
    for i in range(n - 1, -1, -1):
        if not np.isnan(out[i]):
            last_valid = out[i]
        elif last_valid is not None:
            out[i] = last_valid

    # Interpolate
    i = 0
    while i < n:
        if not np.isnan(out[i]):
            j = i + 1
            while j < n and np.isnan(out[j]):
                j += 1
            if j < n:
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
