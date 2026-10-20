"""Base analyzer — shared pipeline for all swim strokes.

Pipeline: pose → clean → canonicalize → segment → metrics → decide

All strokes share:
  - Body-relative normalization (torso length, hip-centred)
  - Direction canonicalization (swimmer always heads right)
  - Near arm / far arm labeling with limb-swap detection
  - Bad-frame rejection (visibility, bone-length, Hampel, Savitzky-Golay)
  - Time-based windows in seconds (not frames)
  - Per-cycle metrics → median across cycles → persistence rule (≥60% cycles)
  - Confidence gating (output "not measurable" when too few valid frames)
  - Continuous severity

Stroke-specific: only cycle signal, phase expectations, kick pattern, metric list.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
from abc import ABC, abstractmethod

from src.stroke_config import StrokeConfig, CONFIGS, SHARED_METRICS


# ── Shared thresholds (tuneable) ──
MIN_VISIBILITY = 0.3
BONE_LENGTH_TOLERANCE = 0.25
HAMPEL_WINDOW = 5
HAMPEL_NSIGMA = 3
SAVITZKY_GOLAY_WINDOW = 7                     # odd, ~0.3 s at 30 fps
TORSO_LENGTH_SMOOTHING = 0.1                  # EMA alpha
PERSISTENCE_THRESHOLD = 0.6                   # fraction of cycles
MIN_VALID_FRAME_RATIO = 0.3
MIN_CYCLES_FOR_METRIC = 3
CONFIDENCE_GATE_ENABLED = True
NEAR_ARM_VISIBILITY_ADVANTAGE = 0.05
SIDE_VIEW_SHOULDER_RATIO_THRESHOLD = 0.6
SYNC_MIN_VISIBLE_FAR_ARM = 0.3

# ── Severity ──
SEVERITY_CRITICAL = "critical"
SEVERITY_MODERATE = "moderate"
SEVERITY_MINOR = "minor"


class BaseIssue:
    """Generic technique issue shared across strokes."""
    def __init__(self, issue_type: str, severity: str, description: str, tip: str,
                 metric_value=None, cycle_frac: float = 1.0):
        self.issue_type = issue_type
        self.severity = severity
        self.description = description
        self.tip = tip
        self.metric_value = metric_value
        self.cycle_frac = cycle_frac   # fraction of cycles affected

    def __repr__(self):
        return f"BaseIssue({self.issue_type}, {self.severity}, {self.cycle_frac:.0%} cycles)"


# ─────────────────────────────────────────────────────────────
# Shared utility functions (static)
# ─────────────────────────────────────────────────────────────

def calculate_angle(p1: Dict, p2: Dict, p3: Dict, use_3d: bool = True) -> float:
    """Angle at p2 in degrees.  Uses 3D by default, 2D for near-arm in side view."""
    if use_3d:
        a = np.array([p1.get('x', 0), p1.get('y', 0), p1.get('z', 0)])
        b = np.array([p2.get('x', 0), p2.get('y', 0), p2.get('z', 0)])
        c = np.array([p3.get('x', 0), p3.get('y', 0), p3.get('z', 0)])
    else:
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


def moving_average(values: List[float], window: int) -> List[float]:
    """Centred moving average."""
    if window < 2 or len(values) < window:
        return list(values)
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def hampel_filter(values: List[Optional[float]], half_window: int = HAMPEL_WINDOW,
                  n_sigma: float = HAMPEL_NSIGMA) -> List[Optional[float]]:
    """Hampel filter — replaces outliers with median of local window."""
    if len(values) < 2 * half_window + 1:
        return list(values)
    filtered = list(values)
    for i in range(len(values)):
        if values[i] is None:
            continue
        lo = max(0, i - half_window)
        hi = min(len(values), i + half_window + 1)
        window = [v for v in values[lo:hi] if v is not None]
        if len(window) < 3:
            continue
        med = np.median(window)
        mad = np.median([abs(v - med) for v in window])
        if mad == 0:
            continue
        if abs(values[i] - med) / mad > n_sigma:
            filtered[i] = med
    return filtered


def savitzky_golay(values: List[float], window: int = SAVITZKY_GOLAY_WINDOW) -> List[float]:
    """Savitzky–Golay smoother.  Window must be odd and >= 3."""
    if window < 3 or window % 2 == 0 or len(values) < window:
        return list(values)
    half = window // 2
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        seg = values[lo:hi]
        if len(seg) < 3:
            smoothed.append(values[i])
            continue
        # Polynomial fit of order 2
        xs = np.arange(len(seg))
        coeffs = np.polyfit(xs, seg, 2)
        smoothed.append(np.polyval(coeffs, len(seg) // 2))
    return smoothed


# ─────────────────────────────────────────────────────────────
# BaseAnalyzer
# ─────────────────────────────────────────────────────────────

class BaseAnalyzer(ABC):
    """Abstract base analyzer — shared pipeline for all strokes.

    Subclasses override:
      - _detect_cycles(frames) → List[(start, end)]
      - _stroke_specific_metrics(frames, cycles, arm_labels) → Dict
    """

    def __init__(self, stroke_name: str):
        self.cfg: StrokeConfig = CONFIGS[stroke_name]
        self.metrics = {}
        self.issues = []
        self._arm_labels = {}
        self._camera_view = 'unknown'

    # ── Public API ──

    def analyze_video(self, pose_data: List[Dict]) -> Dict:
        """Full pipeline: pose → clean → canonicalize → segment → metrics → decide."""
        print(f"\nAnalyzing {self.cfg.name} mechanics...")

        valid_frames = [f for f in pose_data if f['pose'] is not None]
        if not valid_frames:
            return {'error': 'No valid pose data', 'metrics': {}, 'issues': []}
        print(f"Valid frames: {len(valid_frames)}/{len(pose_data)}")

        # 1. Camera perspective
        self._camera_view = self._detect_camera_view(valid_frames)
        print(f"Camera view: {self._camera_view}")

        # 2. Direction canonicalization (flip so swimmer heads right)
        canonical = self._canonicalize_direction(valid_frames)

        # 3. Torso-length normalisation (all modes)
        torso_lengths = self._compute_torso_lengths(canonical)
        normalized = self._normalize_to_torso(canonical, torso_lengths)

        # 4. Bad-frame rejection
        clean = self._filter_bad_frames(normalized)

        # 5. Savitzky–Golay smoothing on landmark coordinates
        smoothed = self._smooth_landmarks(clean)

        # 6. Active swimming filter
        active = self._filter_active_swimming_frames(smoothed)
        if len(active) < len(smoothed):
            print(f"Active swimming frames: {len(active)}/{len(smoothed)}")

        # 7. Near/far arm labeling
        self._arm_labels = self._detect_arm_labels(active)

        # 8. Stroke-cycle segmentation
        cycles = self._detect_cycles(active)
        num_cycles = len(cycles)
        print(f"Cycles detected: {num_cycles}")

        # 9. Stroke rate from cycles + timestamps
        timestamps = [f['timestamp'] for f in active]
        if cycles and len(timestamps) > 1:
            duration = timestamps[-1] - timestamps[0]
            total_strokes = len(cycles) * (2 if self.cfg.name in ('freestyle', 'backstroke') else 1)
            spm = (total_strokes / duration) * 60 if duration > 0 else None
            spm = spm if spm is None or (30 <= spm <= 80) else None
        else:
            spm, total_strokes, duration = None, 0, 0
        stroke_rate_metrics = {
            'spm': spm,
            'total_strokes': total_strokes,
            'duration': duration,
        } if spm is not None else {'spm': None}

        # 10. Shared metrics (computed for every stroke)
        shared = self._compute_shared_metrics(active, cycles, torso_lengths)

        # 11. Stroke-specific metrics
        specific = self._stroke_specific_metrics(active, cycles)

        # 11. Confidence gating
        valid_ratio = len(valid_frames) / len(pose_data)

        self.metrics = {
            'camera_view': self._camera_view,
            'num_cycles': num_cycles,
            'valid_frame_ratio': valid_ratio,
            'stroke_rate': stroke_rate_metrics,
            **shared,
            **specific,
        }

        # 12. Issue detection
        self.issues = self._detect_issues()

        return {'metrics': self.metrics, 'issues': self.issues}

    # ── Shared pipeline steps ──

    @staticmethod
    def _detect_camera_view(frames: List[Dict]) -> str:
        """Detect side-view vs overhead from shoulder-separation ratio."""
        ratios = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            ls = landmarks.get('left_shoulder')
            rs = landmarks.get('right_shoulder')
            lh = landmarks.get('left_hip')
            rh = landmarks.get('right_hip')
            if (ls and rs and lh and rh and
                ls['visibility'] >= MIN_VISIBILITY and rs['visibility'] >= MIN_VISIBILITY):
                sw = abs(ls['x'] - rs['x'])
                hw = abs(lh['x'] - rh['x'])
                torso_len = ((ls['y'] + rs['y']) / 2 - (lh['y'] + rh['y']) / 2)
                if torso_len > 0:
                    ratio = (sw + hw) / (2 * torso_len)
                    ratios.append(ratio)
        if not ratios:
            return 'unknown'
        return 'side' if np.median(ratios) < SIDE_VIEW_SHOULDER_RATIO_THRESHOLD else 'overhead'

    @staticmethod
    def _canonicalize_direction(frames: List[Dict]) -> List[Dict]:
        """Flip x-coordinates so swimmer always heads right."""
        if not frames:
            return frames
        for f in frames:
            landmarks = f['pose']['landmarks']
            nose = landmarks.get('nose')
            lh = landmarks.get('left_hip')
            rh = landmarks.get('right_hip')
            if (nose and lh and rh and
                nose['visibility'] >= MIN_VISIBILITY and
                lh['visibility'] >= MIN_VISIBILITY):
                hip_x = (lh['x'] + rh['x']) / 2
                if nose['x'] < hip_x:
                    return frames
                flipped = []
                for f2 in frames:
                    lm = f2['pose']['landmarks'].copy()
                    for k in lm:
                        lm[k] = lm[k].copy()
                        lm[k]['x'] = 1.0 - lm[k]['x']
                    flipped.append(f2.copy())
                    flipped[-1]['pose'] = f2['pose'].copy()
                    flipped[-1]['pose']['landmarks'] = lm
                return flipped
        return frames

    @staticmethod
    def _compute_torso_lengths(frames: List[Dict]) -> List[Optional[float]]:
        """Torso length per frame (shoulder_mid → hip_mid vertical)."""
        lengths = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            ls = landmarks.get('left_shoulder')
            rs = landmarks.get('right_shoulder')
            lh = landmarks.get('left_hip')
            rh = landmarks.get('right_hip')
            if (ls and rs and lh and rh and
                ls['visibility'] >= MIN_VISIBILITY and rs['visibility'] >= MIN_VISIBILITY and
                lh['visibility'] >= MIN_VISIBILITY and rh['visibility'] >= MIN_VISIBILITY):
                lengths.append(abs((ls['y'] + rs['y']) / 2 - (lh['y'] + rh['y']) / 2))
            else:
                lengths.append(None)
        # EMA smoothing
        smoothed = []
        prev = None
        for v in lengths:
            if v is not None:
                if prev is None:
                    prev = v
                else:
                    prev = prev * (1 - TORSO_LENGTH_SMOOTHING) + v * TORSO_LENGTH_SMOOTHING
                smoothed.append(prev)
            else:
                smoothed.append(None)
        return smoothed

    @staticmethod
    def _normalize_to_torso(frames: List[Dict], torso_lengths: List[float]) -> List[Dict]:
        """Scale y-coordinates by 1/torso_length.  Returns new frame dicts."""
        result = []
        for f, tl in zip(frames, torso_lengths):
            if tl is None or tl <= 0:
                result.append(f)
                continue
            scale = 1.0 / tl
            lm = f['pose']['landmarks'].copy()
            for k in lm:
                lm[k] = lm[k].copy()
                lm[k]['y'] = lm[k]['y'] * scale
            result.append(f.copy())
            result[-1]['pose'] = f['pose'].copy()
            result[-1]['pose']['landmarks'] = lm
        return result

    @staticmethod
    def _compute_bone_length(landmarks: Dict, j1: str, j2: str) -> Optional[float]:
        p1 = landmarks.get(j1)
        p2 = landmarks.get(j2)
        if p1 and p2 and p1['visibility'] >= MIN_VISIBILITY and p2['visibility'] >= MIN_VISIBILITY:
            dx = p1['x'] - p2['x']
            dy = p1['y'] - p2['y']
            return math.sqrt(dx * dx + dy * dy)
        return None

    @staticmethod
    def _filter_bad_frames(frames: List[Dict]) -> List[Dict]:
        """Reject frames where bone lengths deviate >25% from median."""
        if len(frames) < 5:
            return frames
        bones_se = []
        bones_hk = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            se = BaseAnalyzer._compute_bone_length(landmarks, 'left_shoulder', 'left_elbow')
            hk = BaseAnalyzer._compute_bone_length(landmarks, 'left_hip', 'left_knee')
            if se is not None:
                bones_se.append(se)
            if hk is not None:
                bones_hk.append(hk)
        if not bones_se and not bones_hk:
            return frames
        med_se = np.median(bones_se) if bones_se else None
        med_hk = np.median(bones_hk) if bones_hk else None
        good = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            ok = True
            if med_se is not None:
                se = BaseAnalyzer._compute_bone_length(landmarks, 'left_shoulder', 'left_elbow')
                if se is not None and abs(se - med_se) / med_se > BONE_LENGTH_TOLERANCE:
                    ok = False
            if ok and med_hk is not None:
                hk = BaseAnalyzer._compute_bone_length(landmarks, 'left_hip', 'left_knee')
                if hk is not None and abs(hk - med_hk) / med_hk > BONE_LENGTH_TOLERANCE:
                    ok = False
            if ok:
                good.append(f)
        return good

    @staticmethod
    def _smooth_landmarks(frames: List[Dict]) -> List[Dict]:
        """Savitzky–Golay smoothing on (x, y) for each landmark across time."""
        if len(frames) < SAVITZKY_GOLAY_WINDOW:
            return frames

        # Collect all landmark keys
        keys = list(frames[0]['pose']['landmarks'].keys())
        n = len(frames)

        # Build per-key, per-coordinate arrays
        smoothed = {k: {'x': [None]*n, 'y': [None]*n} for k in keys}
        for i, f in enumerate(frames):
            lm = f['pose']['landmarks']
            for k in keys:
                if k in lm:
                    smoothed[k]['x'][i] = lm[k]['x']
                    smoothed[k]['y'][i] = lm[k]['y']

        # Apply SG
        for k in keys:
            xs = smoothed[k]['x']
            ys = smoothed[k]['y']
            valid_x = [v for v in xs if v is not None]
            valid_y = [v for v in ys if v is not None]
            if len(valid_x) >= SAVITZKY_GOLAY_WINDOW:
                smoothed[k]['x'] = savitzky_golay([v if v is not None else 0 for v in xs])
                # Re-insert Nones
                for i in range(n):
                    if xs[i] is None:
                        smoothed[k]['x'][i] = None
            if len(valid_y) >= SAVITZKY_GOLAY_WINDOW:
                smoothed[k]['y'] = savitzky_golay([v if v is not None else 0 for v in ys])
                for i in range(n):
                    if ys[i] is None:
                        smoothed[k]['y'][i] = None

        # Build output frames
        result = []
        for i, f in enumerate(frames):
            lm = f['pose']['landmarks'].copy()
            for k in keys:
                if k in lm:
                    lm[k] = lm[k].copy()
                    if smoothed[k]['x'][i] is not None:
                        lm[k]['x'] = smoothed[k]['x'][i]
                    if smoothed[k]['y'][i] is not None:
                        lm[k]['y'] = smoothed[k]['y'][i]
            result.append(f.copy())
            result[-1]['pose'] = f['pose'].copy()
            result[-1]['pose']['landmarks'] = lm
        return result

    @staticmethod
    def _filter_active_swimming_frames(frames: List[Dict]) -> List[Dict]:
        """Filter non-swimming frames using orientation + confidence + periodicity."""
        if len(frames) < 10:
            return frames
        orientations = []
        confidences = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            ls = landmarks.get('left_shoulder')
            rs = landmarks.get('right_shoulder')
            lh = landmarks.get('left_hip')
            rh = landmarks.get('right_hip')
            if (ls and rs and lh and rh and
                ls['visibility'] >= MIN_VISIBILITY and rs['visibility'] >= MIN_VISIBILITY and
                lh['visibility'] >= MIN_VISIBILITY and rh['visibility'] >= MIN_VISIBILITY):
                sx = (ls['x'] + rs['x']) / 2
                sy = (ls['y'] + rs['y']) / 2
                hx = (lh['x'] + rh['x']) / 2
                hy = (lh['y'] + rh['y']) / 2
                angle = abs(math.degrees(math.atan2(hy - sy, hx - sx)))
                orientations.append(angle)
            else:
                orientations.append(None)
            vis = [landmarks.get(k, {}).get('visibility', 0)
                   for k in ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']]
            confidences.append(np.mean(vis) if vis else 0)

        # Wrist-x periodicity via autocorrelation
        wrist_x = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            lw = landmarks.get('left_wrist')
            rw = landmarks.get('right_wrist')
            if lw and lw['visibility'] >= MIN_VISIBILITY:
                wrist_x.append(lw['x'])
            elif rw and rw['visibility'] >= MIN_VISIBILITY:
                wrist_x.append(rw['x'])
            else:
                wrist_x.append(None)
        valid_x = [v for v in wrist_x if v is not None]
        periodicity = 0
        if len(valid_x) > 20:
            mean_x = np.mean(valid_x)
            centered = [v - mean_x for v in valid_x]
            acorr = np.correlate(centered, centered, mode='full')
            mid = len(acorr) // 2
            left = acorr[mid + 1: mid + len(valid_x) // 2]
            if len(left) > 0:
                periodicity = np.max(left) / acorr[mid] if acorr[mid] > 0 else 0

        swimming = []
        for angle, conf in zip(orientations, confidences):
            if angle is not None and angle < 40 and conf > 0.5:
                swimming.append(True)
            else:
                swimming.append(False)

        if periodicity > 0.3 and sum(swimming) < len(frames) * 0.3:
            for i in range(len(swimming)):
                if orientations[i] is not None and orientations[i] < 60 and confidences[i] > 0.4:
                    swimming[i] = True

        # Turn detection (direction reversal)
        nose_x = []
        hip_x = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            n = landmarks.get('nose')
            lh = landmarks.get('left_hip')
            rh = landmarks.get('right_hip')
            if n and lh and rh and n['visibility'] >= MIN_VISIBILITY:
                nose_x.append(n['x'])
                hip_x.append((lh['x'] + rh['x']) / 2)
            else:
                nose_x.append(None)
                hip_x.append(None)
        for i in range(1, len(nose_x)):
            if nose_x[i] is not None and nose_x[i-1] is not None and hip_x[i] is not None:
                if (nose_x[i-1] - hip_x[i]) * (nose_x[i] - hip_x[i]) < 0:
                    for j in range(max(0, i - 5), min(len(swimming), i + 5)):
                        swimming[j] = False

        return [f for i, f in enumerate(frames) if swimming[i]]

    @staticmethod
    def _detect_arm_labels(frames: List[Dict]) -> Dict:
        """Detect near arm / far arm and detect swaps."""
        left_vis = [f['pose']['landmarks']['left_wrist']['visibility']
                    for f in frames if f['pose']['landmarks']['left_wrist']['visibility'] >= MIN_VISIBILITY]
        right_vis = [f['pose']['landmarks']['right_wrist']['visibility']
                     for f in frames if f['pose']['landmarks']['right_wrist']['visibility'] >= MIN_VISIBILITY]
        avg_l = np.mean(left_vis) if left_vis else 0
        avg_r = np.mean(right_vis) if right_vis else 0
        if avg_l > avg_r + NEAR_ARM_VISIBILITY_ADVANTAGE:
            return {'near_arm': 'left', 'far_arm': 'right'}
        elif avg_r > avg_l + NEAR_ARM_VISIBILITY_ADVANTAGE:
            return {'near_arm': 'right', 'far_arm': 'left'}
        # Fall back to wrist-x variance
        left_x = [f['pose']['landmarks']['left_wrist']['x']
                  for f in frames if f['pose']['landmarks']['left_wrist']['visibility'] >= MIN_VISIBILITY]
        right_x = [f['pose']['landmarks']['right_wrist']['x']
                   for f in frames if f['pose']['landmarks']['right_wrist']['visibility'] >= MIN_VISIBILITY]
        var_l = np.var(left_x) if left_x else 0
        var_r = np.var(right_x) if right_x else 0
        return {'near_arm': 'left' if var_l > var_r else 'right',
                'far_arm': 'right' if var_l > var_r else 'left'}

    # ── Cycle detection (stroke-specific) ──

    @abstractmethod
    def _detect_cycles(self, frames: List[Dict]) -> List[Tuple[int, int]]:
        """Detect stroke cycles from frame data.  Returns [(start, end), ...]."""
        pass

    # ── Shared metrics ──

    def _compute_shared_metrics(self, frames: List[Dict], cycles: List[Tuple[int, int]],
                                torso_lengths: List[float]) -> Dict:
        """Compute metrics shared across all strokes."""
        results = {}
        near_arm = self._arm_labels.get('near_arm', 'left')

        # ── Elbow min during pull (2D, near arm) ──
        results['elbow_min_pull'] = self._metric_elbow_min_pull(frames, cycles, near_arm)

        # ── Arm phase via cross-correlation ──
        results['arm_phase'] = self._metric_arm_phase(frames)

        # ── Hip line (shoulder→hip→knee angle vs horizontal) ──
        results['hip_line'] = self._metric_hip_line(frames, torso_lengths)

        # ── Kick ──
        results['kick'] = self._metric_kick(frames)

        # ── Recovery height ──
        results['recovery_height'] = self._metric_recovery_height(frames, torso_lengths)

        # ── Breath lift ──
        results['breath_lift'] = self._metric_breath_lift(frames, torso_lengths)

        return results

    # ── Individual metric implementations ──

    def _metric_elbow_min_pull(self, frames: List[Dict], cycles: List[Tuple[int, int]],
                                near_arm: str) -> Dict:
        """Minimum elbow angle during the pull phase (2D, near arm).

        Uses per-cycle pull-phase minimum, then median across cycles.
        """
        ELBOW_MIN = 30.0
        ELBOW_MAX = 180.0

        if not cycles:
            # Fallback: overall minimum from near arm
            angles = []
            for f in frames:
                landmarks = f['pose']['landmarks']
                sh = landmarks[f'{near_arm}_shoulder']
                el = landmarks[f'{near_arm}_elbow']
                wr = landmarks[f'{near_arm}_wrist']
                if (sh['visibility'] >= MIN_VISIBILITY and
                    el['visibility'] >= MIN_VISIBILITY and
                    wr['visibility'] >= MIN_VISIBILITY):
                    a = calculate_angle(sh, el, wr, use_3d=False)
                    if ELBOW_MIN <= a <= ELBOW_MAX:
                        angles.append(a)
            return {
                'pull_min': np.min(angles) if angles else None,
                'per_cycle': [],
                'num_cycles': 0,
            }

        per_cycle = []
        for start, end in cycles:
            pull_start = start + 3
            pull_end = min(end - 3, len(frames) - 1)
            angles = []
            for i in range(pull_start, pull_end):
                if i >= len(frames):
                    break
                landmarks = frames[i]['pose']['landmarks']
                sh = landmarks[f'{near_arm}_shoulder']
                el = landmarks[f'{near_arm}_elbow']
                wr = landmarks[f'{near_arm}_wrist']
                if (sh['visibility'] >= MIN_VISIBILITY and
                    el['visibility'] >= MIN_VISIBILITY and
                    wr['visibility'] >= MIN_VISIBILITY):
                    a = calculate_angle(sh, el, wr, use_3d=False)
                    if ELBOW_MIN <= a <= ELBOW_MAX:
                        angles.append(a)
            if angles:
                per_cycle.append(np.min(angles))

        if not per_cycle:
            return {'pull_min': None, 'per_cycle': [], 'num_cycles': 0}

        return {
            'pull_min': np.median(per_cycle),
            'per_cycle': per_cycle,
            'num_cycles': len(per_cycle),
        }

    def _metric_arm_phase(self, frames: List[Dict]) -> Dict:
        """L/R arm phase via cross-correlation of wrist-x signals.

        Near 0.5 = windmill/alternating (freestyle, backstroke).
        Near 0.0 = in phase (butterfly, breaststroke).
        """
        if len(frames) < 10:
            return {'phase_gap': None, 'timing_ok': None}

        near_arm = self._arm_labels.get('near_arm', 'left')
        far_arm = 'right' if near_arm == 'left' else 'left'

        # Check far arm visibility
        far_vis = [f['pose']['landmarks'][f'{far_arm}_wrist']['visibility']
                   for f in frames]
        far_ratio = sum(1 for v in far_vis if v >= SYNC_MIN_VISIBLE_FAR_ARM) / len(far_vis)
        if far_ratio < 0.3:
            return {'phase_gap': None, 'timing_ok': None,
                     'note': 'Far arm not visible enough'}

        # Extract signals
        def _extract(arm):
            sig = []
            for f in frames:
                w = f['pose']['landmarks'][f'{arm}_wrist']
                sig.append(w['x'] if w['visibility'] >= MIN_VISIBILITY else None)
            return sig

        near = _extract(near_arm)
        far = _extract(far_arm)

        # Interpolate
        def _interp(arr):
            result = list(arr)
            valid = [(i, v) for i, v in enumerate(arr) if v is not None]
            if len(valid) < 2:
                return result
            for i in range(len(result)):
                if result[i] is None:
                    before = [v for v in valid if v[0] < i]
                    after = [v for v in valid if v[0] > i]
                    if before and after:
                        b, a = before[-1], after[0]
                        result[i] = b[1] + (a[1] - b[1]) * (i - b[0]) / (a[0] - b[0])
                    elif before:
                        result[i] = before[-1][1]
                    elif after:
                        result[i] = after[0][1]
            return result

        near_i = _interp(near)
        far_i = _interp(far)

        # Cross-correlation
        near_d = np.array(near_i) - np.mean(near_i)
        far_d = np.array(far_i) - np.mean(far_i)
        n = len(near_d)
        max_lag = n // 4
        best_lag, best_corr = 0, -1
        for lag in range(1, max_lag):
            if lag >= n:
                break
            corr = np.corrcoef(near_d[:-lag], far_d[lag:])[0, 1]
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr = abs(corr)
                best_lag = lag

        if best_corr < 0.3:
            return {'phase_gap': None, 'timing_ok': None, 'note': 'Low correlation'}

        # Estimate cycle length from FFT of near arm
        timestamps = [f['timestamp'] for f in frames]
        fps = len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 30
        fft = np.fft.rfft(near_d)
        freqs = np.fft.rfftfreq(len(near_d)) * fps
        valid = np.where((freqs >= 0.3) & (freqs <= 1.5))[0]
        if len(valid) == 0:
            cycle_len = n // 2
        else:
            mags = np.abs(fft[valid])
            peak_hz = freqs[valid[np.argmax(mags)]]
            cycle_len = fps / peak_hz if peak_hz > 0 else n // 2

        phase_gap = best_lag / cycle_len
        exp_min, exp_max = self.cfg.expected_arm_phase_range
        timing_ok = exp_min <= phase_gap <= exp_max

        return {
            'phase_gap': float(phase_gap),
            'timing_ok': timing_ok,
            'cross_correlation': float(best_corr),
            'expected_range': (exp_min, exp_max),
        }

    @staticmethod
    def _metric_hip_line(frames: List[Dict], torso_lengths: List[float]) -> Dict:
        """Body line via shoulder→hip→knee angle vs horizontal."""
        if len(frames) < 5:
            return {'deviation': None, 'hip_drop': None}
        angles = []
        hip_rel = []
        for i, f in enumerate(frames):
            landmarks = f['pose']['landmarks']
            ls = landmarks.get('left_shoulder')
            rs = landmarks.get('right_shoulder')
            lh = landmarks.get('left_hip')
            rh = landmarks.get('right_hip')
            lk = landmarks.get('left_knee')
            rk = landmarks.get('right_knee')
            tl = torso_lengths[i] if i < len(torso_lengths) and torso_lengths[i] is not None else None
            if tl is None or tl <= 0:
                continue
            if (ls and rs and lh and rh and lk and rk and
                ls['visibility'] >= MIN_VISIBILITY and rs['visibility'] >= MIN_VISIBILITY and
                lh['visibility'] >= MIN_VISIBILITY and rh['visibility'] >= MIN_VISIBILITY and
                lk['visibility'] >= MIN_VISIBILITY and rk['visibility'] >= MIN_VISIBILITY):
                sx = (ls['x'] + rs['x']) / 2
                sy = (ls['y'] + rs['y']) / 2
                hx = (lh['x'] + rh['x']) / 2
                hy = (lh['y'] + rh['y']) / 2
                kx = (lk['x'] + rk['x']) / 2
                ky = (lk['y'] + rk['y']) / 2
                vec_sh = np.array([hx - sx, hy - sy])
                vec_hk = np.array([kx - hx, ky - hy])
                norm = np.linalg.norm(vec_sh) * np.linalg.norm(vec_hk)
                if norm > 0:
                    cos_a = np.dot(vec_sh, vec_hk) / norm
                    cos_a = np.clip(cos_a, -1.0, 1.0)
                    dev = abs(180 - float(np.degrees(np.arccos(cos_a))))
                    angles.append(dev)
            shoulder_y = (ls['y'] + rs['y']) / 2
            hip_y = (lh['y'] + rh['y']) / 2
            hip_rel.append((hip_y - shoulder_y) / tl)

        return {
            'deviation': np.mean(angles) if angles else None,
            'hip_drop': np.mean(hip_rel) if hip_rel else None,
        }

    @staticmethod
    def _metric_kick(frames: List[Dict]) -> Dict:
        """Knee angle range + ankle amplitude (near leg only, visibility >= 0.5)."""
        KNEE_MIN = 100.0
        KNEE_MAX = 180.0
        KNEE_VIS_GATE = 0.5

        angles = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            for side in ['left', 'right']:
                hip = landmarks[f'{side}_hip']
                knee = landmarks[f'{side}_knee']
                ankle = landmarks[f'{side}_ankle']
                if (hip['visibility'] >= KNEE_VIS_GATE and
                    knee['visibility'] >= KNEE_VIS_GATE and
                    ankle['visibility'] >= KNEE_VIS_GATE):
                    a = calculate_angle(hip, knee, ankle)
                    if KNEE_MIN <= a <= KNEE_MAX:
                        angles.append(a)

        angles = hampel_filter(angles)
        angles = [v for v in angles if v is not None]

        return {
            'avg_knee_angle': np.mean(angles) if angles else None,
            'min_knee_angle': np.min(angles) if angles else None,
        }

    @staticmethod
    def _metric_recovery_height(frames: List[Dict], torso_lengths: List[float]) -> Dict:
        """Wrist clearance above shoulder (torso-normalised)."""
        if len(frames) < 3:
            return {'avg': None}
        clearances = []
        for i, f in enumerate(frames):
            landmarks = f['pose']['landmarks']
            tl = torso_lengths[i] if i < len(torso_lengths) and torso_lengths[i] is not None else None
            if tl is None or tl <= 0:
                continue
            for side in ['left', 'right']:
                sh = landmarks[f'{side}_shoulder']
                wr = landmarks[f'{side}_wrist']
                if sh['visibility'] >= MIN_VISIBILITY and wr['visibility'] >= MIN_VISIBILITY:
                    c = (sh['y'] - wr['y']) / tl
                    if c > 0:   # recovery phase (wrist above shoulder)
                        clearances.append(c)

        return {
            'avg': np.mean(clearances) if clearances else None,
            'max': np.max(clearances) if clearances else None,
            'min': np.min(clearances) if clearances else None,
        }

    @staticmethod
    def _metric_breath_lift(frames: List[Dict], torso_lengths: List[float]) -> Dict:
        """Head lift: nose-y relative to shoulder line, torso-normalised."""
        if not frames:
            return {'lift': None, 'stability': None}
        nose_rel = []
        for i, f in enumerate(frames):
            landmarks = f['pose']['landmarks']
            nose = landmarks.get('nose')
            ls = landmarks.get('left_shoulder')
            rs = landmarks.get('right_shoulder')
            tl = torso_lengths[i] if i < len(torso_lengths) and torso_lengths[i] is not None else None
            if (nose and ls and rs and
                nose['visibility'] >= MIN_VISIBILITY and
                ls['visibility'] >= MIN_VISIBILITY and rs['visibility'] >= MIN_VISIBILITY and
                tl is not None and tl > 0):
                sh_y = (ls['y'] + rs['y']) / 2
                nose_rel.append((nose['y'] - sh_y) / tl)

        if not nose_rel:
            return {'lift': None, 'stability': None}

        window = max(3, len(nose_rel) // 10)
        filtered = moving_average(nose_rel, window)
        y_range = np.max(filtered) - np.min(filtered)
        return {
            'lift': np.mean(nose_rel),
            'stability': 1.0 - y_range,
        }

    # ── Stroke-specific metrics (override in subclass) ──

    def _stroke_specific_metrics(self, frames: List[Dict], cycles: List[Tuple[int, int]]) -> Dict:
        """Override this for stroke-specific metrics beyond the shared ones."""
        return {}

    # ── Issue detection ──

    def _detect_issues(self) -> List[BaseIssue]:
        """Generic issue detection with persistence and confidence gating."""
        issues = []
        metrics = self.metrics
        cfg = self.cfg

        valid_ratio = metrics.get('valid_frame_ratio', 0)
        if CONFIDENCE_GATE_ENABLED and valid_ratio < MIN_VALID_FRAME_RATIO:
            return [BaseIssue('low_confidence', SEVERITY_MODERATE,
                              f'Only {valid_ratio:.1%} valid frames — results may be unreliable',
                              'Improve lighting and camera position')]

        # ── Elbow dropped ──
        elbow = metrics.get('elbow_min_pull', {})
        if elbow.get('pull_min') is not None:
            pull_min = elbow['pull_min']
            dropped_thresh = cfg.thresholds.get('elbow_dropped', 150)
            per_cycle = elbow.get('per_cycle', [])
            if per_cycle:
                bad = sum(1 for v in per_cycle if v > dropped_thresh)
                frac = bad / len(per_cycle)
                if frac >= PERSISTENCE_THRESHOLD:
                    issues.append(BaseIssue('dropped_elbow', SEVERITY_CRITICAL,
                        f"Dropped elbow (pull min {pull_min:.0f}°, {frac:.0%} cycles > {dropped_thresh:.0f}°)",
                        'Keep elbow bent during pull (~90-120°)', pull_min, frac))

        # ── Arm phase ──
        phase = metrics.get('arm_phase', {})
        if phase.get('phase_gap') is not None and not phase.get('timing_ok', True):
            gap = phase['phase_gap']
            exp_min, exp_max = cfg.expected_arm_phase_range
            issues.append(BaseIssue('poor_arm_timing', SEVERITY_CRITICAL,
                f"Arm timing off (phase gap {gap:.2f}, expected {exp_min:.2f}-{exp_max:.2f})",
                'Adjust arm rhythm', gap))

        # ── Hip line ──
        hip = metrics.get('hip_line', {})
        if hip.get('deviation') is not None:
            dev = hip['deviation']
            max_dev = cfg.thresholds.get('hip_line_drop_max', 15)
            if dev > max_dev:
                issues.append(BaseIssue('hips_drop', SEVERITY_CRITICAL,
                    f"Hips drop (body-line deviation {dev:.1f}°)",
                    'Keep hips high — engage core', dev))

        # ── Kick ──
        kick = metrics.get('kick', {})
        if kick.get('avg_knee_angle') is not None:
            avg_knee = kick['avg_knee_angle']
            knee_min = cfg.thresholds.get('knee_bend_min', 120)
            knee_max = cfg.thresholds.get('knee_bend_max', 170)
            if avg_knee < knee_min:
                issues.append(BaseIssue('excessive_knee_bend', SEVERITY_MODERATE,
                    f"Excessive knee bend (avg {avg_knee:.0f}°)",
                    'Kick from hips, not knees', avg_knee))
            elif avg_knee > knee_max:
                issues.append(BaseIssue('knee_too_straight', SEVERITY_MODERATE,
                    f"Knees too straight (avg {avg_knee:.0f}°)",
                    'Allow natural knee bend', avg_knee))

        # ── Recovery height ──
        rec = metrics.get('recovery_height', {})
        if rec.get('avg') is not None:
            avg_rec = rec['avg']
            rec_max = cfg.thresholds.get('recovery_height_max', 0.20)
            if avg_rec > rec_max:
                issues.append(BaseIssue('recovery_too_high', SEVERITY_MODERATE,
                    f"Recovery too high ({avg_rec:.2f}x torso length)",
                    'Skim the water', avg_rec))
            elif rec.get('min') is not None and rec['min'] < cfg.thresholds.get('recovery_height_min', 0.03):
                issues.append(BaseIssue('recovery_too_low', SEVERITY_MODERATE,
                    f"Recovery too low — hand may hit water (min {rec['min']:.2f})",
                    'Raise arm slightly', rec['min']))

        # ── Breath lift ──
        breath = metrics.get('breath_lift', {})
        if breath.get('lift') is not None:
            lift = breath['lift']
            max_lift = cfg.thresholds.get('breath_lift_max', 0.08)
            # Positive lift = nose above shoulders (good for backstroke, bad for others)
            if self.cfg.name != 'backstroke' and abs(lift) > max_lift:
                issues.append(BaseIssue('head_lifting_excessive', SEVERITY_MODERATE,
                    f"Head lift {lift:.3f}x torso length",
                    'Tuck chin, rotate to breathe', lift))

        # ── Stroke rate (from cycle detection + duration) ──
        num_cycles = metrics.get('num_cycles', 0)
        valid_ratio = metrics.get('valid_frame_ratio', 0)
        # Stroke rate is computed in analyze_video from cycles + timestamps
        # and stored in metrics['stroke_rate']
        sr = metrics.get('stroke_rate', {})
        if sr.get('spm') is not None:
            spm = sr['spm']
            slow = cfg.thresholds.get('stroke_rate_slow', 35)
            fast = cfg.thresholds.get('stroke_rate_fast', 70)
            if spm < slow:
                issues.append(BaseIssue('slow_stroke_rate', SEVERITY_MINOR,
                    f'Stroke rate too slow ({spm:.0f} SPM)',
                    'Increase tempo', spm))
            elif spm > fast:
                issues.append(BaseIssue('fast_stroke_rate', SEVERITY_MINOR,
                    f'Stroke rate too fast ({spm:.0f} SPM)',
                    'Focus on longer pulls', spm))

        # Sort by severity
        sev_order = {SEVERITY_CRITICAL: 0, SEVERITY_MODERATE: 1, SEVERITY_MINOR: 2}
        issues.sort(key=lambda x: sev_order.get(x.severity, 99))
        return issues
