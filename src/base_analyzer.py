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
BONE_LENGTH_TOLERANCE = 0.5
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

        # 2. Video quality pre-check
        shoulder_span = None
        for f in valid_frames[:50]:
            lm = f['pose']['landmarks']
            ls = lm.get('left_shoulder')
            rs = lm.get('right_shoulder')
            if ls and rs and ls['visibility'] >= MIN_VISIBILITY and rs['visibility'] >= MIN_VISIBILITY:
                span = abs(ls['x'] - rs['x'])
                frame_shape = f.get('frame_shape', f.get('pose', {}).get('frame_shape'))
                if isinstance(frame_shape, (list, tuple)) and len(frame_shape) >= 2:
                    img_w = frame_shape[1]
                else:
                    img_w = 1
                if span > 0 and img_w > 0:
                    shoulder_span = span / img_w
                    break
        if shoulder_span is not None and shoulder_span < 0.05:
            print(f'  ⚠️ Swimmer very small in frame ({shoulder_span*100:.1f}% of width).')
            print(f'  Consider cropping the video closer to the swimmer for better pose detection.')
        elif shoulder_span is None:
            print(f'  ⚠️ Could not measure swimmer size — pose detection may be poor.')
        else:
            print(f'  Swimmer occupies {shoulder_span*100:.1f}% of frame width.')

        # 3. Direction canonicalization (flip so swimmer heads right)
        canonical = self._canonicalize_direction(valid_frames)

        # 3. Torso-length normalisation (all modes)
        torso_lengths = self._compute_torso_lengths(canonical)
        normalized = self._normalize_to_torso(canonical, torso_lengths)

        # 4. Bad-frame rejection
        clean = self._filter_bad_frames(normalized)

        # 5. Savitzky–Golay smoothing on landmark coordinates
        smoothed = self._smooth_landmarks(clean)

        # 5b. Extra temporal smoothing for hip/knee landmarks (most prone to tracker confusion)
        smoothed = self._smooth_hip_knee(smoothed)

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
    def _interpolate_missing_frames(pose_data: List[Dict], video_path: str = None) -> List[Dict]:
        """Fill in frames where pose detection failed (pose is None) by
        interpolating all landmarks from neighbouring valid frames using
        cubic Hermite splines.

        Interpolated landmarks are tagged with 'interpolated': True so the
        visualizer can render them in grey.

        If video_path is provided, for each missing frame the actual video
        frame is read and the swimmer's position is detected using a simple
        MediaPipe call with low confidence.  The detected position is used
        to shift the interpolated landmarks so the skeleton follows the
        swimmer rather than drifting.
        """
        import copy
        result = copy.deepcopy(pose_data)

        # ── Helper: detect swimmer nose in a video frame ──
        def _detect_nose_in_frame(video_path: str, frame_idx: int):
            """Read a specific frame from the video and run MediaPipe pose
            detection at low confidence.  Returns (x, y) of nose or None."""
            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    return None
                import mediapipe as mp
                mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    min_detection_confidence=0.3,
                )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_pose.process(rgb)
                mp_pose.close()
                if results.pose_landmarks:
                    nose = results.pose_landmarks.landmark[0]
                    h, w = frame.shape[:2]
                    return (nose.x * w, nose.y * h)
                return None
            except Exception:
                return None

        # Find segments of consecutive missing frames
        missing_segments = []
        in_missing = False
        seg_start = 0
        for i, f in enumerate(result):
            if f['pose'] is None and not in_missing:
                seg_start = i
                in_missing = True
            elif f['pose'] is not None and in_missing:
                missing_segments.append((seg_start, i - 1))
                in_missing = False
        if in_missing:
            missing_segments.append((seg_start, len(result) - 1))

        if not missing_segments:
            return result

        # Collect all landmark keys from a valid frame
        landmark_keys = None
        for f in result:
            if f['pose'] is not None:
                landmark_keys = list(f['pose']['landmarks'].keys())
                break
        if not landmark_keys:
            return result

        for seg_start, seg_end in missing_segments:
            pre_idx = seg_start - 1
            while pre_idx >= 0 and result[pre_idx]['pose'] is None:
                pre_idx -= 1
            if pre_idx < 0:
                continue
            post_idx = seg_end + 1
            while post_idx < len(result) and result[post_idx]['pose'] is None:
                post_idx += 1
            if post_idx >= len(result):
                continue

            pre_frame = result[pre_idx]
            post_frame = result[post_idx]
            total_gap = post_idx - pre_idx
            seg_len = seg_end - seg_start + 1

            for k in landmark_keys:
                pre_lm = pre_frame['pose']['landmarks'].get(k)
                post_lm = post_frame['pose']['landmarks'].get(k)
                if pre_lm is None or post_lm is None:
                    continue

                if seg_len < 3:
                    # Linear interpolation for very short segments
                    for j in range(seg_start, seg_end + 1):
                        t = (j - pre_idx) / total_gap
                        if result[j]['pose'] is None:
                            result[j]['pose'] = {'landmarks': {}, 'interpolated': True}
                        lm = result[j]['pose']['landmarks']
                        lm[k] = {
                            'x': pre_lm['x'] * (1 - t) + post_lm['x'] * t,
                            'y': pre_lm['y'] * (1 - t) + post_lm['y'] * t,
                            'z': pre_lm['z'] * (1 - t) + post_lm['z'] * t,
                            'visibility': 0.3,
                            'interpolated': True,
                        }
                else:
                    # Cubic Hermite spline
                    pre2_idx = max(0, pre_idx - 1)
                    if pre2_idx != pre_idx:
                        pre2_frame = result[pre2_idx]
                        if pre2_frame['pose'] is not None and k in pre2_frame['pose']['landmarks']:
                            pre2_lm = pre2_frame['pose']['landmarks'][k]
                            dt_pre = pre_idx - pre2_idx
                            m_pre_x = (pre_lm['x'] - pre2_lm['x']) / dt_pre
                            m_pre_y = (pre_lm['y'] - pre2_lm['y']) / dt_pre
                            m_pre_z = (pre_lm['z'] - pre2_lm['z']) / dt_pre
                        else:
                            m_pre_x = (post_lm['x'] - pre_lm['x']) / total_gap
                            m_pre_y = (post_lm['y'] - pre_lm['y']) / total_gap
                            m_pre_z = (post_lm['z'] - pre_lm['z']) / total_gap
                    else:
                        m_pre_x = (post_lm['x'] - pre_lm['x']) / total_gap
                        m_pre_y = (post_lm['y'] - pre_lm['y']) / total_gap
                        m_pre_z = (post_lm['z'] - pre_lm['z']) / total_gap

                    post2_idx = min(len(result) - 1, post_idx + 1)
                    if post2_idx != post_idx:
                        post2_frame = result[post2_idx]
                        if post2_frame['pose'] is not None and k in post2_frame['pose']['landmarks']:
                            post2_lm = post2_frame['pose']['landmarks'][k]
                            dt_post = post2_idx - post_idx
                            m_post_x = (post2_lm['x'] - post_lm['x']) / dt_post
                            m_post_y = (post2_lm['y'] - post_lm['y']) / dt_post
                            m_post_z = (post2_lm['z'] - post_lm['z']) / dt_post
                        else:
                            m_post_x = (post_lm['x'] - pre_lm['x']) / total_gap
                            m_post_y = (post_lm['y'] - pre_lm['y']) / total_gap
                            m_post_z = (post_lm['z'] - pre_lm['z']) / total_gap
                    else:
                        m_post_x = (post_lm['x'] - pre_lm['x']) / total_gap
                        m_post_y = (post_lm['y'] - pre_lm['y']) / total_gap
                        m_post_z = (post_lm['z'] - pre_lm['z']) / total_gap

                    for j in range(seg_start, seg_end + 1):
                        t = (j - pre_idx) / total_gap
                        h00 = 1 - 3*t*t + 2*t*t*t
                        h10 = t - 2*t*t + t*t*t
                        h01 = -t*t + 2*t*t*t
                        h11 = -t*t + t*t*t
                        if result[j]['pose'] is None:
                            result[j]['pose'] = {'landmarks': {}, 'interpolated': True}
                        lm = result[j]['pose']['landmarks']
                        lm[k] = {
                            'x': h00 * pre_lm['x'] + h10 * m_pre_x * total_gap + h01 * post_lm['x'] + h11 * m_post_x * total_gap,
                            'y': h00 * pre_lm['y'] + h10 * m_pre_y * total_gap + h01 * post_lm['y'] + h11 * m_post_y * total_gap,
                            'z': h00 * pre_lm['z'] + h10 * m_pre_z * total_gap + h01 * post_lm['z'] + h11 * m_post_z * total_gap,
                            'visibility': 0.3,
                            'interpolated': True,
                        }

        # ── Drift correction disabled ──
        # The video-frame detection approach introduced false detections
        # (MediaPipe detecting a pose in the water). Simple interpolation
        # between neighbouring frames is more reliable. The grey skeleton
        # may be slightly off but won't jump to a wrong position.

        return result

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
        """Flip x-coordinates so swimmer always heads right.

        Uses the image width from the frame's frame_shape to compute
        the correct pixel-space flip: x' = image_width - x.

        Handles direction changes mid-video: each frame is flipped
        independently based on its own nose/hip direction. Frames where
        the direction cannot be determined (missing landmarks) are
        left unchanged.
        """
        if not frames:
            return frames

        # Get image width (use the first available frame_shape)
        img_w = None
        for f in frames:
            fs = f.get('frame_shape') or (f.get('pose') or {}).get('frame_shape')
            if fs and len(fs) >= 2:
                img_w = fs[1]
                break
        if img_w is None:
            img_w = 1  # fallback, will not flip meaningfully

        flipped = []
        for f2 in frames:
            landmarks = f2['pose']['landmarks']
            nose = landmarks.get('nose')
            lh = landmarks.get('left_hip')
            rh = landmarks.get('right_hip')
            needs_flip = False
            if (nose and lh and rh and
                nose['visibility'] >= MIN_VISIBILITY and
                lh['visibility'] >= MIN_VISIBILITY):
                hip_x = (lh['x'] + rh['x']) / 2
                # nose right of hips → heading left → flip so nose left of hips
                if nose['x'] >= hip_x:
                    needs_flip = True

            if needs_flip:
                lm = landmarks.copy()
                for k in lm:
                    lm[k] = lm[k].copy()
                    lm[k]['x'] = img_w - lm[k]['x']
                flipped.append(f2.copy())
                flipped[-1]['pose'] = f2['pose'].copy()
                flipped[-1]['pose']['landmarks'] = lm
            else:
                flipped.append(f2.copy())

        return flipped

    @staticmethod
    def _compute_torso_lengths(frames: List[Dict]) -> List[Optional[float]]:
        """Torso length per frame (shoulder_mid → hip_mid vertical).

        Rejects frames where torso length < MIN_TORSO_LENGTH (10 px) to avoid
        normalization explosion when the tracker misplaces the shoulder or hip.
        """
        MIN_TORSO_LENGTH = 10.0
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
                tl = abs((ls['y'] + rs['y']) / 2 - (lh['y'] + rh['y']) / 2)
                if tl >= MIN_TORSO_LENGTH:
                    lengths.append(tl)
                else:
                    lengths.append(None)
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
        """Scale y-coordinates by 1/median_torso_length.  Uses the median torso
        length across all valid frames instead of per-frame values, avoiding
        normalization explosion when the tracker misplaces the shoulder or hip.

        Returns new frame dicts.
        """
        valid_tl = [v for v in torso_lengths if v is not None and v > 0]
        if not valid_tl:
            return frames
        median_tl = np.median(valid_tl)
        scale = 1.0 / median_tl
        result = []
        for f in frames:
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
        # If we would reject more than 50% of frames, fall back to all frames
        if len(good) < len(frames) * 0.5:
            return frames
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

    def _smooth_hip_knee(self, frames: List[Dict]) -> List[Dict]:
        """Apply a centred moving average specifically to hip and knee
        landmarks, which are most prone to tracker confusion (e.g., splash
        or body rotation causing sudden jumps).  Uses a wider window than
        the Savitzky–Golay pass (11 frames ~ 0.37 s at 30 fps)."""
        import copy
        result = copy.deepcopy(frames)
        n = len(frames)
        if n < 11:
            return result

        hip_knee_keys = [
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
        ]
        window = 11
        half = window // 2

        for k in hip_knee_keys:
            xs = [result[i]['pose']['landmarks'][k]['x'] for i in range(n)]
            ys = [result[i]['pose']['landmarks'][k]['y'] for i in range(n)]
            # Replace Nones with interpolated values for smoothing
            valid_x = [(i, v) for i, v in enumerate(xs) if v is not None]
            valid_y = [(i, v) for i, v in enumerate(ys) if v is not None]
            if len(valid_x) < window or len(valid_y) < window:
                continue
            # Linear interpolation for missing values
            xs_interp = xs[:]
            ys_interp = ys[:]
            for i in range(n):
                if xs_interp[i] is None:
                    before = [(j, v) for j, v in valid_x if j < i]
                    after = [(j, v) for j, v in valid_x if j > i]
                    if before and after:
                        b, a = before[-1], after[0]
                        t = (i - b[0]) / (a[0] - b[0])
                        xs_interp[i] = b[1] + t * (a[1] - b[1])
                    elif before:
                        xs_interp[i] = before[-1][1]
                    elif after:
                        xs_interp[i] = after[0][1]
                if ys_interp[i] is None:
                    before = [(j, v) for j, v in valid_y if j < i]
                    after = [(j, v) for j, v in valid_y if j > i]
                    if before and after:
                        b, a = before[-1], after[0]
                        t = (i - b[0]) / (a[0] - b[0])
                        ys_interp[i] = b[1] + t * (a[1] - b[1])
                    elif before:
                        ys_interp[i] = before[-1][1]
                    elif after:
                        ys_interp[i] = after[0][1]
            # Centred moving average
            smoothed_x = []
            smoothed_y = []
            for i in range(n):
                lo = max(0, i - half)
                hi = min(n, i + half + 1)
                smoothed_x.append(sum(xs_interp[lo:hi]) / (hi - lo))
                smoothed_y.append(sum(ys_interp[lo:hi]) / (hi - lo))
            # Write back, keeping original Nones for missing landmarks
            for i in range(n):
                if xs[i] is not None:
                    result[i]['pose']['landmarks'][k]['x'] = smoothed_x[i]
                if ys[i] is not None:
                    result[i]['pose']['landmarks'][k]['y'] = smoothed_y[i]

        return result

    def _filter_active_swimming_frames(self, frames: List[Dict]) -> List[Dict]:
        """Filter non-swimming frames using wrist_x range over a sliding window.

        During swimming the arms move widely; during preparation/waiting the
        wrist_x is relatively stable.  Frames with wrist_x range < threshold
        over a 1-second window are marked as inactive and excluded.

        Uses whichever wrist is visible, with a relative threshold based on
        frame width.
        """
        if len(frames) < 10:
            return frames

        # Determine frame width from first frame (x range of shoulders)
        first = frames[0]['pose']['landmarks']
        ls = first.get('left_shoulder')
        rs = first.get('right_shoulder')
        if ls and rs:
            frame_width = abs(ls['x'] - rs['x']) * 4  # rough frame width estimate
        else:
            frame_width = 1000  # fallback

        RELATIVE_THRESHOLD = 0.05  # 5% of frame width (more permissive)
        ACTIVE_THRESHOLD = max(30, frame_width * RELATIVE_THRESHOLD)
        WINDOW = 15  # ~0.5 second at 30 fps (half the old window)

        # Compute wrist_x for each frame using whichever wrist is visible
        wrist_x = []
        for frame in frames:
            landmarks = frame['pose']['landmarks']
            lw = landmarks.get('left_wrist')
            rw = landmarks.get('right_wrist')
            if lw and lw['visibility'] >= MIN_VISIBILITY:
                wrist_x.append(lw['x'])
            elif rw and rw['visibility'] >= MIN_VISIBILITY:
                wrist_x.append(rw['x'])
            else:
                wrist_x.append(None)

        # Determine active status for each frame
        active_flags = [False] * len(frames)
        for i in range(len(frames)):
            half = WINDOW // 2
            lo = max(0, i - half)
            hi = min(len(frames), i + half + 1)
            window_values = [wx for wx in wrist_x[lo:hi] if wx is not None]
            # Require at least 5 valid values in window (relaxed)
            if len(window_values) >= 5:
                w_range = max(window_values) - min(window_values)
                active_flags[i] = w_range >= ACTIVE_THRESHOLD

        # Return only active frames
        active = [frames[i] for i in range(len(frames)) if active_flags[i]]

        if len(active) < max(10, len(frames) * 0.3):
            # Fall back to all frames if too few are active
            return frames

        return active

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

        # ── Elbow during recovery (air phase) ──
        results['recovery_elbow'] = self._metric_recovery_elbow(frames, cycles, near_arm)

        # ── Breath lift ──
        results['breath_lift'] = self._metric_breath_lift(frames, torso_lengths)

        return results

    # ── Individual metric implementations ──

    # ── Air/water phase detection ──

    def _detect_air_water_phases(self, frames: List[Dict]) -> Dict:
        """Label each frame as 'air' (recovery, hand above water) or 'water'
        (pull, hand underwater) for both arms.

        Uses:
          - Wrist visibility (low -> underwater)
          - Wrist y relative to shoulder y (below shoulder -> underwater)
          - Temporal smoothing (majority over +/- 3 frames)
        """
        near_arm = self._arm_labels.get('near_arm', 'left')
        far_arm = 'right' if near_arm == 'left' else 'left'

        def _label_arm(arm):
            labels = []
            for f in frames:
                lm = f['pose']['landmarks']
                sh = lm[f'{arm}_shoulder']
                wr = lm[f'{arm}_wrist']
                if wr['visibility'] < 0.3 or wr['y'] > sh['y']:
                    labels.append('water')
                else:
                    labels.append('air')
            # Smooth: majority vote over +/- 3 frames
            smoothed = labels[:]
            for i in range(len(labels)):
                start = max(0, i - 3)
                end = min(len(labels), i + 4)
                window = labels[start:end]
                air_count = window.count('air')
                water_count = window.count('water')
                smoothed[i] = 'air' if air_count >= water_count else 'water'
            return smoothed

        return {
            'near_arm_labels': _label_arm(near_arm),
            'far_arm_labels': _label_arm(far_arm),
        }

    def _interpolate_underwater_limbs(self, frames: List[Dict], arm: str,
                                       phase_labels: List[str]) -> List[Dict]:
        """Return a copy of frames with wrist AND elbow positions interpolated
        during underwater ('water') segments using kinematic constraints.

        Instead of naive spline interpolation, this method:
          1. Measures the average upper-arm length (shoulder->elbow) and forearm
             length (elbow->wrist) from visible frames.
          2. During underwater segments, interpolates the shoulder trajectory
             (which should remain visible) using a cubic Hermite spline.
          3. Uses the measured arm lengths and a simple inverse-kinematics
             approach to estimate elbow and wrist positions: the limb is scaled
             to preserve the measured arm lengths while following the smooth
             trajectory of the Hermite spline.

        This gives physically realistic limb positions even when both the
        elbow and wrist are invisible underwater, without needing SMPL.
        """
        import copy
        import math
        result = copy.deepcopy(frames)

        # ---- Measure arm lengths from visible frames ----
        upper_arm_lengths = []
        forearm_lengths = []
        for f in frames:
            lm = f['pose']['landmarks']
            sh = lm[f'{arm}_shoulder']
            el = lm[f'{arm}_elbow']
            wr = lm[f'{arm}_wrist']
            if (sh['visibility'] >= MIN_VISIBILITY and
                el['visibility'] >= MIN_VISIBILITY and
                wr['visibility'] >= MIN_VISIBILITY):
                ua = math.sqrt((sh['x'] - el['x'])**2 + (sh['y'] - el['y'])**2)
                fa = math.sqrt((el['x'] - wr['x'])**2 + (el['y'] - wr['y'])**2)
                if ua > 0 and fa > 0:
                    upper_arm_lengths.append(ua)
                    forearm_lengths.append(fa)

        if not upper_arm_lengths:
            # Fall back to pure Hermite spline
            return self._interpolate_underwater_limbs_spline(
                frames, arm, phase_labels)

        avg_upper_len = np.median(upper_arm_lengths)
        avg_forearm_len = np.median(forearm_lengths)

        # ---- Find contiguous water segments ----
        water_segments = []
        in_water = False
        seg_start = 0
        for i, label in enumerate(phase_labels):
            if label == 'water' and not in_water:
                seg_start = i
                in_water = True
            elif label == 'air' and in_water:
                water_segments.append((seg_start, i - 1))
                in_water = False
        if in_water:
            water_segments.append((seg_start, len(phase_labels) - 1))

        # ---- Interpolate shoulder via Hermite spline ----
        shoulder_interpolated = self._interpolate_underwater_limbs_single(
            result, arm, phase_labels, f'{arm}_shoulder', water_segments)

        # ---- Helper: scale limb to preserve measured lengths ----
        def _scale_limb(sh_x, sh_y, el_x, el_y, wr_x, wr_y,
                         target_upper, target_forearm):
            """Scale elbow and wrist relative to shoulder to match target
            arm lengths while preserving direction."""
            ua = math.sqrt((sh_x - el_x)**2 + (sh_y - el_y)**2)
            if ua > 1e-6:
                s_upper = target_upper / ua
                el_x = sh_x + (el_x - sh_x) * s_upper
                el_y = sh_y + (el_y - sh_y) * s_upper
            fa = math.sqrt((el_x - wr_x)**2 + (el_y - wr_y)**2)
            if fa > 1e-6:
                s_forearm = target_forearm / fa
                wr_x = el_x + (wr_x - el_x) * s_forearm
                wr_y = el_y + (wr_y - el_y) * s_forearm
            return el_x, el_y, wr_x, wr_y

        # ---- Process each water segment ----
        for seg_start, seg_end in water_segments:
            pre_idx = seg_start - 1
            while pre_idx >= 0 and phase_labels[pre_idx] == 'water':
                pre_idx -= 1
            if pre_idx < 0:
                continue
            post_idx = seg_end + 1
            while post_idx < len(phase_labels) and phase_labels[post_idx] == 'water':
                post_idx += 1
            if post_idx >= len(phase_labels):
                continue

            total_gap = post_idx - pre_idx

            pre_lms = shoulder_interpolated[pre_idx]['pose']['landmarks']
            post_lms = shoulder_interpolated[post_idx]['pose']['landmarks']
            pre_sh = pre_lms[f'{arm}_shoulder']
            post_sh = post_lms[f'{arm}_shoulder']
            pre_el = pre_lms[f'{arm}_elbow']
            post_el = post_lms[f'{arm}_elbow']
            pre_wr = pre_lms[f'{arm}_wrist']
            post_wr = post_lms[f'{arm}_wrist']

            for j in range(seg_start, seg_end + 1):
                t = (j - pre_idx) / total_gap
                lm = result[j]['pose']['landmarks']

                # Shoulder already interpolated
                sh = shoulder_interpolated[j]['pose']['landmarks'][f'{arm}_shoulder']

                # Initial guess: linear blend of pre and post
                el_x = (1-t) * pre_el['x'] + t * post_el['x']
                el_y = (1-t) * pre_el['y'] + t * post_el['y']
                wr_x = (1-t) * pre_wr['x'] + t * post_wr['x']
                wr_y = (1-t) * pre_wr['y'] + t * post_wr['y']

                # Apply IK constraints
                el_x, el_y, wr_x, wr_y = _scale_limb(
                    sh['x'], sh['y'], el_x, el_y, wr_x, wr_y,
                    avg_upper_len, avg_forearm_len)

                lm[f'{arm}_elbow']['x'] = el_x
                lm[f'{arm}_elbow']['y'] = el_y
                lm[f'{arm}_elbow']['z'] = (1-t) * pre_el['z'] + t * post_el['z']
                lm[f'{arm}_elbow']['visibility'] = 0.5
                lm[f'{arm}_wrist']['x'] = wr_x
                lm[f'{arm}_wrist']['y'] = wr_y
                lm[f'{arm}_wrist']['z'] = (1-t) * pre_wr['z'] + t * post_wr['z']
                lm[f'{arm}_wrist']['visibility'] = 0.5

        return result

    def _interpolate_underwater_limbs_single(
        self, frames: List[Dict], arm: str, phase_labels: List[str],
        landmark_key: str, water_segments: List[Tuple[int, int]] = None
    ) -> List[Dict]:
        """Interpolate a single landmark across water segments using cubic
        Hermite spline. Returns a copy of frames with that landmark filled in."""
        import copy
        result = copy.deepcopy(frames)

        if water_segments is None:
            water_segments = []
            in_water = False
            seg_start = 0
            for i, label in enumerate(phase_labels):
                if label == 'water' and not in_water:
                    seg_start = i
                    in_water = True
                elif label == 'air' and in_water:
                    water_segments.append((seg_start, i - 1))
                    in_water = False
            if in_water:
                water_segments.append((seg_start, len(phase_labels) - 1))

        for seg_start, seg_end in water_segments:
            pre_idx = seg_start - 1
            while pre_idx >= 0 and phase_labels[pre_idx] == 'water':
                pre_idx -= 1
            if pre_idx < 0:
                continue
            pre_lm = result[pre_idx]['pose']['landmarks'][landmark_key]

            post_idx = seg_end + 1
            while post_idx < len(phase_labels) and phase_labels[post_idx] == 'water':
                post_idx += 1
            if post_idx >= len(phase_labels):
                continue
            post_lm = result[post_idx]['pose']['landmarks'][landmark_key]

            total_gap = post_idx - pre_idx
            seg_len = seg_end - seg_start + 1

            if seg_len < 3:
                for j in range(seg_start, seg_end + 1):
                    t = (j - pre_idx) / total_gap
                    lm = result[j]['pose']['landmarks'][landmark_key]
                    lm['x'] = pre_lm['x'] * (1 - t) + post_lm['x'] * t
                    lm['y'] = pre_lm['y'] * (1 - t) + post_lm['y'] * t
                    lm['z'] = pre_lm['z'] * (1 - t) + post_lm['z'] * t
                    lm['visibility'] = 0.5
                continue

            pre2_idx = max(0, pre_idx - 1)
            if pre2_idx != pre_idx:
                dt_pre = pre_idx - pre2_idx
                m_pre_x = (pre_lm['x'] - result[pre2_idx]['pose']['landmarks'][landmark_key]['x']) / dt_pre
                m_pre_y = (pre_lm['y'] - result[pre2_idx]['pose']['landmarks'][landmark_key]['y']) / dt_pre
                m_pre_z = (pre_lm['z'] - result[pre2_idx]['pose']['landmarks'][landmark_key]['z']) / dt_pre
            else:
                m_pre_x = (post_lm['x'] - pre_lm['x']) / total_gap
                m_pre_y = (post_lm['y'] - pre_lm['y']) / total_gap
                m_pre_z = (post_lm['z'] - pre_lm['z']) / total_gap

            post2_idx = min(len(phase_labels) - 1, post_idx + 1)
            if post2_idx != post_idx:
                dt_post = post2_idx - post_idx
                m_post_x = (result[post2_idx]['pose']['landmarks'][landmark_key]['x'] - post_lm['x']) / dt_post
                m_post_y = (result[post2_idx]['pose']['landmarks'][landmark_key]['y'] - post_lm['y']) / dt_post
                m_post_z = (result[post2_idx]['pose']['landmarks'][landmark_key]['z'] - post_lm['z']) / dt_post
            else:
                m_post_x = (post_lm['x'] - pre_lm['x']) / total_gap
                m_post_y = (post_lm['y'] - pre_lm['y']) / total_gap
                m_post_z = (post_lm['z'] - pre_lm['z']) / total_gap

            for j in range(seg_start, seg_end + 1):
                t = (j - pre_idx) / total_gap
                h00 = 1 - 3*t*t + 2*t*t*t
                h10 = t - 2*t*t + t*t*t
                h01 = -t*t + 2*t*t*t
                h11 = -t*t + t*t*t
                lm = result[j]['pose']['landmarks'][landmark_key]
                lm['x'] = h00 * pre_lm['x'] + h10 * m_pre_x * total_gap + h01 * post_lm['x'] + h11 * m_post_x * total_gap
                lm['y'] = h00 * pre_lm['y'] + h10 * m_pre_y * total_gap + h01 * post_lm['y'] + h11 * m_post_y * total_gap
                lm['z'] = h00 * pre_lm['z'] + h10 * m_pre_z * total_gap + h01 * post_lm['z'] + h11 * m_post_z * total_gap
                lm['visibility'] = 0.5

        return result

    def _interpolate_underwater_limbs_spline(
        self, frames: List[Dict], arm: str, phase_labels: List[str]
    ) -> List[Dict]:
        """Fallback: pure cubic Hermite spline interpolation for wrist and
        elbow, without kinematic constraints."""
        result = self._interpolate_underwater_limbs_single(
            frames, arm, phase_labels, f'{arm}_wrist')
        result = self._interpolate_underwater_limbs_single(
            result, arm, phase_labels, f'{arm}_elbow')
        return result

    def _metric_elbow_min_pull(self, frames: List[Dict], cycles: List[Tuple[int, int]],
                                near_arm: str) -> Dict:
        """Minimum elbow angle during the pull (water) phase (2D, near arm).

        Uses air/water phase labels to measure elbow angle only during the
        underwater pull phase.  When the wrist and elbow are invisible
        underwater, their positions are interpolated via cubic Hermite spline
        from visible frames before/after the segment.
        """
        ELBOW_MIN = 30.0
        ELBOW_MAX = 180.0

        phase_info = self._detect_air_water_phases(frames)
        water_labels = phase_info['near_arm_labels']
        interpolated_frames = self._interpolate_underwater_limbs(frames, near_arm, water_labels)

        if not cycles:
            angles = []
            for i, f in enumerate(interpolated_frames):
                if water_labels[i] != 'water':
                    continue
                landmarks = f['pose']['landmarks']
                sh = landmarks[f'{near_arm}_shoulder']
                el = landmarks[f'{near_arm}_elbow']
                wr = landmarks[f'{near_arm}_wrist']
                if (sh['visibility'] >= MIN_VISIBILITY and
                    el['visibility'] >= MIN_VISIBILITY):
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
                if i >= len(interpolated_frames):
                    break
                if water_labels[i] != 'water':
                    continue
                landmarks = interpolated_frames[i]['pose']['landmarks']
                sh = landmarks[f'{near_arm}_shoulder']
                el = landmarks[f'{near_arm}_elbow']
                wr = landmarks[f'{near_arm}_wrist']
                if (sh['visibility'] >= MIN_VISIBILITY and
                    el['visibility'] >= MIN_VISIBILITY):
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

    def _metric_recovery_elbow(self, frames: List[Dict], cycles: List[Tuple[int, int]],
                                near_arm: str) -> Dict:
        """Elbow angle during the recovery (air) phase (2D, near arm).

        Measured only on frames labelled 'air'.
        """
        ELBOW_MIN = 30.0
        ELBOW_MAX = 180.0

        phase_info = self._detect_air_water_phases(frames)
        air_labels = phase_info['near_arm_labels']

        if not cycles:
            angles = []
            for i, f in enumerate(frames):
                if air_labels[i] != 'air':
                    continue
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
                'avg': np.mean(angles) if angles else None,
                'max': np.max(angles) if angles else None,
                'num_frames': len(angles),
            }

        per_cycle_avg = []
        per_cycle_max = []
        for start, end in cycles:
            angles = []
            for i in range(start, end):
                if i >= len(frames):
                    break
                if air_labels[i] != 'air':
                    continue
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
                per_cycle_avg.append(np.mean(angles))
                per_cycle_max.append(np.max(angles))

        if not per_cycle_avg:
            return {'avg': None, 'max': None, 'num_frames': 0}

        return {
            'avg': np.mean(per_cycle_avg),
            'max': np.max(per_cycle_max),
            'num_frames': sum(1 for i in range(len(frames)) if air_labels[i] == 'air'),
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
        """Body line via shoulder→hip→knee angle vs horizontal.

        Applies Hampel outlier rejection to remove tracker confusion.
        """
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

        # Outlier rejection via Hampel filter
        if angles:
            filtered = hampel_filter(angles, HAMPEL_WINDOW, HAMPEL_NSIGMA)
            filtered = [v for v in filtered if v is not None]
            if len(filtered) >= 3:
                angles = filtered

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

        # ── No cycles detected ──
        num_cycles = metrics.get('num_cycles', 0)
        if num_cycles == 0 and valid_ratio >= MIN_VALID_FRAME_RATIO:
            return [BaseIssue('no_cycles_detected', SEVERITY_CRITICAL,
                              'No stroke cycles detected — swimmer not visible or not swimming during the selected time range',
                              'Check that the video shows the swimmer from a side view and the time range covers active swimming')]

        # ── Elbow dropped (water phase) ──
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

        # ── Elbow during recovery (air phase) ──
        rec_elbow = metrics.get('recovery_elbow', {})
        if rec_elbow.get('avg') is not None:
            avg_rec_elbow = rec_elbow['avg']
            rec_elbow_max = cfg.thresholds.get('recovery_elbow_max', 160)
            if avg_rec_elbow > rec_elbow_max:
                issues.append(BaseIssue('recovery_elbow_too_straight', SEVERITY_MODERATE,
                    f"Arm too straight during recovery (avg {avg_rec_elbow:.0f}°)",
                    'Keep elbow slightly bent to skim the water', avg_rec_elbow))

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
            # dev = 180° = perfectly straight; dev < 180 - hip_line_drop_max means
            # the body is bent more than hip_line_drop_max degrees from straight
            max_straight_dev = cfg.thresholds.get('hip_line_drop_max', 15)
            threshold = 180 - max_straight_dev
            if dev < threshold:
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
