"""
Precision stroke analyzer: cycle-based, phase-specific metrics with confidence.

This is a ground-up rewrite that replaces the old frame-averaging approach with:
  1. Stroke-cycle segmentation (from stroke_cycle_detector)
  2. Phase-specific metrics (catch elbow ≠ pull elbow)
  3. Per-cycle statistics (median, IQR) instead of global averages
  4. Confidence propagation — every metric carries an uncertainty
  5. Body-relative normalization (normalized to shoulder-hip distance)

This trades speed for significantly more robust, coaching-quality feedback.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from src.stroke_cycle_detector import StrokeCycleDetector
from src.models.freestyle_rules import (
    ELBOW_ANGLE_OPTIMAL_MIN, ELBOW_ANGLE_OPTIMAL_MAX, ELBOW_ANGLE_DROPPED,
    BODY_ROTATION_OPTIMAL_MIN, BODY_ROTATION_OPTIMAL_MAX,
    BODY_ROTATION_TOO_FLAT, BODY_ROTATION_TOO_MUCH,
    ARM_ENTRY_CENTERLINE_THRESHOLD, HEAD_LIFT_THRESHOLD,
    STROKE_RATE_OPTIMAL_MIN, STROKE_RATE_OPTIMAL_MAX,
    KNEE_ANGLE_OPTIMAL, KNEE_ANGLE_EXCESSIVE_BEND,
    MIN_VISIBILITY, FreestyleIssue, ISSUE_TYPES,
    SEVERITY_CRITICAL, SEVERITY_MODERATE, SEVERITY_MINOR
)


# ── Body-relative normalization ──

def _body_normalize(pose_data: List[Dict]) -> List[Dict]:
    """
    Convert raw pixel coordinates to body-relative coordinates.

    For each frame with valid pose, transforms landmarks to be relative to
    the hip center, scaled by shoulder-to-hip distance. This makes metrics
    comparable across different body sizes, camera distances, and resolutions.

    Modifies pose_data in-place.
    """
    if not pose_data:
        return pose_data

    # Compute median body length (shoulder-hip distance) across valid frames
    # Uses 3D coordinates to account for body roll and foreshortening
    body_lengths = []
    for fd in pose_data:
        pose = fd.get('pose')
        if pose is None:
            continue
        landmarks = pose.get('landmarks', {})
        if not landmarks:
            continue

        ls = landmarks.get('left_shoulder')
        rs = landmarks.get('right_shoulder')
        lh = landmarks.get('left_hip')
        rh = landmarks.get('right_hip')
        if (ls and rs and lh and rh and
            ls.get('visibility', 0) >= MIN_VISIBILITY and
            rs.get('visibility', 0) >= MIN_VISIBILITY and
            lh.get('visibility', 0) >= MIN_VISIBILITY and
            rh.get('visibility', 0) >= MIN_VISIBILITY):

            # Shoulder midpoint (3D)
            sx = (ls['x'] + rs['x']) / 2
            sy = (ls['y'] + rs['y']) / 2
            sz = (ls.get('z', 0) + rs.get('z', 0)) / 2
            # Hip midpoint (3D)
            hx = (lh['x'] + rh['x']) / 2
            hy = (lh['y'] + rh['y']) / 2
            hz = (lh.get('z', 0) + rh.get('z', 0)) / 2
            # Shoulder-hip distance (3D)
            dist = np.sqrt((sx - hx)**2 + (sy - hy)**2 + (sz - hz)**2)
            body_lengths.append(dist)

    if not body_lengths:
        return pose_data

    # Use median for robustness
    median_body_length = np.median(body_lengths)

    if median_body_length < 1:
        return pose_data

    # Normalize each frame
    for fd in pose_data:
        pose = fd.get('pose')
        if pose is None:
            continue
        landmarks = pose.get('landmarks', {})
        if not landmarks:
            continue

        # Compute hip center for this frame (3D centroid)
        lh = landmarks.get('left_hip')
        rh = landmarks.get('right_hip')
        if lh and rh and lh.get('visibility', 0) >= MIN_VISIBILITY and rh.get('visibility', 0) >= MIN_VISIBILITY:
            cx = (lh['x'] + rh['x']) / 2
            cy = (lh['y'] + rh['y']) / 2
            cz = (lh.get('z', 0) + rh.get('z', 0)) / 2
        else:
            # Fall back to shoulder center (3D)
            ls = landmarks.get('left_shoulder')
            rs = landmarks.get('right_shoulder')
            if ls and rs and ls.get('visibility', 0) >= MIN_VISIBILITY and rs.get('visibility', 0) >= MIN_VISIBILITY:
                cx = (ls['x'] + rs['x']) / 2
                cy = (ls['y'] + rs['y']) / 2
                cz = (ls.get('z', 0) + rs.get('z', 0)) / 2
            else:
                continue

        # Normalize each landmark
        for name, lm in landmarks.items():
            # Store original for angle calculations (which use raw coords)
            lm['x_raw'] = lm['x']
            lm['y_raw'] = lm['y']
            lm['z_raw'] = lm.get('z', 0)
            # Normalized coordinates (3D-aware)
            lm['x_norm'] = (lm['x'] - cx) / median_body_length
            lm['y_norm'] = (lm['y'] - cy) / median_body_length
            lm['z_norm'] = (lm.get('z', 0) - cz) / median_body_length

    return pose_data


# ── Confidence-aware metrics ──

class ConfidenceMetric:
    """A metric value with associated confidence/uncertainty."""

    def __init__(self, value: Optional[float], confidence: float = 0.0,
                 n_samples: int = 0):
        self.value = value
        self.confidence = confidence  # 0-1 scale
        self.n_samples = n_samples

    def to_dict(self):
        return {
            'value': self.value,
            'confidence': self.confidence,
            'n_samples': self.n_samples,
        }

    def __repr__(self):
        if self.value is None:
            return f"ConfidenceMetric(None, conf={self.confidence:.2f})"
        return f"ConfidenceMetric({self.value:.2f}, conf={self.confidence:.2f}, n={self.n_samples})"


def _compute_confidence(visibilities: List[float], model_agreement: Optional[List[float]] = None) -> float:
    """
    Compute overall confidence from per-frame visibility scores.

    Args:
        visibilities: List of visibility values (0-1)
        model_agreement: Optional list of inter-model agreement (0-1)

    Returns:
        Confidence score 0-1
    """
    if not visibilities:
        return 0.0

    # Base: mean visibility
    mean_vis = np.mean(visibilities)

    # Consistency: low std means stable detection
    if len(visibilities) > 1:
        std_vis = np.std(visibilities)
        consistency = 1.0 - min(1.0, std_vis)
    else:
        consistency = 0.5

    # Model agreement
    if model_agreement and model_agreement:
        mean_agree = np.mean(model_agreement)
    else:
        mean_agree = 0.5

    # Combined: weighted average
    confidence = 0.5 * mean_vis + 0.3 * consistency + 0.2 * mean_agree

    return float(max(0.0, min(1.0, confidence)))


# ── Precision Stroke Analyzer ──

class StrokeAnalyzer:
    """Precision freestyle analyzer using cycle-based, phase-specific metrics."""

    def __init__(self, use_cycle_detection: bool = True):
        self.use_cycle_detection = use_cycle_detection
        self.cycle_detector = StrokeCycleDetector(stroke_type='freestyle')
        self.metrics = {}
        self.issues = []

    def analyze_video(self, pose_data: List[Dict]) -> Dict:
        """
        Analyze freestyle technique with precision pipeline.

        Steps:
          1. Body-relative normalization
          2. Stroke-cycle detection
          3. Phase-specific metric computation per cycle
          4. Confidence estimation per metric
          5. Issue detection with confidence gates
        """
        print("\n[Precision] Analyzing stroke mechanics...")

        # Step 1: Normalize to body-relative coordinates
        print("  Normalizing to body-relative coordinates...")
        _body_normalize(pose_data)

        # Step 2: Detect stroke cycles
        cycles = self.cycle_detector.detect_cycles(pose_data)

        # Filter valid frames
        valid_frames = [fd for fd in pose_data if fd['pose'] is not None]
        if not valid_frames:
            return {
                'error': 'No valid pose data detected',
                'metrics': {},
                'issues': []
            }
        print(f"  Valid frames: {len(valid_frames)}/{len(pose_data)}")

        # Step 3: Compute phase-specific metrics
        metrics = {}

        if cycles and self.use_cycle_detection:
            # ── Cycle-based metrics ──
            metrics['cycle_analysis'] = self._analyze_by_cycle(cycles, pose_data)
            metrics['num_cycles'] = len(cycles)

            # Phase-specific metrics (across all cycles)
            metrics['catch'] = self._compute_phase_metric(
                cycles, pose_data, 'catch',
                self._compute_catch_metrics
            )
            metrics['pull'] = self._compute_phase_metric(
                cycles, pose_data, 'pull',
                self._compute_pull_metrics
            )
            metrics['recovery'] = self._compute_phase_metric(
                cycles, pose_data, 'recovery',
                self._compute_recovery_metrics
            )
            metrics['entry'] = self._compute_phase_metric(
                cycles, pose_data, 'entry',
                self._compute_entry_metrics
            )
        else:
            print("  No cycles detected — falling back to frame-based analysis")
            metrics['cycle_analysis'] = None

        # ── Always-computed metrics (even without cycles) ──
        metrics['elbow'] = self._analyze_elbow_angles(valid_frames, cycles)
        metrics['rotation'] = self._analyze_body_rotation(valid_frames)
        metrics['head'] = self._analyze_head_position(valid_frames)
        metrics['stroke_rate'] = self._analyze_stroke_rate(valid_frames, cycles)
        metrics['kick'] = self._analyze_kick(valid_frames)

        # Step 4: Detect issues with confidence gates
        self.metrics = metrics
        self.issues = self._detect_issues()

        return {
            'metrics': metrics,
            'issues': self.issues,
            'cycles': cycles,
        }

    def _compute_phase_metric(self, cycles: List[Dict], pose_data: List[Dict],
                               phase_name: str,
                               compute_fn) -> Dict:
        """Compute metric for a specific phase across all cycles."""
        phase_frames = []
        for cycle in cycles:
            phase_frames_list = cycle.get('phase_frames', {})
            if phase_name in phase_frames_list:
                start, end = phase_frames_list[phase_name]
                # Collect frame indices for this phase
                for i in range(start, end + 1):
                    if i < len(pose_data) and pose_data[i]['pose'] is not None:
                        phase_frames.append(i)

        if not phase_frames:
            return {}

        # Get the relevant pose data frames
        phase_pose = [pose_data[i] for i in phase_frames]
        return compute_fn(phase_pose)

    def _compute_catch_metrics(self, frames: List[Dict]) -> Dict:
        """Compute metrics for the catch phase only."""
        elbow_angles = []
        wrist_to_shoulder = []
        body_rotations = []

        for fd in frames:
            landmarks = fd['pose']['landmarks']

            # Elbow angle
            if (landmarks['left_shoulder']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_elbow']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_wrist']['visibility'] >= MIN_VISIBILITY):
                angle = self._calc_angle(
                    landmarks['left_shoulder'],
                    landmarks['left_elbow'],
                    landmarks['left_wrist']
                )
                elbow_angles.append(angle)

            # Wrist-to-shoulder vertical distance (normalized)
            ls = landmarks.get('left_shoulder')
            lw = landmarks.get('left_wrist')
            if ls and lw and ls.get('visibility', 0) >= MIN_VISIBILITY and lw.get('visibility', 0) >= MIN_VISIBILITY:
                # Use normalized coordinates
                y_dist = lw.get('y_norm', lw['y']) - ls.get('y_norm', ls['y'])
                wrist_to_shoulder.append(y_dist)

            # Body rotation
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
                body_rotations.append(max(0, min(90, rotation)))

        result = {}

        # Elbow angle at catch (median for robustness)
        if elbow_angles:
            result['catch_elbow_angle'] = ConfidenceMetric(
                float(np.median(elbow_angles)),
                confidence=_compute_confidence(
                    [fd['pose']['landmarks']['left_elbow']['visibility']
                     for fd in frames if 'left_elbow' in fd['pose']['landmarks']]
                ),
                n_samples=len(elbow_angles)
            ).to_dict()
            result['catch_elbow_angle_iqr'] = float(
                np.percentile(elbow_angles, 75) - np.percentile(elbow_angles, 25)
            )

        # Wrist position relative to shoulder
        if wrist_to_shoulder:
            result['wrist_to_shoulder_y'] = ConfidenceMetric(
                float(np.median(wrist_to_shoulder)),
                confidence=0.5,
                n_samples=len(wrist_to_shoulder)
            ).to_dict()

        # Body rotation at catch
        if body_rotations:
            result['catch_rotation'] = ConfidenceMetric(
                float(np.median(body_rotations)),
                confidence=0.3,  # Rotation from 2D is heuristic
                n_samples=len(body_rotations)
            ).to_dict()

        return result

    def _compute_pull_metrics(self, frames: List[Dict]) -> Dict:
        """Compute metrics for the pull phase."""
        elbow_angles = []
        hand_path = []  # Wrist x trajectory (normalized)

        for fd in frames:
            landmarks = fd['pose']['landmarks']

            # Elbow angle
            if (landmarks['left_shoulder']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_elbow']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_wrist']['visibility'] >= MIN_VISIBILITY):
                angle = self._calc_angle(
                    landmarks['left_shoulder'],
                    landmarks['left_elbow'],
                    landmarks['left_wrist']
                )
                elbow_angles.append(angle)

            # Hand path (x position relative to shoulder, normalized)
            lw = landmarks.get('left_wrist')
            ls = landmarks.get('left_shoulder')
            if lw and ls and lw.get('visibility', 0) >= MIN_VISIBILITY and ls.get('visibility', 0) >= MIN_VISIBILITY:
                # Use normalized x
                hand_path.append(lw.get('x_norm', lw['x']) - ls.get('x_norm', ls['x']))

        result = {}
        if elbow_angles:
            result['pull_elbow_angle'] = ConfidenceMetric(
                float(np.median(elbow_angles)),
                confidence=_compute_confidence(
                    [fd['pose']['landmarks']['left_elbow']['visibility']
                     for fd in frames if 'left_elbow' in fd['pose']['landmarks']]
                ),
                n_samples=len(elbow_angles)
            ).to_dict()
            result['pull_elbow_max'] = float(np.max(elbow_angles))
            result['pull_elbow_min'] = float(np.min(elbow_angles))

        if hand_path:
            # Hand path length = total x displacement during pull
            result['hand_path_length'] = float(abs(hand_path[-1] - hand_path[0])) if len(hand_path) > 1 else None

        return result

    def _compute_recovery_metrics(self, frames: List[Dict]) -> Dict:
        """Compute metrics for the recovery phase."""
        elbow_angles = []
        wrist_clearance = []  # How high wrist goes above shoulder

        for fd in frames:
            landmarks = fd['pose']['landmarks']

            # Elbow angle during recovery (should be > 100°)
            if (landmarks['left_shoulder']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_elbow']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_wrist']['visibility'] >= MIN_VISIBILITY):
                angle = self._calc_angle(
                    landmarks['left_shoulder'],
                    landmarks['left_elbow'],
                    landmarks['left_wrist']
                )
                elbow_angles.append(angle)

            # Wrist clearance (how much wrist is above shoulder, normalized)
            lw = landmarks.get('left_wrist')
            ls = landmarks.get('left_shoulder')
            if lw and ls and lw.get('visibility', 0) >= MIN_VISIBILITY and ls.get('visibility', 0) >= MIN_VISIBILITY:
                clearance = ls.get('y_norm', ls['y']) - lw.get('y_norm', lw['y'])
                wrist_clearance.append(clearance)

        result = {}
        if elbow_angles:
            result['recovery_elbow_angle'] = ConfidenceMetric(
                float(np.median(elbow_angles)),
                confidence=0.5,
                n_samples=len(elbow_angles)
            ).to_dict()

        if wrist_clearance:
            result['wrist_clearance'] = ConfidenceMetric(
                float(np.median(wrist_clearance)),
                confidence=0.5,
                n_samples=len(wrist_clearance)
            ).to_dict()

        return result

    def _compute_entry_metrics(self, frames: List[Dict]) -> Dict:
        """Compute metrics for the entry phase."""
        centerline_distances = []
        wrist_spread = []

        for fd in frames:
            landmarks = fd['pose']['landmarks']

            # Centerline crossing
            ls = landmarks.get('left_shoulder')
            rs = landmarks.get('right_shoulder')
            lw = landmarks.get('left_wrist')
            rw = landmarks.get('right_wrist')
            if (ls and rs and ls.get('visibility', 0) >= MIN_VISIBILITY and
                rs.get('visibility', 0) >= MIN_VISIBILITY):
                cx = (ls['x'] + rs['x']) / 2
                if lw and lw.get('visibility', 0) >= MIN_VISIBILITY:
                    dist = abs(lw['x'] - cx)
                    centerline_distances.append(dist)

            # Wrist spread (both wrists visible)
            if lw and rw and lw.get('visibility', 0) >= MIN_VISIBILITY and rw.get('visibility', 0) >= MIN_VISIBILITY:
                spread = abs(lw['x'] - rw['x'])
                wrist_spread.append(spread)

        result = {}
        if centerline_distances:
            result['centerline_distance'] = ConfidenceMetric(
                float(np.median(centerline_distances)),
                confidence=0.6,
                n_samples=len(centerline_distances)
            ).to_dict()

        if wrist_spread:
            result['wrist_spread'] = ConfidenceMetric(
                float(np.median(wrist_spread)),
                confidence=0.5,
                n_samples=len(wrist_spread)
            ).to_dict()

        return result

    def _analyze_by_cycle(self, cycles: List[Dict], pose_data: List[Dict]) -> Dict:
        """Aggregate per-cycle metrics for consistency analysis."""
        cycle_metrics = []
        for cycle in cycles:
            cm = {
                'cycle': cycle['cycle_number'],
                'duration': cycle.get('duration_frames', 0),
            }

            # Collect average elbow angle in this cycle
            elbow_vals = []
            for i in range(cycle['start_frame'], cycle['end_frame'] + 1):
                if i < len(pose_data) and pose_data[i]['pose'] is not None:
                    landmarks = pose_data[i]['pose']['landmarks']
                    if (landmarks['left_shoulder']['visibility'] >= MIN_VISIBILITY and
                        landmarks['left_elbow']['visibility'] >= MIN_VISIBILITY and
                        landmarks['left_wrist']['visibility'] >= MIN_VISIBILITY):
                        angle = self._calc_angle(
                            landmarks['left_shoulder'],
                            landmarks['left_elbow'],
                            landmarks['left_wrist']
                        )
                        elbow_vals.append(angle)

            if elbow_vals:
                cm['avg_elbow'] = float(np.median(elbow_vals))
                cm['elbow_iqr'] = float(
                    np.percentile(elbow_vals, 75) - np.percentile(elbow_vals, 25)
                )

            cycle_metrics.append(cm)

        return {
            'per_cycle': cycle_metrics,
            'num_cycles': len(cycle_metrics),
            'elbow_consistency': self._compute_consistency(
                [c.get('avg_elbow') for c in cycle_metrics if c.get('avg_elbow') is not None]
            ),
        }

    @staticmethod
    def _compute_consistency(values: List[float]) -> Dict:
        """Compute how consistent a metric is across cycles."""
        if not values:
            return {'score': None, 'variability': None}

        values = [v for v in values if v is not None]
        if len(values) < 3:
            return {'score': None, 'variability': None}

        # Coefficient of variation
        mean = np.mean(values)
        std = np.std(values)
        cv = std / (mean + 1e-6)

        # Score: 0-1, higher = more consistent
        score = max(0.0, 1.0 - cv * 2)

        return {
            'score': float(score),
            'variability': float(cv),
            'mean': float(mean),
            'std': float(std),
            'n_cycles': len(values),
        }

    # ── Legacy metrics (kept for backward compatibility) ──

    def _analyze_elbow_angles(self, frames: List[Dict], cycles: Optional[List] = None) -> Dict:
        """Elbow angle metrics with confidence."""
        left_elbow_angles = []
        right_elbow_angles = []
        left_vis = []
        right_vis = []

        for fd in frames:
            landmarks = fd['pose']['landmarks']

            if (landmarks['left_shoulder']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_elbow']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_wrist']['visibility'] >= MIN_VISIBILITY):
                angle = self._calc_angle(
                    landmarks['left_shoulder'],
                    landmarks['left_elbow'],
                    landmarks['left_wrist']
                )
                left_elbow_angles.append(angle)
                left_vis.append(landmarks['left_elbow']['visibility'])

            if (landmarks['right_shoulder']['visibility'] >= MIN_VISIBILITY and
                landmarks['right_elbow']['visibility'] >= MIN_VISIBILITY and
                landmarks['right_wrist']['visibility'] >= MIN_VISIBILITY):
                angle = self._calc_angle(
                    landmarks['right_shoulder'],
                    landmarks['right_elbow'],
                    landmarks['right_wrist']
                )
                right_elbow_angles.append(angle)
                right_vis.append(landmarks['right_elbow']['visibility'])

        all_angles = left_elbow_angles + right_elbow_angles
        all_vis = left_vis + right_vis

        result = {}
        if all_angles:
            result['avg_angle'] = float(np.mean(all_angles))
            result['median_angle'] = float(np.median(all_angles))
            result['min_angle'] = float(np.min(all_angles))
            result['max_angle'] = float(np.max(all_angles))
            result['iqr'] = float(np.percentile(all_angles, 75) - np.percentile(all_angles, 25))
            result['confidence'] = float(_compute_confidence(all_vis))

        if left_elbow_angles:
            result['left_avg'] = float(np.mean(left_elbow_angles))
        if right_elbow_angles:
            result['right_avg'] = float(np.mean(right_elbow_angles))

        return result

    def _analyze_body_rotation(self, frames: List[Dict]) -> Dict:
        """Body rotation with confidence."""
        rotations = []
        for fd in frames:
            landmarks = fd['pose']['landmarks']
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
                rotations.append(max(0, min(90, rotation)))

        result = {}
        if rotations:
            result['avg_rotation'] = float(np.mean(rotations))
            result['median_rotation'] = float(np.median(rotations))
            result['min_rotation'] = float(np.min(rotations))
            result['max_rotation'] = float(np.max(rotations))
            result['std_rotation'] = float(np.std(rotations))
            result['confidence'] = float(
                _compute_confidence([fd['pose']['landmarks']['left_shoulder']['visibility']
                                     for fd in frames if 'left_shoulder' in fd['pose']['landmarks']])
            )

        return result

    def _analyze_head_position(self, frames: List[Dict]) -> Dict:
        """Head stability with confidence."""
        nose_y_positions = []
        nose_vis = []

        for fd in frames:
            landmarks = fd['pose']['landmarks']
            nose = landmarks.get('nose')
            if nose and nose['visibility'] >= MIN_VISIBILITY:
                nose_y_positions.append(nose['y'])
                nose_vis.append(nose['visibility'])

        result = {}
        if nose_y_positions:
            y_range = np.max(nose_y_positions) - np.min(nose_y_positions)
            frame_h = frames[0]['pose']['frame_shape'][0]
            normalized_range = y_range / frame_h
            result['vertical_movement'] = float(normalized_range)
            result['avg_y'] = float(np.mean(nose_y_positions))
            result['stability'] = float(1.0 - normalized_range)
            result['confidence'] = float(_compute_confidence(nose_vis))

        return result

    def _analyze_stroke_rate(self, frames: List[Dict], cycles: Optional[List] = None) -> Dict:
        """Stroke rate from cycles or wrist trajectory."""
        if cycles and len(cycles) >= 2:
            # Use cycle timestamps for rate
            timestamps = []
            for fd in frames:
                if fd['frame_number'] in [cycles[0]['start_frame'], cycles[-1]['end_frame']]:
                    timestamps.append(fd['timestamp'])
            if len(timestamps) >= 2:
                duration = timestamps[-1] - timestamps[0]
                if duration > 0:
                    num_strokes = len(cycles) * 2  # Each cycle = 2 arm strokes
                    spm = (num_strokes / duration) * 60
                    result = {'spm': float(spm), 'total_strokes': num_strokes,
                              'duration': duration, 'n_cycles': len(cycles),
                              'confidence': 0.7}
                    if 20 <= spm <= 120:
                        return result
                    return {'spm': None}

        # Fall back to wrist-x peak detection
        return self._legacy_stroke_rate(frames)

    def _legacy_stroke_rate(self, frames: List[Dict]) -> Dict:
        """Original stroke rate from wrist trajectory."""
        if len(frames) < 10:
            return {'spm': None}

        samples = []
        for fd in frames:
            landmarks = fd['pose']['landmarks']
            if landmarks['left_wrist']['visibility'] >= MIN_VISIBILITY:
                samples.append((fd['timestamp'], landmarks['left_wrist']['x']))

        if len(samples) < 10:
            return {'spm': None}

        timestamps = [s[0] for s in samples]
        wrist_x = [s[1] for s in samples]
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            return {'spm': None}

        window = max(3, len(wrist_x) // 20)
        smoothed = self._moving_average(wrist_x, window)

        fps_equiv = len(smoothed) / duration
        min_peak_gap = max(3, int(fps_equiv * 0.5))

        candidate_peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
                candidate_peaks.append(i)

        filtered_peaks = []
        for peak in candidate_peaks:
            if not filtered_peaks or (peak - filtered_peaks[-1]) >= min_peak_gap:
                filtered_peaks.append(peak)

        total_strokes = len(filtered_peaks) * 2
        spm = (total_strokes / duration) * 60

        if spm < 20 or spm > 120:
            spm = None

        return {
            'spm': spm,
            'total_strokes': total_strokes if spm is not None else None,
            'duration': duration,
            'confidence': 0.5,
        }

    def _analyze_kick(self, frames: List[Dict]) -> Dict:
        """Kick analysis with confidence."""
        knee_angles = []
        knee_vis = []

        for fd in frames:
            landmarks = fd['pose']['landmarks']
            for side in ('left', 'right'):
                hip = landmarks.get(f'{side}_hip')
                knee = landmarks.get(f'{side}_knee')
                ankle = landmarks.get(f'{side}_ankle')
                if (hip and knee and ankle and
                    hip['visibility'] >= MIN_VISIBILITY and
                    knee['visibility'] >= MIN_VISIBILITY and
                    ankle['visibility'] >= MIN_VISIBILITY):
                    angle = self._calc_angle(hip, knee, ankle)
                    knee_angles.append(angle)
                    knee_vis.append(knee['visibility'])

        result = {}
        if knee_angles:
            result['avg_knee_angle'] = float(np.mean(knee_angles))
            result['median_knee_angle'] = float(np.median(knee_angles))
            result['min_knee_angle'] = float(np.min(knee_angles))
            result['confidence'] = float(_compute_confidence(knee_vis))

        return result

    def _detect_issues(self) -> List[FreestyleIssue]:
        """Detect technique issues with confidence gates.

        Only generates an issue if the metric confidence exceeds a threshold.
        """
        issues = []
        metrics = self.metrics

        # ── Elbow angle ──
        if metrics.get('elbow', {}).get('avg_angle') is not None:
            avg_elbow = metrics['elbow']['avg_angle']
            confidence = metrics['elbow'].get('confidence', 0.3)

            # Only flag if confidence is reasonable
            if confidence >= 0.4:
                if avg_elbow > ELBOW_ANGLE_DROPPED:
                    issues.append(FreestyleIssue(
                        'dropped_elbow',
                        ISSUE_TYPES['dropped_elbow']['severity'],
                        f"Dropped elbow during catch (avg {avg_elbow:.0f}° — "
                        f"optimal {ELBOW_ANGLE_OPTIMAL_MIN}-{ELBOW_ANGLE_OPTIMAL_MAX}°, "
                        f"confidence: {confidence:.0%})",
                        ISSUE_TYPES['dropped_elbow']['tip'],
                        avg_elbow
                    ))

        # ── Catch-specific elbow ──
        catch_metrics = metrics.get('catch', {})
        if catch_metrics.get('catch_elbow_angle', {}).get('value') is not None:
            catch_elbow = catch_metrics['catch_elbow_angle']['value']
            catch_conf = catch_metrics['catch_elbow_angle'].get('confidence', 0.3)
            if catch_conf >= 0.4 and catch_elbow > ELBOW_ANGLE_DROPPED:
                issues.append(FreestyleIssue(
                    'dropped_elbow',
                    ISSUE_TYPES['dropped_elbow']['severity'],
                    f"Dropped elbow during CATCH phase (median {catch_elbow:.0f}° "
                    f"across {catch_metrics['catch_elbow_angle'].get('n_samples', 0)} frames, "
                    f"confidence: {catch_conf:.0%})",
                    ISSUE_TYPES['dropped_elbow']['tip'],
                    catch_elbow
                ))

        # ── Body rotation ──
        if metrics.get('rotation', {}).get('avg_rotation') is not None:
            avg_rotation = metrics['rotation']['avg_rotation']
            rot_conf = metrics['rotation'].get('confidence', 0.3)

            if rot_conf >= 0.3:
                if avg_rotation < BODY_ROTATION_TOO_FLAT:
                    issues.append(FreestyleIssue(
                        'flat_body',
                        ISSUE_TYPES['flat_body']['severity'],
                        f"Limited body rotation ({avg_rotation:.0f}° avg — "
                        f"optimal {BODY_ROTATION_OPTIMAL_MIN}-{BODY_ROTATION_OPTIMAL_MAX}°)",
                        ISSUE_TYPES['flat_body']['tip'],
                        avg_rotation
                    ))
                elif avg_rotation > BODY_ROTATION_TOO_MUCH:
                    issues.append(FreestyleIssue(
                        'over_rotation',
                        ISSUE_TYPES['over_rotation']['severity'],
                        f"Over-rotation ({avg_rotation:.0f}° avg — "
                        f"optimal {BODY_ROTATION_OPTIMAL_MIN}-{BODY_ROTATION_OPTIMAL_MAX}°)",
                        ISSUE_TYPES['over_rotation']['tip'],
                        avg_rotation
                    ))

        # ── Head position ──
        if metrics.get('head', {}).get('vertical_movement') is not None:
            vm = metrics['head']['vertical_movement']
            head_conf = metrics['head'].get('confidence', 0.3)
            if head_conf >= 0.4 and vm > HEAD_LIFT_THRESHOLD:
                issues.append(FreestyleIssue(
                    'head_lifting',
                    ISSUE_TYPES['head_lifting']['severity'],
                    f"Head lifting during breathing ({vm*100:.0f}% vertical movement)",
                    ISSUE_TYPES['head_lifting']['tip'],
                    vm
                ))

        # ── Stroke rate ──
        if metrics.get('stroke_rate', {}).get('spm') is not None:
            spm = metrics['stroke_rate']['spm']
            sr_conf = metrics['stroke_rate'].get('confidence', 0.3)
            if sr_conf >= 0.3:
                if spm < STROKE_RATE_OPTIMAL_MIN:
                    issues.append(FreestyleIssue(
                        'slow_stroke_rate',
                        ISSUE_TYPES['slow_stroke_rate']['severity'],
                        f"Stroke rate below optimal ({spm:.0f} SPM — "
                        f"optimal {STROKE_RATE_OPTIMAL_MIN}-{STROKE_RATE_OPTIMAL_MAX})",
                        ISSUE_TYPES['slow_stroke_rate']['tip'],
                        spm
                    ))
                elif spm > STROKE_RATE_OPTIMAL_MAX:
                    issues.append(FreestyleIssue(
                        'fast_stroke_rate',
                        ISSUE_TYPES['fast_stroke_rate']['severity'],
                        f"Stroke rate above optimal ({spm:.0f} SPM — "
                        f"optimal {STROKE_RATE_OPTIMAL_MIN}-{STROKE_RATE_OPTIMAL_MAX})",
                        ISSUE_TYPES['fast_stroke_rate']['tip'],
                        spm
                    ))

        # ── Kick ──
        if metrics.get('kick', {}).get('avg_knee_angle') is not None:
            avg_knee = metrics['kick']['avg_knee_angle']
            kick_conf = metrics['kick'].get('confidence', 0.3)
            if kick_conf >= 0.4 and avg_knee < KNEE_ANGLE_EXCESSIVE_BEND:
                issues.append(FreestyleIssue(
                    'excessive_knee_bend',
                    ISSUE_TYPES['excessive_knee_bend']['severity'],
                    f"Excessive knee bend ({avg_knee:.0f}° — "
                    f"should be near {KNEE_ANGLE_OPTIMAL}°)",
                    ISSUE_TYPES['excessive_knee_bend']['tip'],
                    avg_knee
                ))

        # ── Consistency across cycles ──
        if metrics.get('cycle_analysis', {}).get('elbow_consistency', {}).get('score') is not None:
            consistency = metrics['cycle_analysis']['elbow_consistency']
            if consistency.get('n_cycles', 0) >= 3 and consistency['score'] < 0.3:
                issues.append(FreestyleIssue(
                    'inconsistent_catch',
                    SEVERITY_MODERATE,
                    f"Inconsistent catch angle across cycles "
                    f"(variability: {consistency['variability']:.2f}, "
                    f"{consistency['n_cycles']} cycles detected)",
                    "Focus on maintaining a consistent elbow position at catch. "
                    "Practice catch drills with a slow tempo to lock in the feel.",
                    consistency['score']
                ))

        # Sort by severity
        severity_order = {SEVERITY_CRITICAL: 0, SEVERITY_MODERATE: 1, SEVERITY_MINOR: 2}
        issues.sort(key=lambda x: severity_order.get(x.severity, 99))

        return issues

    @staticmethod
    def _calc_angle(p1: Dict, p2: Dict, p3: Dict, use_3d: bool = True) -> float:
        """Angle at p2 in degrees.

        Uses (x, y, z) as a 3D vector by default, accounting for foreshortening
        and body roll. Falls back to 2D (x, y) when use_3d=False.
        """
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

    @staticmethod
    def _moving_average(values: List[float], window: int) -> List[float]:
        """Moving average smoother."""
        if window < 2 or len(values) < window:
            return list(values)
        half = window // 2
        smoothed = []
        for i in range(len(values)):
            lo = max(0, i - half)
            hi = min(len(values), i + half + 1)
            smoothed.append(sum(values[lo:hi]) / (hi - lo))
        return smoothed
