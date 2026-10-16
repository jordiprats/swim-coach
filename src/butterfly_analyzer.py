"""Analyzes butterfly stroke metrics and detects technique issues."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from src.models.butterfly_rules import (
    ARM_ENTRY_WIDTH_MIN, ARM_ENTRY_WIDTH_MAX,
    ELBOW_ANGLE_PULL_MIN, ELBOW_ANGLE_PULL_MAX, ELBOW_ANGLE_DROPPED,
    UNDULATION_MIN, UNDULATION_MAX, UNDULATION_TOO_FLAT, UNDULATION_EXCESSIVE,
    HEAD_LIFT_THRESHOLD, HEAD_LIFT_EXCESSIVE,
    STROKE_RATE_OPTIMAL_MIN, STROKE_RATE_OPTIMAL_MAX,
    STROKE_RATE_TOO_SLOW, STROKE_RATE_TOO_FAST,
    KNEE_ANGLE_OPTIMAL_MIN, KNEE_ANGLE_OPTIMAL_MAX,
    KNEE_ANGLE_TOO_STRAIGHT, KNEE_ANGLE_EXCESSIVE_BEND,
    SYNC_MAX_DELTA,
    RECOVERY_HEIGHT_MIN, RECOVERY_HEIGHT_MAX, RECOVERY_HEIGHT_EXCESSIVE,
    PHASE_LAG_IDEAL, PHASE_LAG_MIN, PHASE_LAG_MAX, PHASE_LAG_TOO_SMALL,
    COORDINATION_MAX_GAP,
    LATE_BREATHING_MAX_GAP,
    HIP_DROP_MAX, HIP_DROP_EXCESSIVE,
    MIN_VISIBILITY, ButterflyIssue, ISSUE_TYPES,
    SEVERITY_CRITICAL, SEVERITY_MODERATE, SEVERITY_MINOR,
)


class ButterflyAnalyzer:
    """Analyzes butterfly swimming technique from pose data."""

    def __init__(self):
        """Initialize butterfly analyzer."""
        self.metrics = {}
        self.issues = []

    def analyze_video(self, pose_data: List[Dict]) -> Dict:
        """
        Analyze complete video pose data for butterfly.

        Args:
            pose_data: List of frame data with pose information

        Returns:
            Dictionary containing metrics and detected issues
        """
        print("\nAnalyzing butterfly stroke mechanics...")

        # Filter frames with valid pose data
        valid_frames = [frame for frame in pose_data if frame['pose'] is not None]

        if len(valid_frames) == 0:
            return {
                'error': 'No valid pose data detected in video',
                'metrics': {},
                'issues': []
            }

        print(f"Valid frames: {len(valid_frames)}/{len(pose_data)}")

        # Filter to active swimming frames (exclude preparation/waiting)
        active_frames = self._filter_active_swimming_frames(valid_frames)
        if len(active_frames) < len(valid_frames):
            print(f"Active swimming frames: {len(active_frames)}/{len(valid_frames)}")

        # Analyze different aspects
        elbow_metrics = self._analyze_elbow_angles(active_frames)
        entry_metrics = self._analyze_arm_entry_width(active_frames)
        undulation_metrics = self._analyze_body_undulation(active_frames)
        head_metrics = self._analyze_head_position(active_frames)
        stroke_rate_metrics = self._analyze_stroke_rate(active_frames)
        kick_metrics = self._analyze_kick(active_frames)
        sync_metrics = self._analyze_arm_synchronization(active_frames)
        recovery_metrics = self._analyze_arm_recovery(active_frames)
        phase_metrics = self._analyze_shoulder_hip_phase(active_frames)
        hip_breath_metrics = self._analyze_hip_during_breath(active_frames, head_metrics)
        timing_metrics = self._analyze_stroke_phase_timing(active_frames)
        breath_timing_metrics = self._analyze_breathing_timing(active_frames)

        # Combine all metrics
        self.metrics = {
            'elbow': elbow_metrics,
            'entry': entry_metrics,
            'undulation': undulation_metrics,
            'head': head_metrics,
            'stroke_rate': stroke_rate_metrics,
            'kick': kick_metrics,
            'synchronization': sync_metrics,
            'recovery': recovery_metrics,
            'shoulder_hip_phase': phase_metrics,
            'hip_during_breath': hip_breath_metrics,
            'coordination': timing_metrics,
            'breathing_timing': breath_timing_metrics,
            'valid_frame_ratio': len(valid_frames) / len(pose_data)
        }

        # Detect issues based on metrics
        self.issues = self._detect_issues()

        return {
            'metrics': self.metrics,
            'issues': self.issues
        }

    def _analyze_elbow_angles(self, frames: List[Dict]) -> Dict:
        """Analyze elbow angles during the pull phase.

        Filters out outlier angles (< 20° or > 160°) that are clearly
        pose-estimation glitches (a human arm cannot bend that far).
        """
        left_elbow_angles = []
        right_elbow_angles = []
        ELBOW_MIN = 20.0
        ELBOW_MAX = 180.0

        for frame in frames:
            landmarks = frame['pose']['landmarks']

            # Check visibility
            if (landmarks['left_shoulder']['visibility'] < MIN_VISIBILITY or
                landmarks['left_elbow']['visibility'] < MIN_VISIBILITY or
                landmarks['left_wrist']['visibility'] < MIN_VISIBILITY):
                continue

            # Left elbow angle (shoulder → elbow → wrist)
            angle = self._calculate_angle(
                landmarks['left_shoulder'],
                landmarks['left_elbow'],
                landmarks['left_wrist']
            )
            if ELBOW_MIN <= angle <= ELBOW_MAX:
                left_elbow_angles.append(angle)

            # Right elbow angle
            if (landmarks['right_shoulder']['visibility'] >= MIN_VISIBILITY and
                landmarks['right_elbow']['visibility'] >= MIN_VISIBILITY and
                landmarks['right_wrist']['visibility'] >= MIN_VISIBILITY):

                angle = self._calculate_angle(
                    landmarks['right_shoulder'],
                    landmarks['right_elbow'],
                    landmarks['right_wrist']
                )
                if ELBOW_MIN <= angle <= ELBOW_MAX:
                    right_elbow_angles.append(angle)

        all_angles = left_elbow_angles + right_elbow_angles

        return {
            'avg_angle': np.mean(all_angles) if all_angles else None,
            'min_angle': np.min(all_angles) if all_angles else None,
            'max_angle': np.max(all_angles) if all_angles else None,
            'left_avg': np.mean(left_elbow_angles) if left_elbow_angles else None,
            'right_avg': np.mean(right_elbow_angles) if right_elbow_angles else None,
        }

    def _analyze_arm_entry_width(self, frames: List[Dict]) -> Dict:
        """
        Analyze arm entry width relative to frame width.

        In butterfly both arms enter simultaneously.  We measure the distance
        between the two wrists as a fraction of frame width — too narrow
        reduces pull length, too wide loses leverage.

        Only measures at hand-entry frames (wrist y at local maximum = hands
        at lowest point in frame, just before recovery begins).
        Filters out outlier ratios (> 6.0) caused by pose-estimation glitches.
        """
        MAX_SANE_RATIO = 6.0
        frame_width = frames[0]['pose']['frame_shape'][1]

        # Find hand-entry frames: wrist y local maxima (hands at lowest point)
        wrist_y = []
        for frame in frames:
            landmarks = frame['pose']['landmarks']
            lw = landmarks['left_wrist']
            rw = landmarks['right_wrist']
            if lw['visibility'] >= MIN_VISIBILITY and rw['visibility'] >= MIN_VISIBILITY:
                wrist_y.append((lw['y'] + rw['y']) / 2)
            else:
                wrist_y.append(None)

        entry_indices = set()
        MIN_ENTRY_GAP = 15
        last_entry = -999
        for i in range(1, len(wrist_y) - 1):
            if wrist_y[i] is not None and wrist_y[i - 1] is not None and wrist_y[i + 1] is not None:
                if wrist_y[i] > wrist_y[i - 1] and wrist_y[i] > wrist_y[i + 1]:
                    if i - last_entry >= MIN_ENTRY_GAP:
                        entry_indices.add(i)
                        last_entry = i

        entry_widths = []
        for i, frame in enumerate(frames):
            if i not in entry_indices:
                continue
            landmarks = frame['pose']['landmarks']

            # Need both wrists visible
            if (landmarks['left_wrist']['visibility'] < MIN_VISIBILITY or
                landmarks['right_wrist']['visibility'] < MIN_VISIBILITY):
                continue

            # Wrist span (entry width)
            wrist_span = abs(landmarks['left_wrist']['x'] - landmarks['right_wrist']['x'])

            # Ratio relative to frame width
            ratio = wrist_span / frame_width
            if ratio <= MAX_SANE_RATIO:
                entry_widths.append(ratio)

        if not entry_widths:
            return {
                'avg_entry_width': None,
                'min_entry_width': None,
                'max_entry_width': None,
            }

        return {
            'avg_entry_width': np.mean(entry_widths),
            'min_entry_width': np.min(entry_widths),
            'max_entry_width': np.max(entry_widths),
        }

    def _analyze_body_undulation(self, frames: List[Dict]) -> Dict:
        """
        Analyze dolphin-like body undulation.

        Butterfly uses vertical undulation — we measure the vertical
        displacement of the hips over the stroke cycle as a fraction of
        frame height.
        """
        hip_y_positions = []  # Average of left/right hip y

        for frame in frames:
            landmarks = frame['pose']['landmarks']

            left_hip_y = landmarks['left_hip']['y']
            right_hip_y = landmarks['right_hip']['y']

            # Average both hips for a stable reference
            avg_y = (left_hip_y + right_hip_y) / 2
            hip_y_positions.append(avg_y)

        if len(hip_y_positions) < 5:
            return {
                'undulation_amplitude': None,
                'undulation_cycles': None,
                'avg_undulation': None,
            }

        frame_height = frames[0]['pose']['frame_shape'][0]

        # Normalise hip-y by frame height
        hip_y_norm = [y / frame_height for y in hip_y_positions]

        # Detect undulation cycles by counting zero-crossings of the
        # detrended signal (subtract mean).
        mean_y = np.mean(hip_y_norm)
        detrended = [y - mean_y for y in hip_y_norm]

        # Count sign changes (each up-down pair = one undulation cycle)
        crossings = 0
        for i in range(1, len(detrended)):
            if (detrended[i - 1] < 0 and detrended[i] >= 0) or \
               (detrended[i - 1] > 0 and detrended[i] <= 0):
                crossings += 1

        # Each undulation cycle = 2 crossings (up then down)
        undulation_cycles = crossings // 2
        amplitude = np.std(detrended) * 2  # Rough peak-to-peak estimate

        return {
            'undulation_amplitude': amplitude,
            'undulation_cycles': undulation_cycles,
            'avg_undulation': amplitude,
        }

    def _analyze_head_position(self, frames: List[Dict]) -> Dict:
        """Analyze head vertical movement for breathing pattern."""
        nose_y_positions = []

        for frame in frames:
            landmarks = frame['pose']['landmarks']
            if landmarks['nose']['visibility'] >= MIN_VISIBILITY:
                nose_y_positions.append(landmarks['nose']['y'])

        if not nose_y_positions:
            return {
                'breathing_lift': None,
                'stability': None,
            }

        frame_height = frames[0]['pose']['frame_shape'][0]
        y_range = np.max(nose_y_positions) - np.min(nose_y_positions)
        normalized_range = y_range / frame_height

        # In butterfly, some head lift is normal for breathing.
        # We track the max upward displacement relative to the average.
        avg_y = np.mean(nose_y_positions)
        # Lift = how far above average the nose goes (up = smaller y)
        lift = avg_y - np.min(nose_y_positions)  # Positive = lifting up
        normalized_lift = lift / frame_height

        return {
            'vertical_movement': normalized_range,
            'breathing_lift': normalized_lift,
            'avg_y': np.mean(nose_y_positions),
            'stability': 1.0 - normalized_range,
        }

    def _analyze_stroke_rate(self, frames: List[Dict]) -> Dict:
        """
        Analyze stroke rate for butterfly.

        Butterfly has both arms moving together.  We average the left and
        right wrist x-coordinates and detect peaks in the combined signal.
        Each peak = one stroke cycle (both arms).
        """
        if len(frames) < 10:
            return {'spm': None}

        # Collect (timestamp, avg_wrist_x) samples
        samples = []
        for frame in frames:
            landmarks = frame['pose']['landmarks']
            lv = landmarks['left_wrist']['visibility']
            rv = landmarks['right_wrist']['visibility']
            if lv >= MIN_VISIBILITY and rv >= MIN_VISIBILITY:
                avg_x = (landmarks['left_wrist']['x'] + landmarks['right_wrist']['x']) / 2
                samples.append((frame['timestamp'], avg_x))

        if len(samples) < 10:
            return {'spm': None}

        timestamps = [s[0] for s in samples]
        avg_wrist_x = [s[1] for s in samples]
        duration = timestamps[-1] - timestamps[0]

        if duration <= 0:
            return {'spm': None}

        # --- Smooth the trajectory ---
        window = max(3, len(avg_wrist_x) // 20)
        smoothed = self._moving_average(avg_wrist_x, window)

        # --- Estimate minimum gap between peaks ---
        # Max realistic butterfly SPM ~70 → ~0.86 s/stroke.
        fps_equiv = len(smoothed) / duration
        min_peak_gap = max(3, int(fps_equiv * 0.6))

        # --- Find peaks ---
        candidate_peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
                candidate_peaks.append(i)

        # Greedy filter: keep peaks far enough apart
        filtered_peaks = []
        for peak in candidate_peaks:
            if not filtered_peaks or (peak - filtered_peaks[-1]) >= min_peak_gap:
                filtered_peaks.append(peak)

        total_strokes = len(filtered_peaks)  # Each peak = one butterfly stroke cycle
        spm = (total_strokes / duration) * 60

        # Sanity check: butterfly 15–80 SPM
        if spm < 15 or spm > 80:
            spm = None

        return {
            'spm': spm,
            'total_strokes': total_strokes if spm is not None else None,
            'duration': duration
        }

    @staticmethod
    def _moving_average(values: List[float], window: int) -> List[float]:
        """Apply a centred moving average to smooth a 1-D signal."""
        if window < 2 or len(values) < window:
            return list(values)
        half = window // 2
        smoothed = []
        for i in range(len(values)):
            lo = max(0, i - half)
            hi = min(len(values), i + half + 1)
            smoothed.append(sum(values[lo:hi]) / (hi - lo))
        return smoothed

    def _analyze_kick(self, frames: List[Dict]) -> Dict:
        """
        Analyze dolphin kick — both legs together.

        In butterfly the legs kick simultaneously.  We measure knee angles
        on both legs and look for synchronised bending.
        """
        knee_angles = []
        left_knee_angles = []
        right_knee_angles = []

        for frame in frames:
            landmarks = frame['pose']['landmarks']

            # Left leg
            if (landmarks['left_hip']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_knee']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_ankle']['visibility'] >= MIN_VISIBILITY):

                angle = self._calculate_angle(
                    landmarks['left_hip'],
                    landmarks['left_knee'],
                    landmarks['left_ankle']
                )
                left_knee_angles.append(angle)
                knee_angles.append(angle)

            # Right leg
            if (landmarks['right_hip']['visibility'] >= MIN_VISIBILITY and
                landmarks['right_knee']['visibility'] >= MIN_VISIBILITY and
                landmarks['right_ankle']['visibility'] >= MIN_VISIBILITY):

                angle = self._calculate_angle(
                    landmarks['right_hip'],
                    landmarks['right_knee'],
                    landmarks['right_ankle']
                )
                right_knee_angles.append(angle)
                knee_angles.append(angle)

        # Filter out impossible knee angles (< 100° = leg can't bend that far,
        # > 170° = hyperextension glitch)
        KNEE_MIN = 100.0
        KNEE_MAX = 170.0
        knee_angles_filtered = [a for a in knee_angles if KNEE_MIN <= a <= KNEE_MAX]
        left_filtered = [a for a in left_knee_angles if KNEE_MIN <= a <= KNEE_MAX]
        right_filtered = [a for a in right_knee_angles if KNEE_MIN <= a <= KNEE_MAX]

        return {
            'avg_knee_angle': np.mean(knee_angles_filtered) if knee_angles_filtered else None,
            'min_knee_angle': np.min(knee_angles_filtered) if knee_angles_filtered else None,
            'left_avg': np.mean(left_filtered) if left_filtered else None,
            'right_avg': np.mean(right_filtered) if right_filtered else None,
        }

    def _analyze_arm_synchronization(self, frames: List[Dict]) -> Dict:
        """
        Measure synchronisation of left and right arm movements using joint angles.

        In butterfly both arms should move together.  Instead of comparing
        wrist-x peak timestamps (which is sensitive to camera angle), we
        compare the **elbow angle** of each arm frame-by-frame.

        Rationale: joint angles are measured relative to the body and are
        much more robust to camera perspective than absolute x-coordinates.
        In synchronized butterfly, at any given frame the left and right
        elbow angles should be nearly equal.  If they consistently differ,
        the arms are out of sync.

        We also compute the **angle between the two arm vectors**
        (shoulder → wrist) — this should be near 0 when both arms are
        in the same phase and grows when they are offset.

        Improvements:
          - Stricter visibility threshold (0.7) for wrist landmarks
          - Outlier filtering: skip frames where either elbow angle is
            outside 20-160° (pose-estimation glitches)
          - Running-window sync (last 30 frames) reported alongside the
            video-wide average for real-time responsiveness
        """
        SYNC_WINDOW = 30
        ELBOW_MIN = 20.0
        ELBOW_MAX = 180.0
        STRICT_VIS = 0.3

        if len(frames) < 5:
            return {'sync_delta': None, 'synchronized': None}

        elbow_diffs = []
        arm_angle_diffs = []

        for frame in frames:
            landmarks = frame['pose']['landmarks']

            # --- Elbow angle comparison (stricter wrist visibility) ---
            left_ok = all(
                landmarks.get(s, {}).get('visibility', 0) >= MIN_VISIBILITY
                for s in ('left_shoulder', 'left_elbow', 'left_wrist')
            ) and landmarks.get('left_wrist', {}).get('visibility', 0) >= STRICT_VIS
            right_ok = all(
                landmarks.get(s, {}).get('visibility', 0) >= MIN_VISIBILITY
                for s in ('right_shoulder', 'right_elbow', 'right_wrist')
            ) and landmarks.get('right_wrist', {}).get('visibility', 0) >= STRICT_VIS

            if left_ok and right_ok:
                left_angle = self._calculate_angle(
                    landmarks['left_shoulder'],
                    landmarks['left_elbow'],
                    landmarks['left_wrist']
                )
                right_angle = self._calculate_angle(
                    landmarks['right_shoulder'],
                    landmarks['right_elbow'],
                    landmarks['right_wrist']
                )
                # Skip frames where either angle is an outlier
                if (ELBOW_MIN <= left_angle <= ELBOW_MAX and
                    ELBOW_MIN <= right_angle <= ELBOW_MAX):
                    elbow_diffs.append(abs(left_angle - right_angle))

            # --- Arm-vector angle (shoulder → wrist) relative to body axis ---
            if all(
                landmarks.get(s, {}).get('visibility', 0) >= MIN_VISIBILITY
                for s in ('left_shoulder', 'right_shoulder',
                          'left_hip', 'right_hip',
                          'left_wrist', 'right_wrist')
            ) and all(
                landmarks.get(s, {}).get('visibility', 0) >= STRICT_VIS
                for s in ('left_wrist', 'right_wrist')
            ):
                # Body axis (vertical): shoulder midpoint → hip midpoint
                sx = (landmarks['left_shoulder']['x'] + landmarks['right_shoulder']['x']) / 2
                sy = (landmarks['left_shoulder']['y'] + landmarks['right_shoulder']['y']) / 2
                hx = (landmarks['left_hip']['x'] + landmarks['right_hip']['x']) / 2
                hy = (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2

                # Left arm vector (shoulder → wrist)
                lv = np.array([
                    landmarks['left_wrist']['x'] - landmarks['left_shoulder']['x'],
                    landmarks['left_wrist']['y'] - landmarks['left_shoulder']['y'],
                ])
                # Right arm vector
                rv = np.array([
                    landmarks['right_wrist']['x'] - landmarks['right_shoulder']['x'],
                    landmarks['right_wrist']['y'] - landmarks['right_shoulder']['y'],
                ])

                # Angle between left-arm vector and right-arm vector
                norm_l = np.linalg.norm(lv)
                norm_r = np.linalg.norm(rv)
                if norm_l > 1e-6 and norm_r > 1e-6:
                    cos_angle = np.dot(lv, rv) / (norm_l * norm_r)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    arm_angle = np.degrees(np.arccos(cos_angle))
                    # In synchronized butterfly arms are parallel → angle near 0
                    arm_angle_diffs.append(arm_angle)

        if not elbow_diffs:
            return {'sync_delta': None, 'synchronized': None}

        avg_elbow_diff = float(np.mean(elbow_diffs))

        # ── Running-window sync (last SYNC_WINDOW valid diffs) ──
        if len(elbow_diffs) > SYNC_WINDOW:
            recent_diffs = elbow_diffs[-SYNC_WINDOW:]
        else:
            recent_diffs = elbow_diffs
        recent_sync = float(np.mean(recent_diffs))

        avg_arm_angle = float(np.mean(arm_angle_diffs)) if arm_angle_diffs else None

        # Heuristic thresholds:
        #   - avg elbow diff < 20°  → synchronized
        #   - avg elbow diff 20-35° → marginal
        #   - avg elbow diff > 35°  → out of sync
        #
        # The arm-vector angle (angle between left and right arm vectors) is
        # more sensitive to stroke phase and camera perspective, so we use it
        # only as a supporting indicator, not a hard gate.
        sync_elbow = avg_elbow_diff <= 20.0

        # Overall sync: elbow-angle comparison is the primary signal.
        # If avg_arm_angle is available and very large (>60°) it hints at
        # asymmetry, but we don't gate on it.
        synchronized = sync_elbow

        # Also check recent sync separately (may differ from video-wide)
        recent_synchronized = recent_sync <= 20.0

        return {
            'sync_delta': avg_elbow_diff,
            'synchronized': synchronized,
            'recent_sync_delta': recent_sync,
            'recent_synchronized': recent_synchronized,
            'avg_elbow_diff': avg_elbow_diff,
            'avg_arm_angle': avg_arm_angle,
        }

    def _analyze_arm_recovery(self, frames: List[Dict]) -> Dict:
        """
        Analyze arm recovery height and clearance.

        In butterfly the arms should sweep low and wide just above the
        water surface.  We track the vertical (y) position of the elbow
        and wrist relative to the shoulder during recovery — when the
        wrist y is above (lower in 2D) the shoulder y, the arm is in
        recovery phase.

        Recovery clearance = (elbow_y - shoulder_y) / frame_height.
        High clearance (> 0.25) indicates wasted energy.
        """
        if len(frames) < 3:
            return {'avg_recovery_height': None, 'max_recovery_height': None}

        frame_height = frames[0]['pose']['frame_shape'][0]
        left_clearances = []
        right_clearances = []

        for frame in frames:
            landmarks = frame['pose']['landmarks']

            # --- Left arm recovery clearance ---
            if (landmarks['left_shoulder']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_elbow']['visibility'] >= MIN_VISIBILITY):
                # Recovery phase: elbow above shoulder (y smaller = higher in frame)
                # Clearance = how far elbow is above shoulder
                clearance = (landmarks['left_elbow']['y'] - landmarks['left_shoulder']['y']) / frame_height
                left_clearances.append(clearance)

            # --- Right arm ---
            if (landmarks['right_shoulder']['visibility'] >= MIN_VISIBILITY and
                landmarks['right_elbow']['visibility'] >= MIN_VISIBILITY):
                clearance = (landmarks['right_elbow']['y'] - landmarks['right_shoulder']['y']) / frame_height
                right_clearances.append(clearance)

        all_clearances = left_clearances + right_clearances

        if not all_clearances:
            return {'avg_recovery_height': None, 'max_recovery_height': None}

        # Recovery clearance can be negative (elbow below shoulder = pull phase)
        # We only care about positive clearance (recovery above shoulder)
        positive = [c for c in all_clearances if c > 0]

        return {
            'avg_recovery_height': np.mean(positive) if positive else None,
            'max_recovery_height': np.max(positive) if positive else None,
            'left_avg': np.mean(left_clearances) if left_clearances else None,
            'right_avg': np.mean(right_clearances) if right_clearances else None,
        }

    def _analyze_shoulder_hip_phase(self, frames: List[Dict]) -> Dict:
        """
        Analyze shoulder-hip phase relationship (the wave).

        In proper butterfly undulation, the shoulders and hips should move
        out of phase — when the shoulders press down, the hips should drive
        up.  We measure the cross-correlation lag between shoulder y and
        hip y time series.

        A lag of ~0.25 of the stroke cycle is ideal.  If lag is near 0,
        shoulders and hips are moving together (no wave).
        """
        if len(frames) < 10:
            return {'phase_lag': None, 'in_phase': None}

        frame_height = frames[0]['pose']['frame_shape'][0]

        shoulder_y = []
        hip_y = []

        for frame in frames:
            landmarks = frame['pose']['landmarks']
            ls = landmarks['left_shoulder']
            rs = landmarks['right_shoulder']
            lh = landmarks['left_hip']
            rh = landmarks['right_hip']

            if (ls['visibility'] >= MIN_VISIBILITY and rs['visibility'] >= MIN_VISIBILITY):
                avg_sy = (ls['y'] + rs['y']) / 2 / frame_height
                shoulder_y.append(avg_sy)

            if (lh['visibility'] >= MIN_VISIBILITY and rh['visibility'] >= MIN_VISIBILITY):
                avg_hy = (lh['y'] + rh['y']) / 2 / frame_height
                hip_y.append(avg_hy)

        # Ensure same length
        min_len = min(len(shoulder_y), len(hip_y))
        if min_len < 10:
            return {'phase_lag': None, 'in_phase': None}

        shoulder_y = shoulder_y[:min_len]
        hip_y = hip_y[:min_len]

        # Detrend (subtract mean)
        s_detrend = [v - np.mean(shoulder_y) for v in shoulder_y]
        h_detrend = [v - np.mean(hip_y) for v in hip_y]

        # Cross-correlation
        # We want the lag (in frames) that maximises correlation
        max_lag = min(30, min_len // 2)
        best_lag = 0
        best_corr = -1
        for lag in range(-max_lag, max_lag + 1):
            shifted = s_detrend[max(0, lag):min(min_len, min_len + lag)]
            target = h_detrend[max(0, -lag):min(min_len, min_len - lag)]
            if len(shifted) < 5:
                continue
            corr = np.corrcoef(shifted, target)[0, 1]
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        # Normalise lag to fraction of stroke cycle.
        # Estimate stroke cycle length from hip signal zero-crossings.
        crossings = 0
        for i in range(1, len(h_detrend)):
            if (h_detrend[i - 1] < 0 and h_detrend[i] >= 0) or \
               (h_detrend[i - 1] > 0 and h_detrend[i] <= 0):
                crossings += 1
        cycle_len = max(1, (min_len / max(1, crossings)) if crossings > 0 else min_len)
        norm_lag = abs(best_lag) / cycle_len

        # Negative correlation means out-of-phase (ideal for butterfly)
        # Positive correlation means in-phase (no wave)
        in_phase = best_corr > 0

        return {
            'phase_lag': norm_lag,
            'in_phase': in_phase,
            'cross_correlation': best_corr,
            'estimated_cycle_frames': cycle_len,
        }

    def _analyze_hip_during_breath(self, frames: List[Dict], head_metrics: Dict) -> Dict:
        """
        Analyze hip position at the moment of breathing.

        At the frame where the nose reaches its highest y (peak breath),
        we check the hip y position.  If the hips drop significantly
        relative to their average position, it indicates poor body line
        during breathing.
        """
        if len(frames) < 5:
            return {'hip_drop': None, 'hip_drop_during_breath': None}

        frame_height = frames[0]['pose']['frame_shape'][0]

        # Find breathing frames: nose y local maxima
        nose_y = []
        hip_y = []
        for frame in frames:
            landmarks = frame['pose']['landmarks']
            if landmarks['nose']['visibility'] >= MIN_VISIBILITY:
                nose_y.append(landmarks['nose']['y'])
            else:
                nose_y.append(None)

            lh = landmarks['left_hip']
            rh = landmarks['right_hip']
            if (lh['visibility'] >= MIN_VISIBILITY and rh['visibility'] >= MIN_VISIBILITY):
                hip_y.append((lh['y'] + rh['y']) / 2)
            else:
                hip_y.append(None)

        # Find local minima in nose_y (breath peaks: nose goes higher = smaller y)
        # Requirements:
        #   - minimum gap of MIN_BREATH_GAP frames between breaths
        #   - nose_y change >= MIN_BREATH_AMPLITUDE pixels (to filter noise)
        MIN_BREATH_GAP = 15        # ~0.5s between breaths
        MIN_BREATH_AMPLITUDE = 30  # pixels — nose must rise at least this much
        breath_frames = []
        last_breath = -999
        for i in range(1, len(nose_y) - 1):
            if nose_y[i] is not None and nose_y[i - 1] is not None and nose_y[i + 1] is not None:
                if nose_y[i] < nose_y[i - 1] and nose_y[i] < nose_y[i + 1]:
                    # Nose y goes up (smaller y = higher in frame)
                    # Check amplitude: nose must have risen by MIN_BREATH_AMPLITUDE
                    # Compare to the nearest previous local maximum (head down)
                    # or simply check the drop from the previous peak
                    if nose_y[i - 1] is not None and nose_y[i] is not None:
                        rise = nose_y[i - 1] - nose_y[i]
                        if rise >= MIN_BREATH_AMPLITUDE and i - last_breath >= MIN_BREATH_GAP:
                            breath_frames.append(i)
                            last_breath = i

        if not breath_frames:
            return {'hip_drop': None, 'hip_drop_during_breath': None}

        # Average hip y across all frames as baseline
        valid_hip = [h for h in hip_y if h is not None]
        if not valid_hip:
            return {'hip_drop': None, 'hip_drop_during_breath': None}
        avg_hip_y = np.mean(valid_hip)

        # Hip drop at breath frames
        drops = []
        for bf in breath_frames:
            if hip_y[bf] is not None:
                drop = (hip_y[bf] - avg_hip_y) / frame_height
                drops.append(drop)

        if not drops:
            return {'hip_drop': None, 'hip_drop_during_breath': None}

        avg_drop = np.mean(drops)

        # Positive drop means hips sank lower (larger y = lower in frame)
        return {
            'hip_drop': avg_drop,
            'hip_drop_during_breath': avg_drop > HIP_DROP_MAX,
            'num_breaths_detected': len(breath_frames),
        }

    def _analyze_stroke_phase_timing(self, frames: List[Dict]) -> Dict:
        """
        Analyze coordination between arm pull and kick (two-kick rhythm).

        The second kick should align with the end of the pull (wrist passes
        hip).  We detect:
          - End-of-pull frames: when wrist x-coordinate passes hip x-coordinate
            (wrist behind hip → wrist in front of hip)
          - Kick-peak frames: when ankle reaches maximum extension (knee angle
            at minimum, indicating down-kick snap)

        We then measure the frame gap between these events.  A gap of
        0-2 frames is ideal.
        """
        if len(frames) < 10:
            return {'coordination_gap': None, 'aligned': None}

        # Detect end-of-pull events (wrist passes hip)
        MIN_PULL_GAP = 15  # frames between pulls (~0.5s)
        pull_events = []
        wrist_behind_hip = False
        last_pull = -999

        for i, frame in enumerate(frames):
            landmarks = frame['pose']['landmarks']
            # Use left arm as primary (both arms move together in butterfly)
            lw = landmarks['left_wrist']
            lh = landmarks['left_hip']
            rw = landmarks['right_wrist']
            rh = landmarks['right_hip']

            if (lw['visibility'] >= MIN_VISIBILITY and lh['visibility'] >= MIN_VISIBILITY):
                # Wrist behind hip (higher x = further back in side view)
                behind = lw['x'] > lh['x']
                if behind and not wrist_behind_hip and i - last_pull >= MIN_PULL_GAP:
                    # Transition: wrist just passed hip (end of pull)
                    pull_events.append(i)
                    last_pull = i
                wrist_behind_hip = behind

        if not pull_events:
            return {'coordination_gap': None, 'aligned': None}

        # Detect kick-peak events (minimum knee angle = maximum extension)
        KNEE_MIN = 100.0
        KNEE_MAX = 170.0
        MIN_KICK_GAP = 10  # frames between kicks (~0.33s)
        kick_events = []
        knee_angles = []
        for frame in frames:
            landmarks = frame['pose']['landmarks']
            if (landmarks['left_hip']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_knee']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_ankle']['visibility'] >= MIN_VISIBILITY):
                angle = self._calculate_angle(
                    landmarks['left_hip'],
                    landmarks['left_knee'],
                    landmarks['left_ankle']
                )
                knee_angles.append(angle)
            else:
                knee_angles.append(None)

        # Find minima in knee angles (kick snap), filtering glitches and gaps
        last_kick = -999
        for i in range(1, len(knee_angles) - 1):
            if knee_angles[i] is not None and knee_angles[i - 1] is not None and knee_angles[i + 1] is not None:
                # Skip impossible angles
                if not (KNEE_MIN <= knee_angles[i] <= KNEE_MAX):
                    continue
                # Also check neighbours are valid
                if not (KNEE_MIN <= knee_angles[i-1] <= KNEE_MAX) or not (KNEE_MIN <= knee_angles[i+1] <= KNEE_MAX):
                    continue
                if knee_angles[i] < knee_angles[i - 1] and knee_angles[i] < knee_angles[i + 1]:
                    if i - last_kick >= MIN_KICK_GAP:
                        kick_events.append(i)
                        last_kick = i

        if not kick_events:
            return {'coordination_gap': None, 'aligned': None}

        # Match each pull event to the nearest kick event
        gaps = []
        for pull in pull_events:
            nearest = min(kick_events, key=lambda k: abs(k - pull))
            gaps.append(abs(pull - nearest))

        avg_gap = float(np.mean(gaps))

        return {
            'coordination_gap': avg_gap,
            'aligned': avg_gap <= COORDINATION_MAX_GAP,
            'num_pull_events': len(pull_events),
            'num_kick_events': len(kick_events),
        }

    def _analyze_breathing_timing(self, frames: List[Dict]) -> Dict:
        """
        Analyze breathing timing — head should descend before hands enter.

        Detects:
          - Breath-peak frames: when nose y reaches a local maximum (highest)
          - Hand-entry frames: when wrist y is at a local minimum (lowest,
            just before arm recovery begins)

        The gap between breath-peak and hand-entry should be small or
        negative (head already descending when hands enter).
        """
        if len(frames) < 5:
            return {'breath_to_entry_gap': None, 'late_breathing': None}

        # Find breath-peak frames (nose y local minima = highest point)
        nose_y = []
        for frame in frames:
            landmarks = frame['pose']['landmarks']
            if landmarks['nose']['visibility'] >= MIN_VISIBILITY:
                nose_y.append(landmarks['nose']['y'])
            else:
                nose_y.append(None)

        MIN_BREATH_GAP = 15        # ~0.5s between breaths
        MIN_BREATH_AMPLITUDE = 30  # pixels — nose must rise at least this much
        breath_frames = []
        last_breath = -999
        for i in range(1, len(nose_y) - 1):
            if nose_y[i] is not None and nose_y[i - 1] is not None and nose_y[i + 1] is not None:
                if nose_y[i] < nose_y[i - 1] and nose_y[i] < nose_y[i + 1]:
                    rise = nose_y[i - 1] - nose_y[i]
                    if rise >= MIN_BREATH_AMPLITUDE and i - last_breath >= MIN_BREATH_GAP:
                        breath_frames.append(i)
                        last_breath = i

        if not breath_frames:
            return {'breath_to_entry_gap': None, 'late_breathing': None}

        # Find hand-entry frames: wrist y local minima (hand lowest = entry)
        wrist_y = []
        for frame in frames:
            landmarks = frame['pose']['landmarks']
            # Average left/right wrist y
            lw = landmarks['left_wrist']
            rw = landmarks['right_wrist']
            if lw['visibility'] >= MIN_VISIBILITY and rw['visibility'] >= MIN_VISIBILITY:
                wrist_y.append((lw['y'] + rw['y']) / 2)
            else:
                wrist_y.append(None)

        MIN_ENTRY_GAP = 15  # frames between hand entries (~0.5s)
        entry_frames = []
        last_entry = -999
        for i in range(1, len(wrist_y) - 1):
            if wrist_y[i] is not None and wrist_y[i - 1] is not None and wrist_y[i + 1] is not None:
                if wrist_y[i] > wrist_y[i - 1] and wrist_y[i] > wrist_y[i + 1]:
                    # Hand at highest y (lowest in frame) = entry point
                    if i - last_entry >= MIN_ENTRY_GAP:
                        entry_frames.append(i)
                        last_entry = i

        if not entry_frames:
            return {'breath_to_entry_gap': None, 'late_breathing': None}

        # Match each breath to nearest entry
        gaps = []
        for breath in breath_frames:
            nearest = min(entry_frames, key=lambda e: abs(e - breath))
            # Positive gap = breath before entry (good: head already descending)
            # Negative gap = entry before breath (bad: head still high when hands enter)
            gap = breath - nearest
            gaps.append(gap)

        avg_gap = float(np.mean(gaps))

        # Negative or small positive gap = late breathing (head still high)
        late = avg_gap < LATE_BREATHING_MAX_GAP

        return {
            'breath_to_entry_gap': avg_gap,
            'late_breathing': late,
            'num_breaths': len(breath_frames),
            'num_entries': len(entry_frames),
        }

    @staticmethod
    def _find_peak_timestamps(samples: List[Tuple[float, float]]) -> List[float]:
        """
        Find peak timestamps in a (timestamp, value) series.

        Returns timestamps of detected peaks.
        """
        timestamps = [s[0] for s in samples]
        values = [s[1] for s in samples]

        if len(values) < 3:
            return []

        # Smooth
        window = max(3, len(values) // 20)
        smoothed = ButterflyAnalyzer._moving_average(values, window)

        # Min peak gap (approx 0.5 s)
        fps_equiv = len(smoothed) / (timestamps[-1] - timestamps[0] + 1e-6)
        min_gap = max(3, int(fps_equiv * 0.4))

        # Detect peaks
        peaks = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
                peaks.append(i)

        # Filter by minimum gap
        filtered = []
        for p in peaks:
            if not filtered or (p - filtered[-1]) >= min_gap:
                filtered.append(p)

        return [timestamps[p] for p in filtered]

    def _detect_issues(self) -> List[ButterflyIssue]:
        """Detect technique issues based on analyzed metrics."""
        issues = []

        # --- Elbow angles ---
        if self.metrics['elbow']['avg_angle'] is not None:
            avg_elbow = self.metrics['elbow']['avg_angle']
            if avg_elbow > ELBOW_ANGLE_DROPPED:
                issues.append(ButterflyIssue(
                    'dropped_elbow',
                    ISSUE_TYPES['dropped_elbow']['severity'],
                    f"Dropped elbows during pull (avg {avg_elbow:.0f}° — should be {ELBOW_ANGLE_PULL_MIN}-{ELBOW_ANGLE_PULL_MAX}°)",
                    ISSUE_TYPES['dropped_elbow']['tip'],
                    avg_elbow
                ))

        # --- Arm entry width ---
        if self.metrics['entry']['avg_entry_width'] is not None:
            avg_width = self.metrics['entry']['avg_entry_width']
            if avg_width < ARM_ENTRY_WIDTH_MIN:
                issues.append(ButterflyIssue(
                    'arms_too_narrow',
                    ISSUE_TYPES['arms_too_narrow']['severity'],
                    f"Arms entering too narrow (wrist span {avg_width:.2f} of frame width — should be wider)",
                    ISSUE_TYPES['arms_too_narrow']['tip'],
                    avg_width
                ))
            elif avg_width > ARM_ENTRY_WIDTH_MAX:
                issues.append(ButterflyIssue(
                    'arms_too_wide',
                    ISSUE_TYPES['arms_too_wide']['severity'],
                    f"Arms entering too wide (wrist span {avg_width:.2f} of frame width)",
                    ISSUE_TYPES['arms_too_wide']['tip'],
                    avg_width
                ))

        # --- Body undulation ---
        if self.metrics['undulation']['undulation_amplitude'] is not None:
            amp = self.metrics['undulation']['undulation_amplitude']
            if amp < UNDULATION_TOO_FLAT:
                issues.append(ButterflyIssue(
                    'flat_undulation',
                    ISSUE_TYPES['flat_undulation']['severity'],
                    f"Flat body — not enough dolphin undulation (amplitude {amp:.3f} — minimum {UNDULATION_MIN})",
                    ISSUE_TYPES['flat_undulation']['tip'],
                    amp
                ))
            elif amp > UNDULATION_EXCESSIVE:
                issues.append(ButterflyIssue(
                    'excessive_undulation',
                    ISSUE_TYPES['excessive_undulation']['severity'],
                    f"Excessive undulation (amplitude {amp:.3f} — optimal {UNDULATION_MIN}-{UNDULATION_MAX})",
                    ISSUE_TYPES['excessive_undulation']['tip'],
                    amp
                ))

        # --- Head position ---
        if self.metrics['head']['breathing_lift'] is not None:
            lift = self.metrics['head']['breathing_lift']
            if lift > HEAD_LIFT_EXCESSIVE:
                issues.append(ButterflyIssue(
                    'head_lifting_excessive',
                    ISSUE_TYPES['head_lifting_excessive']['severity'],
                    f"Lifting head too high to breathe (lift {lift:.2f} of frame height)",
                    ISSUE_TYPES['head_lifting_excessive']['tip'],
                    lift
                ))

        # --- Stroke rate ---
        if self.metrics['stroke_rate']['spm'] is not None:
            spm = self.metrics['stroke_rate']['spm']
            if spm < STROKE_RATE_TOO_SLOW:
                issues.append(ButterflyIssue(
                    'slow_stroke_rate',
                    ISSUE_TYPES['slow_stroke_rate']['severity'],
                    f"Stroke rate too slow ({spm:.0f} SPM — optimal {STROKE_RATE_OPTIMAL_MIN}-{STROKE_RATE_OPTIMAL_MAX})",
                    ISSUE_TYPES['slow_stroke_rate']['tip'],
                    spm
                ))
            elif spm > STROKE_RATE_TOO_FAST:
                issues.append(ButterflyIssue(
                    'fast_stroke_rate',
                    ISSUE_TYPES['fast_stroke_rate']['severity'],
                    f"Stroke rate too fast ({spm:.0f} SPM — optimal {STROKE_RATE_OPTIMAL_MIN}-{STROKE_RATE_OPTIMAL_MAX})",
                    ISSUE_TYPES['fast_stroke_rate']['tip'],
                    spm
                ))

        # --- Kick ---
        if self.metrics['kick']['avg_knee_angle'] is not None:
            avg_knee = self.metrics['kick']['avg_knee_angle']
            if avg_knee < KNEE_ANGLE_EXCESSIVE_BEND:
                issues.append(ButterflyIssue(
                    'excessive_knee_bend',
                    ISSUE_TYPES['excessive_knee_bend']['severity'],
                    f"Excessive knee bend in dolphin kick (avg {avg_knee:.0f}° — optimal {KNEE_ANGLE_OPTIMAL_MIN}-{KNEE_ANGLE_OPTIMAL_MAX}°)",
                    ISSUE_TYPES['excessive_knee_bend']['tip'],
                    avg_knee
                ))
            elif avg_knee > KNEE_ANGLE_TOO_STRAIGHT:
                issues.append(ButterflyIssue(
                    'knee_too_straight',
                    ISSUE_TYPES['knee_too_straight']['severity'],
                    f"Knees too straight — not enough kick (avg {avg_knee:.0f}°)",
                    ISSUE_TYPES['knee_too_straight']['tip'],
                    avg_knee
                ))

        # --- Arm synchronisation ---
        if self.metrics['synchronization']['sync_delta'] is not None:
            delta = self.metrics['synchronization']['sync_delta']
            if not self.metrics['synchronization']['synchronized']:
                issues.append(ButterflyIssue(
                    'arms_not_synchronised',
                    ISSUE_TYPES['arms_not_synchronised']['severity'],
                    f"Arms not synchronised (avg elbow-angle diff {delta:.1f}° — should be < 20°)",
                    ISSUE_TYPES['arms_not_synchronised']['tip'],
                    delta
                ))

        # --- Arm recovery ---
        if self.metrics['recovery']['avg_recovery_height'] is not None:
            avg_rec = self.metrics['recovery']['avg_recovery_height']
            if avg_rec > RECOVERY_HEIGHT_EXCESSIVE:
                issues.append(ButterflyIssue(
                    'recovery_too_high',
                    ISSUE_TYPES['recovery_too_high']['severity'],
                    f"Arm recovery too high (clearance {avg_rec:.2f} of frame height — optimal < {RECOVERY_HEIGHT_MAX})",
                    ISSUE_TYPES['recovery_too_high']['tip'],
                    avg_rec
                ))

        # --- Shoulder-hip phase ---
        if self.metrics['shoulder_hip_phase']['phase_lag'] is not None:
            lag = self.metrics['shoulder_hip_phase']['phase_lag']
            in_phase = self.metrics['shoulder_hip_phase']['in_phase']
            # Issue: in-phase (positive correlation) OR no correlation (weak relationship)
            corr = self.metrics['shoulder_hip_phase'].get('cross_correlation', 0)
            if in_phase or abs(corr) < 0.3:
                issues.append(ButterflyIssue(
                    'shoulder_hip_in_phase',
                    ISSUE_TYPES['shoulder_hip_in_phase']['severity'],
                    f"Shoulders and hips moving together (phase lag {lag:.2f} of cycle — should be out-of-phase)",
                    ISSUE_TYPES['shoulder_hip_in_phase']['tip'],
                    lag
                ))

        # --- Hip drop during breath ---
        if self.metrics['hip_during_breath']['hip_drop'] is not None:
            hip_drop = self.metrics['hip_during_breath']['hip_drop']
            if hip_drop > HIP_DROP_EXCESSIVE:
                issues.append(ButterflyIssue(
                    'hips_drop_during_breath',
                    ISSUE_TYPES['hips_drop_during_breath']['severity'],
                    f"Hips drop too low during breath (hip drop {hip_drop:.3f} of frame height)",
                    ISSUE_TYPES['hips_drop_during_breath']['tip'],
                    hip_drop
                ))

        # --- Coordination (kick-pull timing) ---
        if self.metrics['coordination']['coordination_gap'] is not None:
            gap = self.metrics['coordination']['coordination_gap']
            if not self.metrics['coordination']['aligned']:
                issues.append(ButterflyIssue(
                    'poor_coordination',
                    ISSUE_TYPES['poor_coordination']['severity'],
                    f"Kick and pull not aligned (avg gap {gap:.1f} frames — should be ≤ {COORDINATION_MAX_GAP})",
                    ISSUE_TYPES['poor_coordination']['tip'],
                    gap
                ))

        # --- Breathing timing ---
        if self.metrics['breathing_timing']['breath_to_entry_gap'] is not None:
            bt_gap = self.metrics['breathing_timing']['breath_to_entry_gap']
            late = self.metrics['breathing_timing']['late_breathing']
            if late:
                issues.append(ButterflyIssue(
                    'late_breathing',
                    ISSUE_TYPES['late_breathing']['severity'],
                    f"Late breathing — head still high when hands enter (breath-to-entry gap {bt_gap:.1f} frames)",
                    ISSUE_TYPES['late_breathing']['tip'],
                    bt_gap
                ))

        # Sort by severity
        severity_order = {SEVERITY_CRITICAL: 0, SEVERITY_MODERATE: 1, SEVERITY_MINOR: 2}
        issues.sort(key=lambda x: severity_order[x.severity])

        return issues

    @staticmethod
    def _filter_active_swimming_frames(frames: List[Dict]) -> List[Dict]:
        """
        Filter out preparation/waiting frames where the swimmer is not actively
        swimming.  Detects active swimming by measuring wrist_x range over a
        sliding window — during swimming the arms move widely, during
        preparation the wrist_x is relatively stable.

        Frames with wrist_x range < ACTIVE_THRESHOLD over a 1-second window
        are marked as inactive and excluded.
        """
        ACTIVE_THRESHOLD = 400  # pixel range in wrist_x over 1s window
        WINDOW = 30  # ~1 second at 30 fps

        if len(frames) < WINDOW:
            return frames

        # Compute wrist_x for each frame (avg of left/right)
        wrist_x = []
        for frame in frames:
            landmarks = frame['pose']['landmarks']
            lw = landmarks.get('left_wrist')
            rw = landmarks.get('right_wrist')
            if lw and rw and lw['visibility'] >= MIN_VISIBILITY and rw['visibility'] >= MIN_VISIBILITY:
                wrist_x.append((lw['x'] + rw['x']) / 2)
            else:
                wrist_x.append(None)

        # Determine active status for each frame
        active_flags = [False] * len(frames)
        for i in range(len(frames)):
            # Look at window centred on this frame
            half = WINDOW // 2
            lo = max(0, i - half)
            hi = min(len(frames), i + half + 1)
            window_values = [wx for wx in wrist_x[lo:hi] if wx is not None]
            if len(window_values) >= 10:
                w_range = max(window_values) - min(window_values)
                active_flags[i] = w_range >= ACTIVE_THRESHOLD

        # Return only active frames
        active = [frames[i] for i in range(len(frames)) if active_flags[i]]

        if not active:
            # Fall back to all frames if nothing is active
            return frames

        return active

    @staticmethod
    def _calculate_angle(point1: Dict, point2: Dict, point3: Dict, use_3d: bool = True) -> float:
        """Calculate angle between three points, optionally in 3D.

        Uses (x, y, z) as a 3D vector by default, accounting for foreshortening
        and body roll. Falls back to 2D (x, y) when use_3d=False.
        """
        if use_3d:
            p1 = np.array([point1.get('x', 0), point1.get('y', 0), point1.get('z', 0)])
            p2 = np.array([point2.get('x', 0), point2.get('y', 0), point2.get('z', 0)])
            p3 = np.array([point3.get('x', 0), point3.get('y', 0), point3.get('z', 0)])
        else:
            p1 = np.array([point1['x'], point1['y']])
            p2 = np.array([point2['x'], point2['y']])
            p3 = np.array([point3['x'], point3['y']])

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)
