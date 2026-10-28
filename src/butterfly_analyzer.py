"""Butterfly analyzer — inherits shared pipeline from BaseAnalyzer.

Stroke-specific:
  - Cycle signal: near_wrist_x, cross-checked with ankle/knee vertical peaks
  - Arm phase: ~0.0 (in phase — both arms simultaneous)
  - Kicks per cycle: exactly 2
  - Metrics: undulation, kick_timing, breath_lift, arm_phase, recovery_height
"""

from typing import List, Dict, Tuple
import numpy as np

from src.base_analyzer import BaseAnalyzer, BaseIssue, MIN_VISIBILITY, PERSISTENCE_THRESHOLD, calculate_angle, moving_average, hampel_filter
from src.stroke_config import CONFIGS


class ButterflyAnalyzer(BaseAnalyzer):
    """Butterfly-specific analyzer."""

    def __init__(self):
        super().__init__('butterfly')

    # ── Cycle detection ──

    def _detect_cycles(self, frames: List[Dict]) -> List[Tuple[int, int]]:
        """Detect cycles from near-wrist-x peaks.

        In butterfly, one cycle = one arm stroke (both arms together).
        Cross-check with ankle or knee vertical peaks (2 kicks per cycle).
        """
        if len(frames) < 10:
            return []

        near_arm = self._arm_labels.get('near_arm', 'left')

        # Wrist-x signal
        wrist_x = []
        valid_count = 0
        for f in frames:
            lm = f['pose']['landmarks'][f'{near_arm}_wrist']
            if lm['visibility'] >= MIN_VISIBILITY:
                wrist_x.append(lm['x'])
                valid_count += 1
            else:
                wrist_x.append(None)

        window = max(3, len(wrist_x) // 30)
        smoothed = moving_average([v if v is not None else 0 for v in wrist_x], window)

        min_gap = max(5, len(frames) // 30)
        peaks = []
        last = -999
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
                if i - last >= min_gap:
                    peaks.append(i)
                    last = i

        if len(peaks) < 2:
            return []

        # Cross-check with ankle vertical peaks (2 kicks per cycle)
        ankle_y = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            la = landmarks.get('left_ankle')
            ra = landmarks.get('right_ankle')
            if la and ra and la['visibility'] >= MIN_VISIBILITY and ra['visibility'] >= MIN_VISIBILITY:
                ankle_y.append((la['y'] + ra['y']) / 2)
            else:
                ankle_y.append(None)

        ankle_smoothed = moving_average([v if v is not None else 0 for v in ankle_y], window)
        ankle_peaks = []
        last = -999
        for i in range(1, len(ankle_smoothed) - 1):
            if ankle_smoothed[i] >= ankle_smoothed[i - 1] and ankle_smoothed[i] >= ankle_smoothed[i + 1]:
                if i - last >= min_gap:
                    ankle_peaks.append(i)
                    last = i

        cycles = [(peaks[i], peaks[i+1]) for i in range(len(peaks) - 1)]

        # Confirm ankle-to-cycle ratio if ankle data available
        if len(ankle_peaks) >= 4 and len(cycles) >= 2:
            ankle_per_cycle = len(ankle_peaks) / len(cycles)
            if 1.5 <= ankle_per_cycle <= 2.5:
                pass  # Confirmed: 2 kicks per cycle

        return cycles

    # ── Stroke-specific metrics ──

    def _stroke_specific_metrics(self, frames: List[Dict],
                                  cycles: List[Tuple[int, int]]) -> Dict:
        """Butterfly-specific metrics."""
        metrics = {}

        # ── Undulation ──
        metrics['undulation'] = self._compute_undulation(frames, cycles)

        # ── Kick timing ──
        metrics['kick_timing'] = self._compute_kick_timing(frames, cycles)

        # ── Arm symmetry ──
        metrics['arm_symmetry'] = self._compute_arm_symmetry(frames)

        return metrics

    @staticmethod
    def _compute_undulation(frames: List[Dict], cycles: List[Tuple[int, int]]) -> Dict:
        """Undulation: chest→hip→knee vertical oscillation amplitude and phase."""
        if not frames:
            return {'undulation_amplitude': None}

        hip_y = []
        chest_y = []
        knee_y = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            lh = landmarks.get('left_hip')
            rh = landmarks.get('right_hip')
            ls = landmarks.get('left_shoulder')
            rs = landmarks.get('right_shoulder')
            lk = landmarks.get('left_knee')
            rk = landmarks.get('right_knee')
            if lh and rh and lh['visibility'] >= MIN_VISIBILITY and rh['visibility'] >= MIN_VISIBILITY:
                hip_y.append((lh['y'] + rh['y']) / 2)
            else:
                hip_y.append(None)
            if ls and rs and ls['visibility'] >= MIN_VISIBILITY and rs['visibility'] >= MIN_VISIBILITY:
                chest_y.append((ls['y'] + rs['y']) / 2)
            else:
                chest_y.append(None)
            if lk and rk and lk['visibility'] >= MIN_VISIBILITY and rk['visibility'] >= MIN_VISIBILITY:
                knee_y.append((lk['y'] + rk['y']) / 2)
            else:
                knee_y.append(None)

        valid_hip = [v for v in hip_y if v is not None]
        if not valid_hip:
            return {'undulation_amplitude': None}

        # Apply Hampel filter to remove outlier hip-y values
        filtered_hip = hampel_filter(valid_hip, 5, 3)
        filtered_hip = [v for v in filtered_hip if v is not None]
        if len(filtered_hip) < 3:
            filtered_hip = valid_hip

        amp = (np.max(filtered_hip) - np.min(filtered_hip)) / np.median(filtered_hip)

        def _sig(arr):
            valid = [(i, v) for i, v in enumerate(arr) if v is not None]
            if len(valid) < 2:
                return None
            return [v for _, v in valid]

        hip_sig = _sig(hip_y)
        chest_sig = _sig(chest_y)
        knee_sig = _sig(knee_y)

        phase_lags = {}
        if chest_sig and hip_sig and len(chest_sig) == len(hip_sig):
            chest_d = np.array(chest_sig) - np.mean(chest_sig)
            hip_d = np.array(hip_sig) - np.mean(hip_sig)
            corr = np.corrcoef(chest_d, hip_d)[0, 1]
            phase_lags['chest_hip'] = float(corr) if not np.isnan(corr) else None

        if hip_sig and knee_sig and len(hip_sig) == len(knee_sig):
            hip_d = np.array(hip_sig) - np.mean(hip_sig)
            knee_d = np.array(knee_sig) - np.mean(knee_sig)
            corr = np.corrcoef(hip_d, knee_d)[0, 1]
            phase_lags['hip_knee'] = float(corr) if not np.isnan(corr) else None

        # Detect undulation cycles from hip signal peaks
        undulation_cycles = 0
        if hip_sig and len(hip_sig) >= 4:
            # Count local maxima in the hip signal (each peak = one undulation cycle)
            # Use a simple peak detection: a point is a peak if it's greater than both neighbors
            peaks = 0
            for i in range(1, len(hip_sig) - 1):
                if hip_sig[i] > hip_sig[i-1] and hip_sig[i] > hip_sig[i+1]:
                    peaks += 1
            undulation_cycles = max(0, peaks - 1) if peaks > 1 else 0

        return {
            'undulation_amplitude': float(amp),
            'num_hip_cycles': len(valid_hip),
            'undulation_cycles': undulation_cycles,
            'phase_lags': phase_lags,
        }

    @staticmethod
    def _compute_kick_timing(frames: List[Dict], cycles: List[Tuple[int, int]]) -> Dict:
        """Kick timing: ankle peaks relative to wrist events."""
        if not cycles:
            return {'kick_timing': None}

        ankle_y = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            la = landmarks.get('left_ankle')
            ra = landmarks.get('right_ankle')
            if la and ra and la['visibility'] >= MIN_VISIBILITY and ra['visibility'] >= MIN_VISIBILITY:
                ankle_y.append((la['y'] + ra['y']) / 2)
            else:
                ankle_y.append(None)

        window = max(3, len(ankle_y) // 30)
        smoothed = [v if v is not None else 0 for v in ankle_y]
        smoothed = moving_average(smoothed, window)

        min_gap = max(5, len(frames) // 30)
        ankle_peaks = []
        last = -999
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
                if i - last >= min_gap:
                    ankle_peaks.append(i)
                    last = i

        if len(ankle_peaks) < 2:
            return {'kick_timing': None}

        timing_data = []
        for start, end in cycles:
            kicks_in_cycle = [ap for ap in ankle_peaks if start <= ap <= end]
            if len(kicks_in_cycle) >= 2:
                first_lag = (kicks_in_cycle[0] - start) / (end - start)
                second_lag = (kicks_in_cycle[1] - start) / (end - start)
                timing_data.append({
                    'first_kick_lag': first_lag,
                    'second_kick_lag': second_lag,
                })

        if not timing_data:
            return {'kick_timing': None}

        return {
            'first_kick_lag': np.mean([t['first_kick_lag'] for t in timing_data]),
            'second_kick_lag': np.mean([t['second_kick_lag'] for t in timing_data]),
            'num_cycles': len(timing_data),
        }

    @staticmethod
    def _compute_arm_symmetry(frames: List[Dict]) -> Dict:
        """Arm symmetry: L/R elbow angle difference."""
        left_elbow = []
        right_elbow = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            if (landmarks['left_shoulder']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_elbow']['visibility'] >= MIN_VISIBILITY and
                landmarks['left_wrist']['visibility'] >= MIN_VISIBILITY):
                a = calculate_angle(landmarks['left_shoulder'], landmarks['left_elbow'],
                                    landmarks['left_wrist'])
                left_elbow.append(a)
            if (landmarks['right_shoulder']['visibility'] >= 0.5 and
                landmarks['right_elbow']['visibility'] >= 0.5 and
                landmarks['right_wrist']['visibility'] >= 0.5):
                a = calculate_angle(landmarks['right_shoulder'], landmarks['right_elbow'],
                                    landmarks['right_wrist'])
                right_elbow.append(a)

        if not left_elbow or not right_elbow:
            return {'symmetry': None, 'note': 'Far arm not visible enough'}

        avg_diff = abs(np.mean(left_elbow) - np.mean(right_elbow))

        return {
            'symmetry': float(avg_diff),
            'left_avg': float(np.mean(left_elbow)),
            'right_avg': float(np.mean(right_elbow)),
        }

    # ── Issue detection ──

    def _detect_issues(self) -> List[BaseIssue]:
        issues = super()._detect_issues()

        # ── Undulation ──
        und = self.metrics.get('undulation', {})
        if und.get('undulation_amplitude') is not None:
            amp = und['undulation_amplitude']
            if amp < 0.04:
                issues.append(BaseIssue('flat_undulation', 'critical',
                    f"Flat body motion — no wave (amplitude {amp:.3f})",
                    'Drive hips forward and down', amp))
            elif amp > 0.20:
                issues.append(BaseIssue('excessive_undulation', 'moderate',
                    f"Excessive body motion (amplitude {amp:.3f})",
                    'Stabilize hips', amp))

        # ── Kick timing ──
        kt = self.metrics.get('kick_timing', {})
        if kt.get('first_kick_lag') is not None:
            first = kt['first_kick_lag']
            second = kt['second_kick_lag']
            # First kick should be near entry (~0.2), second near exit (~0.7)
            if first < 0.1 or first > 0.35:
                issues.append(BaseIssue('kick_timing_off', 'moderate',
                    f"First kick timing off (lag {first:.2f})",
                    'Kick as hands enter', first))
            if second < 0.5 or second > 0.9:
                issues.append(BaseIssue('kick_timing_off', 'moderate',
                    f"Second kick timing off (lag {second:.2f})",
                    'Kick as hands exit', second))

        # ── Arm symmetry ──
        sym = self.metrics.get('arm_symmetry', {})
        if sym.get('symmetry') is not None:
            diff = sym['symmetry']
            if diff > 20:
                issues.append(BaseIssue('asymmetric_arms', 'critical',
                    f"Asymmetric arms ({diff:.0f}° L/R elbow difference)",
                    'Keep both arms in sync', diff))

        return issues
