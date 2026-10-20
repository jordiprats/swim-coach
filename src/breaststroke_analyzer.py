"""Breaststroke analyzer — inherits shared pipeline from BaseAnalyzer.

Stroke-specific:
  - Cycle signal: wrist→shoulder distance (hand separation)
  - Arm phase: ~0.0 (simultaneous arms)
  - Kicks per cycle: exactly 1
  - Metrics: glide_length, kick_timing, heel_clearance, arm_phase, breath_lift
"""

from typing import List, Dict, Tuple
import numpy as np

from src.base_analyzer import BaseAnalyzer, BaseIssue, MIN_VISIBILITY, PERSISTENCE_THRESHOLD, calculate_angle
from src.stroke_config import CONFIGS


class BreaststrokeAnalyzer(BaseAnalyzer):
    """Breaststroke-specific analyzer."""

    def __init__(self):
        super().__init__('breaststroke')

    # ── Cycle detection ──

    def _detect_cycles(self, frames: List[Dict]) -> List[Tuple[int, int]]:
        """Detect cycles from wrist→shoulder distance peaks.

        In breaststroke, hands come together (pull) then shoot forward (glide).
        Cycle = hands fully forward → next hands fully forward.
        """
        if len(frames) < 10:
            return []

        near_arm = self._arm_labels.get('near_arm', 'left')

        # Wrist-to-shoulder distance (proxy for hand position)
        wrist_dist = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            wr = landmarks[f'{near_arm}_wrist']
            sh = landmarks.get('left_shoulder')
            rh = landmarks.get('right_shoulder')
            if (wr['visibility'] >= MIN_VISIBILITY and
                sh and rh and sh['visibility'] >= MIN_VISIBILITY and rh['visibility'] >= MIN_VISIBILITY):
                sx = (sh['x'] + rh['x']) / 2
                dx = wr['x'] - sx
                wrist_dist.append(dx)
            else:
                wrist_dist.append(None)

        window = max(3, len(wrist_dist) // 30)
        smoothed = self._moving_average([v if v is not None else 0 for v in wrist_dist], window)

        # Peaks = hands fully forward (max distance ahead of shoulder)
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
        return [(peaks[i], peaks[i+1]) for i in range(len(peaks) - 1)]

    # ── Stroke-specific metrics ──

    def _stroke_specific_metrics(self, frames: List[Dict],
                                  cycles: List[Tuple[int, int]]) -> Dict:
        """Breaststroke-specific metrics."""
        metrics = {}

        # ── Glide length ──
        metrics['glide'] = self._compute_glide(frames, cycles)

        # ── Kick timing ──
        metrics['kick_timing'] = self._compute_kick_timing(frames, cycles)

        # ── Heel clearance ──
        metrics['heel_clearance'] = self._compute_heel_clearance(frames)

        # ── Arm symmetry ──
        metrics['arm_symmetry'] = self._compute_arm_symmetry(frames)

        return metrics

    @staticmethod
    def _compute_glide(frames: List[Dict], cycles: List[Tuple[int, int]]) -> Dict:
        """Glide length: time between hand separation and kick, relative to cycle."""
        if not cycles:
            return {'glide_ratio': None}

        near_arm = 'left'  # fallback — caller should use arm_labels
        glide_data = []
        for start, end in cycles:
            landmarks_cycle = [frames[i]['pose']['landmarks'] for i in range(start, min(end + 1, len(frames) - 1))]
            if not landmarks_cycle:
                continue

            # Detect pull phase (hands together): wrist-x close to each other
            # Detect kick: ankle-y sudden drop
            # Glide = time between end of pull and start of kick
            # Simple approach: hands closest together → kick event
            hand_gap = []
            for lm in landmarks_cycle:
                lw = lm.get('left_wrist')
                rw = lm.get('right_wrist')
                if lw and rw and lw['visibility'] >= MIN_VISIBILITY and rw['visibility'] >= MIN_VISIBILITY:
                    hand_gap.append(abs(lw['x'] - rw['x']))
                else:
                    hand_gap.append(None)

            if not hand_gap:
                continue
            valid = [(i, v) for i, v in enumerate(hand_gap) if v is not None]
            if len(valid) < 3:
                continue
            min_idx = min(valid, key=lambda x: x[1])[0]
            # Kick event = ankle-y minimum (peak kick)
            ankle_y = []
            for i in range(start, min(end + 1, len(frames) - 1)):
                la = frames[i]['pose']['landmarks'].get('left_ankle')
                ra = frames[i]['pose']['landmarks'].get('right_ankle')
                if la and ra and la['visibility'] >= MIN_VISIBILITY and ra['visibility'] >= MIN_VISIBILITY:
                    ankle_y.append((la['y'] + ra['y']) / 2)
                else:
                    ankle_y.append(None)
            valid_a = [(i, v) for i, v in enumerate(ankle_y) if v is not None]
            if len(valid_a) < 3:
                continue
            kick_idx = min(valid_a, key=lambda x: x[1])[0]

            glide_len = kick_idx - min_idx
            cycle_len = end - start
            if cycle_len > 0 and glide_len > 0:
                glide_data.append(glide_len / cycle_len)

        return {
            'glide_ratio': np.mean(glide_data) if glide_data else None,
            'num_cycles': len(glide_data),
        }

    @staticmethod
    def _compute_kick_timing(frames: List[Dict], cycles: List[Tuple[int, int]]) -> Dict:
        """Kick timing: ankle minimum relative to cycle."""
        if not cycles:
            return {'kick_lag': None}

        timing_data = []
        for start, end in cycles:
            ankle_y = []
            for i in range(start, min(end + 1, len(frames) - 1)):
                la = frames[i]['pose']['landmarks'].get('left_ankle')
                ra = frames[i]['pose']['landmarks'].get('right_ankle')
                if la and ra and la['visibility'] >= MIN_VISIBILITY and ra['visibility'] >= MIN_VISIBILITY:
                    ankle_y.append((la['y'] + ra['y']) / 2)
                else:
                    ankle_y.append(None)
            valid = [(i, v) for i, v in enumerate(ankle_y) if v is not None]
            if len(valid) < 3:
                continue
            min_idx = min(valid, key=lambda x: x[1])[0]
            lag = (min_idx - start) / (end - start) if (end - start) > 0 else 0.5
            timing_data.append(lag)

        return {
            'kick_lag': np.mean(timing_data) if timing_data else None,
            'num_cycles': len(timing_data),
        }

    @staticmethod
    def _compute_heel_clearance(frames: List[Dict]) -> Dict:
        """Heel clearance: ankle-y relative to hip-y, torso-normalised."""
        clearances = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            la = landmarks.get('left_ankle')
            ra = landmarks.get('right_ankle')
            lh = landmarks.get('left_hip')
            rh = landmarks.get('right_hip')
            if (la and ra and lh and rh and
                la['visibility'] >= MIN_VISIBILITY and ra['visibility'] >= MIN_VISIBILITY and
                lh['visibility'] >= MIN_VISIBILITY and rh['visibility'] >= MIN_VISIBILITY):
                ay = (la['y'] + ra['y']) / 2
                hy = (lh['y'] + rh['y']) / 2
                clearance = ay - hy
                clearances.append(clearance)

        return {
            'avg_heel_clearance': np.mean(clearances) if clearances else None,
            'max_heel_clearance': np.max(clearances) if clearances else None,
        }

    @staticmethod
    def _compute_arm_symmetry(frames: List[Dict]) -> Dict:
        """Arm symmetry: L/R wrist-x separation during pull."""
        if len(frames) < 5:
            return {'symmetry': None}

        hand_gaps = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            lw = landmarks.get('left_wrist')
            rw = landmarks.get('right_wrist')
            if lw and rw and lw['visibility'] >= MIN_VISIBILITY and rw['visibility'] >= MIN_VISIBILITY:
                hand_gaps.append(abs(lw['x'] - rw['x']))

        if not hand_gaps:
            return {'symmetry': None}

        # Lower hand gap = better symmetry (hands together during pull)
        avg_gap = np.mean(hand_gaps)
        return {
            'symmetry': float(avg_gap),
            'min_gap': float(np.min(hand_gaps)),
        }

    # ── Issue detection ──

    def _detect_issues(self) -> List[BaseIssue]:
        issues = super()._detect_issues()

        # ── Glide ──
        glide = self.metrics.get('glide', {})
        if glide.get('glide_ratio') is not None:
            gr = glide['glide_ratio']
            if gr < 0.15:
                issues.append(BaseIssue('short_glide', 'critical',
                    f"Short glide ({gr:.2f} of cycle)",
                    'Extend forward after kick', gr))
            elif gr > 0.50:
                issues.append(BaseIssue('excessive_glide', 'minor',
                    f"Excessive glide ({gr:.2f} of cycle)",
                    'Start pull earlier', gr))

        # ── Kick timing ──
        kt = self.metrics.get('kick_timing', {})
        if kt.get('kick_lag') is not None:
            kl = kt['kick_lag']
            if kl < 0.3:
                issues.append(BaseIssue('early_kick', 'moderate',
                    f"Kick too early (lag {kl:.2f})",
                    'Kick after hands fully forward', kl))
            elif kl > 0.7:
                issues.append(BaseIssue('late_kick', 'moderate',
                    f"Kick too late (lag {kl:.2f})",
                    'Kick earlier', kl))

        # ── Heel clearance ──
        heel = self.metrics.get('heel_clearance', {})
        if heel.get('avg_heel_clearance') is not None:
            hc = heel['avg_heel_clearance']
            if hc > 0.30:
                issues.append(BaseIssue('feet_too_wide', 'moderate',
                    f"Feet too wide (heel clearance {hc:.2f})",
                    'Keep heels closer to hips', hc))

        # ── Arm symmetry ──
        sym = self.metrics.get('arm_symmetry', {})
        if sym.get('symmetry') is not None:
            gap = sym['symmetry']
            if gap > 0.10:
                issues.append(BaseIssue('uneven_hands', 'critical',
                    f"Uneven hands (avg gap {gap:.2f})",
                    'Keep hands together', gap))

        return issues
