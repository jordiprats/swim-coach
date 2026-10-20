"""Freestyle analyzer — inherits shared pipeline from BaseAnalyzer.

Stroke-specific:
  - Cycle signal: near_wrist_x
  - Arm phase: ~0.5 (windmill)
  - Metrics: elbow_min_pull, arm_phase, breath_lift, kick, hip_line
  - Entry: furthest-forward wrist at water level
  - Pull: wrist passing below shoulder; exit: wrist passing hip
"""

from typing import List, Dict, Tuple
import numpy as np

from src.base_analyzer import BaseAnalyzer, BaseIssue, MIN_VISIBILITY, PERSISTENCE_THRESHOLD
from src.stroke_config import CONFIGS


class FreestyleAnalyzer(BaseAnalyzer):
    """Freestyle-specific analyzer."""

    def __init__(self):
        super().__init__('freestyle')

    # ── Cycle detection ──

    def _detect_cycles(self, frames: List[Dict]) -> List[Tuple[int, int]]:
        """Detect cycles from near-wrist-x peaks.

        Entry = wrist furthest forward (max x relative to shoulder).
        Pull = wrist below shoulder.
        Exit = wrist behind hip.
        """
        if len(frames) < 10:
            return []

        near_arm = self._arm_labels.get('near_arm', 'left')

        # Compute wrist-x relative to shoulder midpoint
        wrist_rel = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            wr = landmarks[f'{near_arm}_wrist']
            sh = landmarks.get('left_shoulder')
            rh = landmarks.get('right_shoulder')
            if (wr['visibility'] >= MIN_VISIBILITY and
                sh and rh and sh['visibility'] >= MIN_VISIBILITY and rh['visibility'] >= MIN_VISIBILITY):
                shoulder_x = (sh['x'] + rh['x']) / 2
                wrist_rel.append(wr['x'] - shoulder_x)
            else:
                wrist_rel.append(None)

        # Smooth
        window = max(3, len(wrist_rel) // 30)
        smoothed = self._moving_average([v if v is not None else 0 for v in wrist_rel], window)

        # Peaks (max = entry — hand furthest forward)
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
        """Freestyle-specific metrics."""
        metrics = {}

        # ── Body rotation (shared metric, but freestyle has its own thresholds) ──
        # Rotation is already in shared metrics; just add freestyle-specific notes
        metrics['rotation'] = self._compute_freestyle_rotation(frames)

        return metrics

    @staticmethod
    def _compute_freestyle_rotation(frames: List[Dict]) -> Dict:
        """Body rotation for freestyle (shared formula, freestyle thresholds)."""
        rotations = []
        for f in frames:
            landmarks = f['pose']['landmarks']
            ls = landmarks.get('left_shoulder')
            rs = landmarks.get('right_shoulder')
            lh = landmarks.get('left_hip')
            rh = landmarks.get('right_hip')
            if (ls and rs and lh and rh and
                ls['visibility'] >= MIN_VISIBILITY and rs['visibility'] >= MIN_VISIBILITY and
                lh['visibility'] >= MIN_VISIBILITY and rh['visibility'] >= MIN_VISIBILITY):
                sw = abs(ls['x'] - rs['x'])
                hw = abs(lh['x'] - rh['x'])
                avg = (sw + hw) / 2
                frame_w = f['pose']['frame_shape'][1]
                rot = 90 - (avg / frame_w * 180)
                rotations.append(max(0, min(90, rot)))

        if not rotations:
            return {'avg_rotation': None}

        return {
            'avg_rotation': np.mean(rotations),
            'min_rotation': np.min(rotations),
            'max_rotation': np.max(rotations),
        }

    # ── Issue detection ──

    def _detect_issues(self) -> List[BaseIssue]:
        issues = super()._detect_issues()

        # ── Body rotation ──
        rotation = self.metrics.get('rotation', {})
        if rotation.get('avg_rotation') is not None:
            avg_rot = rotation['avg_rotation']
            if avg_rot < 30:
                issues.append(BaseIssue('flat_body', 'moderate',
                    f"Limited rotation ({avg_rot:.0f}°)",
                    'Increase shoulder roll', avg_rot))
            elif avg_rot > 60:
                issues.append(BaseIssue('over_rotation', 'moderate',
                    f"Over-rotation ({avg_rot:.0f}°)",
                    'Reduce body roll', avg_rot))

        # ── Breathing classification ──
        breath = self.metrics.get('breath_lift', {})
        if breath.get('lift') is not None:
            # For freestyle, head lift > 0.08 is excessive
            # Head rotation itself is hard to see from side view — report low confidence
            lift = breath['lift']
            if abs(lift) > 0.08:
                issues.append(BaseIssue('breathing_excessive', 'moderate',
                    f"Excessive head lift ({lift:.3f}x torso) — [low confidence from side view]",
                    'Rotate to breathe, don\'t lift', lift))

        return issues
