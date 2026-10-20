"""Backstroke analyzer — inherits shared pipeline from BaseAnalyzer.

Stroke-specific:
  - Cycle signal: near_wrist_x
  - Arm phase: ~0.5 (alternating)
  - Metrics: elbow_min_pull, arm_phase, recovery_height, kick, hip_line
"""

from typing import List, Dict, Tuple

from src.base_analyzer import BaseAnalyzer, BaseIssue, MIN_VISIBILITY, PERSISTENCE_THRESHOLD
from src.stroke_config import CONFIGS
from src.models.backstroke_rules import (
    BackstrokeIssue, ISSUE_TYPES,
    get_severity_emoji as bk_get_severity_emoji,
    get_severity_label as bk_get_severity_label,
)


class BackstrokeAnalyzer(BaseAnalyzer):
    """Backstroke-specific analyzer."""

    def __init__(self):
        super().__init__('backstroke')

    # ── Cycle detection ──

    def _detect_cycles(self, frames: List[Dict]) -> List[Tuple[int, int]]:
        """Detect cycles from near-wrist-x peaks."""
        if len(frames) < 10:
            return []

        near_arm = self._arm_labels.get('near_arm', 'left')
        wrist_x = []
        for f in frames:
            lm = f['pose']['landmarks'][f'{near_arm}_wrist']
            wrist_x.append(lm['x'] if lm['visibility'] >= MIN_VISIBILITY else None)

        # Smooth
        window = max(3, len(wrist_x) // 30)
        smoothed = self._moving_average([v if v is not None else 0 for v in wrist_x], window)

        # Peaks (max x = hand furthest forward = entry)
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
        """Backstroke-specific metrics beyond shared ones."""
        return {}   # All backstroke metrics are covered by shared metrics

    # ── Issue detection ──

    def _detect_issues(self) -> List[BaseIssue]:
        """Backstroke issues using shared detection plus stroke-specific overrides."""
        issues = super()._detect_issues()

        # ── Entry width (overhead only) ──
        entry = self.metrics.get('entry', {})
        if entry.get('avg_entry_width') is not None:
            avg_w = entry['avg_entry_width']
            if avg_w < 0.15:
                issues.append(BaseIssue('arms_too_narrow', 'moderate',
                    f"Arms too narrow (wrist span {avg_w:.2f}x frame width)",
                    'Enter wider — 11 o\'clock/1 o\'clock', avg_w))
            elif avg_w > 0.30:
                issues.append(BaseIssue('arms_too_wide', 'minor',
                    f"Arms too wide ({avg_w:.2f}x)",
                    'Bring hands slightly closer', avg_w))

        # ── Body rotation ──
        rotation = self.metrics.get('rotation', {})
        if rotation.get('avg_rotation') is not None:
            avg_rot = rotation['avg_rotation']
            if avg_rot < 20:
                issues.append(BaseIssue('flat_body', 'moderate',
                    f"Limited rotation ({avg_rot:.0f}°)",
                    'Increase shoulder roll', avg_rot))
            elif avg_rot > 65:
                issues.append(BaseIssue('over_rotation', 'moderate',
                    f"Over-rotation ({avg_rot:.0f}°)",
                    'Reduce body roll', avg_rot))

        return issues
