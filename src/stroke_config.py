"""Stroke configuration dataclass — one config per stroke type.

All strokes share the same base pipeline (pose → clean → canonicalize → segment → metrics → decide).
Only the cycle signal, phase expectations, kick pattern, and metric list differ.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class StrokeConfig:
    """Configuration for a single stroke type."""
    name: str
    cycle_signal: str                         # "near_wrist_x", "wrist_shoulder_dist", "knee_angle", ...
    expected_arm_phase_range: Tuple[float, float]  # (min, max) acceptable L/R phase gap
    kicks_per_cycle_range: Tuple[int, int]         # (min, max) kicks per arm cycle
    metrics: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)


# ── Shared metric names ──
SHARED_METRICS = [
    'elbow_min_pull',      # min elbow angle during pull (2D, near arm)
    'arm_phase',            # L/R wrist-x cross-correlation phase gap
    'hip_line',             # shoulder→hip→knee angle vs horizontal
    'kick',                 # knee angle range + ankle amplitude
    'recovery_height',      # wrist clearance above shoulder (torso-normalised)
    'breath_lift',          # nose-y relative to shoulder line
]


CONFIGS: Dict[str, StrokeConfig] = {
    'freestyle': StrokeConfig(
        name='freestyle',
        cycle_signal='near_wrist_x',
        expected_arm_phase_range=(0.40, 0.60),   # windmill: ~0.5
        kicks_per_cycle_range=(2, 6),
        metrics=[
            'elbow_min_pull',
            'arm_phase',
            'breath_lift',
            'kick',
            'hip_line',
        ],
        thresholds={
            'elbow_pull_min_ok': 90,      # ° — minimum during pull
            'elbow_pull_max_ok': 120,     # ° — maximum during pull
            'elbow_dropped': 150,         # ° — flag if pull min > this
            'recovery_elbow_max': 160,    # ° — flag if recovery elbow avg > this
            'breath_lift_max': 0.08,      # torso-normalised head lift
            'knee_bend_min': 120,         # ° — below this = too bent
            'knee_bend_max': 170,         # ° — above this = too straight
            'hip_line_drop_max': 15,      # ° deviation from straight
        },
    ),

    'backstroke': StrokeConfig(
        name='backstroke',
        cycle_signal='near_wrist_x',
        expected_arm_phase_range=(0.40, 0.60),   # ~0.5
        kicks_per_cycle_range=(4, 8),             # continuous flutter
        metrics=[
            'elbow_min_pull',
            'arm_phase',
            'recovery_height',
            'kick',
            'hip_line',
        ],
        thresholds={
            'elbow_pull_min_ok': 90,
            'elbow_pull_max_ok': 120,
            'elbow_dropped': 150,
            'recovery_elbow_max': 160,
            'recovery_height_min': 0.03,
            'recovery_height_max': 0.20,
            'knee_bend_min': 120,
            'knee_bend_max': 170,
            'hip_line_drop_max': 15,
        },
    ),

    'butterfly': StrokeConfig(
        name='butterfly',
        cycle_signal='near_wrist_x',
        expected_arm_phase_range=(0.0, 0.10),   # in phase (~0)
        kicks_per_cycle_range=(2, 2),            # exactly 2 per cycle
        metrics=[
            'undulation',
            'kick_timing',
            'breath_lift',
            'arm_phase',
            'recovery_height',
        ],
        thresholds={
            'undulation_amp_min': 0.04,
            'undulation_amp_max': 0.12,
            'breath_lift_max': 0.08,
            'recovery_height_min': 0.03,
            'recovery_height_max': 0.20,
            'recovery_elbow_max': 160,
        },
    ),

    'breaststroke': StrokeConfig(
        name='breaststroke',
        cycle_signal='wrist_shoulder_dist',      # arms extend → pull back
        expected_arm_phase_range=(0.0, 0.10),    # simultaneous (~0)
        kicks_per_cycle_range=(1, 1),            # 1 kick per cycle
        metrics=[
            'pull_depth',
            'arm_kick_lag',
            'glide',
            'hip_line',
        ],
        thresholds={
            'pull_depth_max': 0.15,          # wrist behind shoulder (torso-normalised)
            'arm_kick_lag_min': 0.1,         # fraction of cycle
            'arm_kick_lag_max': 0.3,
            'glide_duration_min': 0.3,       # seconds
            'hip_line_drop_max': 15,
        },
    ),
}
