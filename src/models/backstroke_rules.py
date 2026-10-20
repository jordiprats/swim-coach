"""Backstroke technique rules and thresholds.

All measurements assume side-view camera (swimmer horizontal in frame).
Metrics that don't work from the side (e.g. arm entry width measured laterally)
are handled in the analyzer via camera-perspective detection.
"""

import math

# ============================================================
# Camera perspective detection
# ============================================================
# Side-view: shoulders overlap → shoulder_x_separation / torso_length < ~0.5
# Overhead: shoulders spread → ratio > ~0.8
SIDE_VIEW_SHOULDER_RATIO_THRESHOLD = 0.6   # ratio below this → side view

# ============================================================
# Near / Far arm labeling
# ============================================================
# In side view, the arm closer to the camera has higher visibility
# and larger wrist_x variance.  The far arm has lower visibility and
# noisier landmarks.
NEAR_ARM_VISIBILITY_ADVANTAGE = 0.05       # visibility delta for near vs far

# ============================================================
# Arm Entry  (side-view compatible)
# ============================================================
# From the side, entry happens when the wrist is furthest forward
# (maximum x coordinate) — not highest y.
# Entry width measured as wrist span relative to frame width (overhead only).
ARM_ENTRY_WIDTH_MIN = 0.15                # 15% of frame width — too narrow
ARM_ENTRY_WIDTH_MAX = 0.30                # 30% — too wide

# ============================================================
# Elbow / Pull
# ============================================================
# During catch the elbow is slightly bent (~120-150°).
# During pull the elbow bends more (~90-120°) as hand sweeps under body.
# During recovery the arm is nearly straight (~155-180°).
ELBOW_ANGLE_CATCH_MIN = 120
ELBOW_ANGLE_CATCH_MAX = 150
ELBOW_ANGLE_PULL_MIN = 90
ELBOW_ANGLE_PULL_MAX = 120
ELBOW_ANGLE_RECOVERY_MIN = 155
ELBOW_ANGLE_RECOVERY_MAX = 180
ELBOW_ANGLE_DROPPED = 160                # Above this during catch → dropped elbow
                                        # NOTE: averaged across all phases this will
                                        # always flag.  Use per-cycle pull-phase min.

# ============================================================
# Body Rotation
# ============================================================
# Backstroke rotation is shoulder-driven, ~45° optimal.
# From side view, shoulder x-separation / torso_length is used.
BODY_ROTATION_OPTIMAL_MIN = 30
BODY_ROTATION_OPTIMAL_MAX = 55
BODY_ROTATION_TOO_FLAT = 20
BODY_ROTATION_EXCESSIVE = 65

# ============================================================
# Head / Breathing
# ============================================================
# Face is above water, breathing is continuous.
# Head should be stable, chin tucked, minimal vertical movement.
# All measurements normalised to torso length, not frame height.
HEAD_VERTICAL_MOVEMENT_MAX = 0.06         # 6% of torso length
HEAD_LIFT_EXCESSIVE = 0.10               # Chin too high → body sinks

# ============================================================
# Stroke Rate
# ============================================================
# Measured via FFT on a single arm's wrist-x signal.
# Filter windows in seconds (not frames) for fps invariance.
STROKE_RATE_OPTIMAL_MIN = 40
STROKE_RATE_OPTIMAL_MAX = 60
STROKE_RATE_TOO_SLOW = 35
STROKE_RATE_TOO_FAST = 70
STROKE_RATE_FFT_WINDOW_S = 2.0           # FFT window in seconds
STROKE_RATE_MIN_CYCLES = 4               # minimum cycles for FFT reliability

# ============================================================
# Kick (Flutter Kick)
# ============================================================
# Alternating legs, knee bend ~140-160°.
#   - Below 120° = too much bend (excessive drag from knee flexion)
#   - Above 170° = too straight (no propulsion)
# Range filter: 100-180° to discard glitches.
# Underwater knees have low pose confidence — gate visibility >= 0.5.
KNEE_ANGLE_OPTIMAL_MIN = 140
KNEE_ANGLE_OPTIMAL_MAX = 160
KNEE_ANGLE_EXCESSIVE_BEND = 120          # avg_knee < this → too bent
KNEE_ANGLE_TOO_STRAIGHT = 170            # avg_knee > this → too straight
KNEE_VISIBILITY_GATE = 0.5               # Stricter gate for underwater knees

# ============================================================
# Arm Synchronisation
# ============================================================
# Arms should be ~180° out of phase (one recovering while other pulls).
# Measured via circular phase difference (cross-correlation of wrist-x signals).
# Ideal gap = 0.5 of a cycle (180° out of phase).
SYNC_PHASE_GAP_IDEAL = 0.5               # Fraction of cycle
SYNC_PHASE_GAP_MIN = 0.35
SYNC_PHASE_GAP_MAX = 0.65
SYNC_MIN_VISIBLE_FAR_ARM = 0.3           # Far arm must be visible enough for timing

# ============================================================
# Recovery Height
# ============================================================
# Arm should clear the water — wrist above shoulder during recovery.
# Normalised to torso length (not frame height).
# Too high wastes energy; too low risks hand hitting water.
RECOVERY_HEIGHT_MIN = 0.03               # Minimum clearance (torso-length units)
RECOVERY_HEIGHT_MAX = 0.20               # Maximum ideal clearance
RECOVERY_HEIGHT_EXCESSIVE = 0.20         # Consistent threshold (not 0.25)

# ============================================================
# Hand Entry Events (side-view compatible)
# ============================================================
# From the side, define events by wrist-x trajectory:
#   Recovery apex   — wrist at highest y (vertical)
#   Entry           — wrist furthest forward (max x)
#   Catch           — elbow starts bending after entry
#   Exit            — wrist passes hip
ENTRY_MIN_GAP_S = 0.4                    # Minimum gap between entries (seconds)
ENTRY_WRIST_X_WINDOW = 0.15              # Fraction of torso length for x peak detection

# ============================================================
# Hip Drop / Body Line
# ============================================================
# From side view, use shoulder→hip→knee angle against horizontal
# (corrected for camera tilt), not hip-y wobble.
#   - Hip drop = deviation from horizontal body line
#   - Measured as angle of shoulder-hip-knee line relative to horizontal
BODY_LINE_ANGLE_DROP_MAX = 15.0          # Degrees below horizontal → hips dropped
BODY_LINE_CHIN_TO_HIP_MIN = 0.10
BODY_LINE_CHIN_TO_HIP_MAX = 0.25
HIP_DROP_MAX = 0.03                      # Legacy — kept for backward compat

# ============================================================
# Body Normalization
# ============================================================
# All metrics normalised to torso length (shoulder_mid → hip_mid distance)
# in all modes, not just precision mode.
TORSO_LENGTH_SMOOTHING = 0.1             # EMA alpha for torso-length tracking

# ============================================================
# Pose Quality
# ============================================================
MIN_VISIBILITY = 0.3
BONE_LENGTH_TOLERANCE = 0.25            # Reject frame if bone length deviates >25%
HAMPEL_WINDOW = 5                       # Hampel filter half-window
HAMPEL_NSIGMA = 3                       # Outlier threshold for Hampel filter

# ============================================================
# Decision Logic
# ============================================================
# Persistence: flag an issue only if it appears in >= 60% of cycles.
# Hysteresis: thresholds have different entry/exit values to reduce flicker.
# Confidence gating: require minimum valid-frame fraction before issuing verdict.
PERSISTENCE_THRESHOLD = 0.6             # Fraction of cycles for issue to fire
MIN_VALID_FRAME_RATIO = 0.3            # Minimum valid frames for any metric
CONFIDENCE_GATE_ENABLED = True

# ============================================================
# Severity Levels
# ============================================================
SEVERITY_CRITICAL = "critical"
SEVERITY_MODERATE = "moderate"
SEVERITY_MINOR = "minor"


class BackstrokeIssue:
    """Represents a detected backstroke technique issue."""

    def __init__(self, issue_type: str, severity: str, description: str, tip: str, metric_value=None):
        self.issue_type = issue_type
        self.severity = severity
        self.description = description
        self.tip = tip
        self.metric_value = metric_value

    def __repr__(self):
        return f"BackstrokeIssue({self.issue_type}, {self.severity})"


ISSUE_TYPES = {
    'dropped_elbow': {
        'name': 'Dropped Elbow During Catch',
        'tip': 'Keep your elbow slightly bent (~120-150°) at catch. A straight arm loses leverage and strains your shoulder. Imagine reaching up to grab a shelf — your elbow should stay above your hand.',
        'severity': SEVERITY_CRITICAL,
    },
    'arms_too_narrow': {
        'name': 'Arms Entering Too Narrow',
        'tip': 'Enter your hands wider — aim for 11 o\'clock (left) and 1 o\'clock (right) relative to your head. A narrow entry reduces your pull length.',
        'severity': SEVERITY_MODERATE,
    },
    'arms_too_wide': {
        'name': 'Arms Entering Too Wide',
        'tip': 'Bring your hands slightly closer together on entry. Excessively wide entry makes it harder to catch water effectively.',
        'severity': SEVERITY_MINOR,
    },
    'flat_body': {
        'name': 'Limited Body Rotation',
        'tip': 'Increase your shoulder roll (~30-55°). Rotate from your hips, not just your shoulders. More rotation helps your arm clear the water cleanly.',
        'severity': SEVERITY_MODERATE,
    },
    'over_rotation': {
        'name': 'Over-rotation',
        'tip': 'Reduce your body roll — too much rotation makes you unstable and can lead to timing issues. Focus on a smooth, controlled roll.',
        'severity': SEVERITY_MODERATE,
    },
    'head_lifting_excessive': {
        'name': 'Chin Too High',
        'tip': 'Tuck your chin toward your chest. Your eyes should look straight up or slightly backward. A high chin drops your hips and creates drag.',
        'severity': SEVERITY_MODERATE,
    },
    'poor_arm_timing': {
        'name': 'Arm Timing Off',
        'tip': 'Arms should be opposite — one recovering while the other pulls. If both arms are in the same phase, you\'re losing propulsion. Practice with a tempo trainer to lock in the rhythm.',
        'severity': SEVERITY_CRITICAL,
    },
    'recovery_too_high': {
        'name': 'Arm Recovery Too High',
        'tip': 'Keep your arm low and sweeping just above the water surface. A high recovery wastes energy and can cause shoulder impingement. Think of skimming the water with your hand.',
        'severity': SEVERITY_MODERATE,
    },
    'recovery_too_low': {
        'name': 'Arm Recovery Too Low — Hand Hits Water',
        'tip': 'Raise your recovery slightly so your hand clears the water. A low recovery creates drag and disrupts your rhythm. Imagine tracing a rainbow arc with your fingertips.',
        'severity': SEVERITY_MODERATE,
    },
    'excessive_knee_bend': {
        'name': 'Excessive Knee Bend in Flutter Kick',
        'tip': 'Kick from your hips, not your knees. Keep your legs relatively straight with a gentle bend (~140-160°). Too much knee bend creates drag.',
        'severity': SEVERITY_MODERATE,
    },
    'knee_too_straight': {
        'name': 'Knees Too Straight — Not Enough Kick',
        'tip': 'Allow a natural bend in your knees (~140-160°) to create propulsion. A straight leg doesn\'t generate enough kick power.',
        'severity': SEVERITY_MODERATE,
    },
    'slow_stroke_rate': {
        'name': 'Stroke Rate Too Slow',
        'tip': 'Increase your tempo slightly. For backstroke aim for 40-60 SPM depending on distance. Too slow = momentum loss.',
        'severity': SEVERITY_MINOR,
    },
    'fast_stroke_rate': {
        'name': 'Stroke Rate Too Fast',
        'tip': 'Slow down and focus on longer, stronger pulls. Rushing usually means a poor catch and wasted energy.',
        'severity': SEVERITY_MINOR,
    },
    'hips_drop': {
        'name': 'Hips Drop — Poor Body Line',
        'tip': 'Keep your hips high and close to the surface. Engage your core to maintain a straight body line from head to hips. A dropped hip creates a seesaw effect.',
        'severity': SEVERITY_CRITICAL,
    },
}


def get_severity_emoji(severity: str) -> str:
    """Get emoji indicator for severity level."""
    if severity == SEVERITY_CRITICAL:
        return "🔴"
    elif severity == SEVERITY_MODERATE:
        return "🟡"
    else:
        return "🔵"


def get_severity_label(severity: str) -> str:
    """Get text label for severity level."""
    if severity == SEVERITY_CRITICAL:
        return "Critical Issues"
    elif severity == SEVERITY_MODERATE:
        return "Areas for Improvement"
    else:
        return "Minor Suggestions"
