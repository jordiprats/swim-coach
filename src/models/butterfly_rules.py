"""Butterfly stroke technique rules and thresholds."""

# --- Elbow / Arm Pull ---
# During the pull phase the arms stay relatively extended compared with
# freestyle's high-elbow catch.  A "dropped elbow" in butterfly means the
# elbows sink too low during the pull, losing leverage.
ARM_ENTRY_WIDTH_MIN = 0.20          # 20% of frame width – arms enter wide
ARM_ENTRY_WIDTH_MAX = 0.35          # 35% – too narrow reduces pull length
ELBOW_ANGLE_PULL_MIN = 120          # Minimal acceptable during pull phase
ELBOW_ANGLE_PULL_MAX = 160          # Healthy range
ELBOW_ANGLE_DROPPED = 160           # Above this → elbows too low

# --- Body Undulation (replaces body rotation) ---
# Butterfly uses vertical undulation rather than side-to-side rotation.
# Measured as vertical displacement of the hips over a stroke cycle,
# normalised by frame height.
UNDULATION_MIN = 0.04               # 4% of frame height – minimum healthy undulation
UNDULATION_MAX = 0.12               # 12% – too much undulation wastes energy
UNDULATION_TOO_FLAT = 0.03          # Below this → not undulating enough
UNDULATION_EXCESSIVE = 0.14         # Above this → over-undulating

# --- Head / Breathing ---
# Butterfly breathing is a forward lift, not a side rotation.
HEAD_LIFT_THRESHOLD = 0.08          # 8% of frame height – normal breathing lift
HEAD_LIFT_EXCESSIVE = 0.18          # 18% – lifting too high
BREATHING_EVERY_N_STROKES_MIN = 1   # Minimum strokes between breaths
BREATHING_EVERY_N_STROKES_MAX = 2   # Maximum recommended (sprinters breath every 1–2)

# --- Stroke Rate ---
STROKE_RATE_OPTIMAL_MIN = 30        # Distance butterfly (long course)
STROKE_RATE_OPTIMAL_MAX = 55        # Sprint butterfly
STROKE_RATE_TOO_SLOW = 25           # Below this → gliding too long
STROKE_RATE_TOO_FAST = 65           # Above this → rushing, likely poor catch

# --- Dolphin Kick ---
# Measured as knee angle during the downbeat of the dolphin kick.
# Dolphin kick involves a whole-body wave, so the knee angle is naturally
# more bent than in freestyle.
KNEE_ANGLE_OPTIMAL_MIN = 120
KNEE_ANGLE_OPTIMAL_MAX = 150
KNEE_ANGLE_TOO_STRAIGHT = 160       # Above this → not bending knees enough
KNEE_ANGLE_EXCESSIVE_BEND = 110     # Below this → too much knee, not enough hip drive

# --- Arm Synchronisation ---
# In butterfly both arms should enter and pull simultaneously.
# Measured as the absolute difference in wrist-x peak timestamps (in seconds).
SYNC_MAX_DELTA = 0.15               # Max allowed delay between left & right peaks

# --- Arm Recovery ---
# Recovery height measured as (elbow_y - shoulder_y) / frame_height
# In butterfly the arms sweep low and wide; a high recovery wastes energy.
RECOVERY_HEIGHT_MIN = 0.05          # 5% of frame height – minimum clearance
RECOVERY_HEIGHT_MAX = 0.20          # 20% – maximum before energy waste
RECOVERY_HEIGHT_EXCESSIVE = 0.25    # Above this → arms too high, taxing recovery

# --- Shoulder-Hip Phase Relationship ---
# Proper undulation means shoulder and hip move out of phase.
# Measured as cross-correlation lag between shoulder y and hip y.
PHASE_LAG_IDEAL = 0.25              # Ideal lag as fraction of stroke cycle
PHASE_LAG_MIN = 0.15                # Minimum acceptable lag
PHASE_LAG_MAX = 0.35                # Maximum acceptable lag
PHASE_LAG_TOO_SMALL = 0.10          # Below → shoulders and hips move together (no wave)

# --- Stroke Phase Timing (Coordination Index) ---
# The second kick should align with the end of the pull (wrist passes hip).
# Measured as frame gap between wrist-passes-hip and ankle-max-extension.
COORDINATION_MAX_GAP = 2            # Max frames between pull-end and kick-peak

# --- Late Breathing ---
# Head should descend before hands enter.  Measured as frame gap between
# max nose-y (breath peak) and hand-entry frame.
LATE_BREATHING_MAX_GAP = 3          # Max frames between breath peak and hand entry

# --- Hip Drop During Breath ---
# Hip y at breathing frame relative to average hip y.
HIP_DROP_MAX = 0.03                 # 3% of frame height – max hip drop during breath
HIP_DROP_EXCESSIVE = 0.06           # Above → hips sink too much when breathing

# --- Visibility ---
MIN_VISIBILITY = 0.3

# --- Severity Levels ---
SEVERITY_CRITICAL = "critical"
SEVERITY_MODERATE = "moderate"
SEVERITY_MINOR = "minor"


class ButterflyIssue:
    """Represents a detected butterfly technique issue."""

    def __init__(self, issue_type: str, severity: str, description: str, tip: str, metric_value=None):
        self.issue_type = issue_type
        self.severity = severity
        self.description = description
        self.tip = tip
        self.metric_value = metric_value

    def __repr__(self):
        return f"ButterflyIssue({self.issue_type}, {self.severity})"


ISSUE_TYPES = {
    'dropped_elbow': {
        'name': 'Dropped Elbows',
        'tip': 'Keep your elbows higher during the pull. Imagine pressing your palms against a wall in front of you — your elbows should stay above your hands throughout the pull.',
        'severity': SEVERITY_CRITICAL,
    },
    'arms_too_narrow': {
        'name': 'Arms Entering Too Narrow',
        'tip': 'Enter your hands wider — roughly at 11 o\'clock and 1 o\'clock relative to your head. A narrow entry shortens your pull and reduces propulsion.',
        'severity': SEVERITY_MODERATE,
    },
    'arms_too_wide': {
        'name': 'Arms Entering Too Wide',
        'tip': 'Bring your hands slightly closer together on entry. Excessively wide entry reduces your pull length through the stroke.',
        'severity': SEVERITY_MINOR,
    },
    'flat_undulation': {
        'name': 'Flat Body Position — Not Undulating',
        'tip': 'Initiate the dolphin wave from your chest, not your knees. Press your chest down, let your hips rise behind you. Think of a sine wave from your shoulders to your toes.',
        'severity': SEVERITY_CRITICAL,
    },
    'excessive_undulation': {
        'name': 'Excessive Undulation',
        'tip': 'Reduce the amplitude of your dolphin wave. Too much up-and-down movement creates drag. Focus on a smoother, more efficient wave.',
        'severity': SEVERITY_MODERATE,
    },
    'head_lifting_excessive': {
        'name': 'Lifting Head Too High to Breathe',
        'tip': 'Keep your chin low — just break the water surface to breathe. Lifting your whole head breaks your body line and slows you down.',
        'severity': SEVERITY_MODERATE,
    },
    'arms_not_synchronised': {
        'name': 'Arms Not Synchronised',
        'tip': 'Both arms must enter and pull together. Practice on land with a towel — swing both arms forward simultaneously. In water, use a slower tempo until you nail the timing.',
        'severity': SEVERITY_CRITICAL,
    },
    'slow_stroke_rate': {
        'name': 'Stroke Rate Too Slow',
        'tip': 'Increase your tempo slightly. For butterfly you want 30–55 SPM depending on distance. Longer glide = less momentum.',
        'severity': SEVERITY_MINOR,
    },
    'fast_stroke_rate': {
        'name': 'Stroke Rate Too Fast',
        'tip': 'Slow down and focus on longer, stronger pulls. Rushing your strokes usually means a poor catch and wasted energy.',
        'severity': SEVERITY_MINOR,
    },
    'excessive_knee_bend': {
        'name': 'Excessive Knee Bend in Dolphin Kick',
        'tip': 'Drive the dolphin kick from your hips, not your knees. Keep your legs relatively straight and let the wave travel through your whole body.',
        'severity': SEVERITY_MODERATE,
    },
    'knee_too_straight': {
        'name': 'Knees Too Straight — Not Enough Kick',
        'tip': 'Allow a natural bend in your knees during the downbeat (~120–150°). A straight leg doesn\'t create enough propulsion.',
        'severity': SEVERITY_MODERATE,
    },
    'recovery_too_high': {
        'name': 'Arm Recovery Too High',
        'tip': 'Keep your arms low and sweeping just above the water surface. High arm recovery wastes energy and causes shoulder strain. Imagine skimming the water with your fingertips.',
        'severity': SEVERITY_MODERATE,
    },
    'shoulder_hip_in_phase': {
        'name': 'Shoulders and Hips Moving Together (No Wave)',
        'tip': 'Initiate the dolphin wave from your chest, letting your hips rise as your chest presses down. Your shoulders and hips should move out of phase — when shoulders go down, hips go up.',
        'severity': SEVERITY_CRITICAL,
    },
    'poor_coordination': {
        'name': 'Kick and Pull Not Aligned',
        'tip': 'Time your second kick to finish exactly when your hands finish the pull past your hips. The down-kick should accelerate as your palms push back. Practice with a 3-3-3 drill to rebuild timing.',
        'severity': SEVERITY_CRITICAL,
    },
    'late_breathing': {
        'name': 'Late Breathing — Head Still High When Hands Enter',
        'tip': 'Snap your head back down into the water BEFORE your hands enter. If your head is still up at hand entry, your hips will drop and create drag. Practice head-down dolphin kicks to retrain the timing.',
        'severity': SEVERITY_MODERATE,
    },
    'hips_drop_during_breath': {
        'name': 'Hips Drop Too Low During Breath',
        'tip': 'Keep your hips high and close to the surface when you breathe. If your hips sink when you lift your head, you\'re creating a seesaw effect that kills momentum. Focus on pressing your chest down to keep hips up.',
        'severity': SEVERITY_MODERATE,
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
