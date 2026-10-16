"""
Exports per-frame datapoints as CSV for offline analysis.

Produces a CSV file with one row per analyzed frame containing:
  - frame number, timestamp
  - Raw landmark positions (x, y) and visibility for all 13 key landmarks
  - Calculated angles (elbow, knee) for both sides
  - Per-frame estimates of body rotation, arm entry, head position, undulation

This helps debug issues like camera-angle bias in synchronization detection
by letting you plot wrist trajectories and angle curves over time.
"""

import csv
import os
import numpy as np
from typing import List, Dict, Optional

# Landmark names used in the pipeline
LANDMARK_NAMES = [
    'nose',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]

MIN_VISIBILITY = 0.5


def _calculate_angle(p1: Dict, p2: Dict, p3: Dict) -> Optional[float]:
    """Calculate angle (degrees) at p2 between vectors p1-p2 and p3-p2."""
    a = np.array([p1['x'], p1['y']])
    b = np.array([p2['x'], p2['y']])
    c = np.array([p3['x'], p3['y']])
    v1 = a - b
    v2 = c - b
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-6:
        return None
    cos = np.dot(v1, v2) / norm
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def export_datapoints_csv(
    pose_data: List[Dict],
    output_path: str,
) -> str:
    """
    Export per-frame datapoints to a CSV file.

    Args:
        pose_data: List of frame dicts (from PoseDetector.process_video)
        output_path: Where to save the CSV file

    Returns:
        The output_path that was written to.
    """
    # Build CSV header
    header = ['frame_number', 'timestamp']

    # Landmark columns: {name}_x, {name}_y, {name}_visibility
    for lm in LANDMARK_NAMES:
        header.append(f'{lm}_x')
        header.append(f'{lm}_y')
        header.append(f'{lm}_visibility')

    # Per-frame calculated metrics
    header.extend([
        'left_elbow_angle',
        'right_elbow_angle',
        'left_knee_angle',
        'right_knee_angle',
        'body_rotation_estimate',
        'centerline_distance_left',
        'centerline_distance_right',
        'nose_y_normalized',
        'hip_y_avg',
        'wrist_span',
        'shoulder_span',
    ])

    rows = []

    for frame_data in pose_data:
        frame_number = frame_data.get('frame_number', 0)
        timestamp = frame_data.get('timestamp', 0.0)
        pose = frame_data.get('pose')

        if pose is None:
            # Emit a row with empty landmark values
            row = [frame_number, timestamp] + [None] * (len(LANDMARK_NAMES) * 3) + [None] * 11
            rows.append(row)
            continue

        landmarks = pose.get('landmarks', {})
        h, w = pose.get('frame_shape', (1, 1))

        # --- Landmark positions ---
        lm_values = []
        for lm in LANDMARK_NAMES:
            l = landmarks.get(lm)
            if l is not None:
                lm_values.extend([l['x'], l['y'], l['visibility']])
            else:
                lm_values.extend([None, None, None])

        # --- Calculated metrics ---
        # Elbow angles
        left_elbow = None
        if all(landmarks.get(s, {}).get('visibility', 0) >= MIN_VISIBILITY
               for s in ('left_shoulder', 'left_elbow', 'left_wrist')):
            left_elbow = _calculate_angle(
                landmarks['left_shoulder'],
                landmarks['left_elbow'],
                landmarks['left_wrist'],
            )

        right_elbow = None
        if all(landmarks.get(s, {}).get('visibility', 0) >= MIN_VISIBILITY
               for s in ('right_shoulder', 'right_elbow', 'right_wrist')):
            right_elbow = _calculate_angle(
                landmarks['right_shoulder'],
                landmarks['right_elbow'],
                landmarks['right_wrist'],
            )

        # Knee angles
        left_knee = None
        if all(landmarks.get(s, {}).get('visibility', 0) >= MIN_VISIBILITY
               for s in ('left_hip', 'left_knee', 'left_ankle')):
            left_knee = _calculate_angle(
                landmarks['left_hip'],
                landmarks['left_knee'],
                landmarks['left_ankle'],
            )

        right_knee = None
        if all(landmarks.get(s, {}).get('visibility', 0) >= MIN_VISIBILITY
               for s in ('right_hip', 'right_knee', 'right_ankle')):
            right_knee = _calculate_angle(
                landmarks['right_hip'],
                landmarks['right_knee'],
                landmarks['right_ankle'],
            )

        # Body rotation estimate (shoulder/hip width heuristic)
        body_rotation = None
        if all(landmarks.get(s, {}).get('visibility', 0) >= MIN_VISIBILITY
               for s in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip')):
            sw = abs(landmarks['left_shoulder']['x'] - landmarks['right_shoulder']['x'])
            hw = abs(landmarks['left_hip']['x'] - landmarks['right_hip']['x'])
            avg = (sw + hw) / 2
            body_rotation = max(0, min(90, 90 - (avg / w * 180)))

        # Centerline distance
        centerline_left = None
        centerline_right = None
        if all(landmarks.get(s, {}).get('visibility', 0) >= MIN_VISIBILITY
               for s in ('left_shoulder', 'right_shoulder')):
            cx = (landmarks['left_shoulder']['x'] + landmarks['right_shoulder']['x']) / 2
            if landmarks.get('left_wrist', {}).get('visibility', 0) >= MIN_VISIBILITY:
                centerline_left = abs(landmarks['left_wrist']['x'] - cx) / w
            if landmarks.get('right_wrist', {}).get('visibility', 0) >= MIN_VISIBILITY:
                centerline_right = abs(landmarks['right_wrist']['x'] - cx) / w

        # Nose y normalized
        nose_y_norm = None
        if landmarks.get('nose', {}).get('visibility', 0) >= MIN_VISIBILITY:
            nose_y_norm = landmarks['nose']['y'] / h

        # Hip y average
        hip_y_avg = None
        if (landmarks.get('left_hip', {}).get('visibility', 0) >= MIN_VISIBILITY and
            landmarks.get('right_hip', {}).get('visibility', 0) >= MIN_VISIBILITY):
            hip_y_avg = (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2

        # Wrist span / shoulder span
        wrist_span = None
        shoulder_span = None
        if (landmarks.get('left_wrist', {}).get('visibility', 0) >= MIN_VISIBILITY and
            landmarks.get('right_wrist', {}).get('visibility', 0) >= MIN_VISIBILITY):
            wrist_span = abs(landmarks['left_wrist']['x'] - landmarks['right_wrist']['x'])
        if (landmarks.get('left_shoulder', {}).get('visibility', 0) >= MIN_VISIBILITY and
            landmarks.get('right_shoulder', {}).get('visibility', 0) >= MIN_VISIBILITY):
            shoulder_span = abs(landmarks['left_shoulder']['x'] - landmarks['right_shoulder']['x'])

        metrics = [
            left_elbow, right_elbow,
            left_knee, right_knee,
            body_rotation,
            centerline_left, centerline_right,
            nose_y_norm,
            hip_y_avg,
            wrist_span, shoulder_span,
        ]

        row = [frame_number, timestamp] + lm_values + metrics
        rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Datapoints exported: {output_path} ({len(rows)} rows)")
    return output_path
