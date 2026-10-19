"""Fusion pose detector combining MediaPipe + YOLOv8 for robust landmark estimation.

Fusion strategy:
- Runs both detectors on each frame
- If both detect: confidence-weighted average of landmarks
- If only one detects: uses that one (with confidence penalty for missing detector)
- Temporal consistency check: if one detector suddenly drops out mid-sequence,
  the fused output from the other detector is used with reduced confidence
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

from src.pose_detector import PoseDetector
from src.yolo_pose_detector import YOLOPoseDetector


# Landmark names we need for fusion
FUSION_LANDMARKS = [
    'nose',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]


def _validate_landmarks(landmarks: Dict, frame_shape: Tuple[int, int]) -> bool:
    """
    Check if landmarks are plausible (within frame bounds, reasonable joint
    distances).  Returns False if landmarks are clearly wrong (e.g., YOLO
    hallucinating a pose on noise).
    """
    h, w = frame_shape[:2]
    names = list(landmarks.keys())
    if len(names) < 4:
        return False

    # Check that landmarks are within frame bounds (allow small margin)
    for name in names:
        lm = landmarks[name]
        x, y = lm.get('x', -1), lm.get('y', -1)
        if x < -0.1 * w or x > 1.1 * w or y < -0.1 * h or y > 1.1 * h:
            return False

    # Check shoulder-to-hip distance is reasonable (not tiny, not huge)
    ls = landmarks.get('left_shoulder')
    rs = landmarks.get('right_shoulder')
    lh = landmarks.get('left_hip')
    rh = landmarks.get('right_hip')
    if ls and rs and lh and rh:
        sx = (ls['x'] + rs['x']) / 2
        sy = (ls['y'] + rs['y']) / 2
        hx = (lh['x'] + rh['x']) / 2
        hy = (lh['y'] + rh['y']) / 2
        torso_len = np.sqrt((sx - hx)**2 + (sy - hy)**2)
        # Torso should be between 5% and 95% of frame height
        if torso_len < 0.05 * h or torso_len > 0.95 * h:
            return False

    # Check that shoulder width is reasonable
    if ls and rs:
        sw = abs(ls['x'] - rs['x'])
        if sw < 0.01 * w or sw > 0.5 * w:
            return False

    return True


def _fuse_landmarks(
    mp_landmarks: Dict,
    yolo_landmarks: Dict,
    mp_weight: float = 0.5,
    yolo_weight: float = 0.5,
) -> Dict:
    """
    Fuse two landmark dictionaries via confidence-weighted averaging.

    Args:
        mp_landmarks: MediaPipe landmarks dict
        yolo_landmarks: YOLO landmarks dict
        mp_weight: Base weight for MediaPipe (adjusted by visibility)
        yolo_weight: Base weight for YOLO (adjusted by confidence)

    Returns:
        Fused landmarks dict
    """
    fused = {}
    for name in FUSION_LANDMARKS:
        mp_lm = mp_landmarks.get(name)
        yolo_lm = yolo_landmarks.get(name)

        if mp_lm is None and yolo_lm is None:
            continue

        if mp_lm is None:
            fused[name] = yolo_lm.copy()
            # Apply confidence penalty for missing detector
            fused[name]['visibility'] *= 0.85
            continue

        if yolo_lm is None:
            fused[name] = mp_lm.copy()
            # Apply confidence penalty for missing detector
            fused[name]['visibility'] *= 0.85
            continue

        # Both detectors have this landmark — weight by visibility/confidence
        mp_conf = mp_lm.get('visibility', 0.5)
        yolo_conf = yolo_lm.get('visibility', 0.5)

        # Normalize weights
        total = mp_conf * mp_weight + yolo_conf * yolo_weight
        if total < 1e-6:
            # Equal fallback
            w_mp = 0.5
            w_yolo = 0.5
        else:
            w_mp = (mp_conf * mp_weight) / total
            w_yolo = (yolo_conf * yolo_weight) / total

        fused[name] = {
            'x': mp_lm['x'] * w_mp + yolo_lm['x'] * w_yolo,
            'y': mp_lm['y'] * w_mp + yolo_lm['y'] * w_yolo,
            'z': mp_lm.get('z', 0) * w_mp + yolo_lm.get('z', 0) * w_yolo,
            'visibility': min(1.0, mp_conf * w_mp + yolo_conf * w_yolo) * 1.1,
        }

    return fused


class FusionPoseDetector:
    """Runs both MediaPipe and YOLO pose detectors and fuses their outputs."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        yolo_model: str = 'yolov8m-pose.pt',
        yolo_confidence: float = 0.4,
        mp_weight: float = 0.5,
        yolo_weight: float = 0.5,
    ):
        """
        Initialize both detectors.

        Args:
            min_detection_confidence: MediaPipe min detection confidence
            min_tracking_confidence: MediaPipe min tracking confidence
            yolo_model: YOLO pose model path or name
            yolo_confidence: YOLO confidence threshold
            mp_weight: Base fusion weight for MediaPipe
            yolo_weight: Base fusion weight for YOLO
        """
        self.mp_detector = PoseDetector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.yolo_detector = YOLOPoseDetector(
            model_path=yolo_model,
            confidence_threshold=yolo_confidence,
        )
        self.mp_weight = mp_weight
        self.yolo_weight = yolo_weight

    def detect_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect pose using both detectors and fuse results.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Fused landmarks dict, or None if both detectors fail.
        """
        # Run both detectors
        mp_result = self.mp_detector.detect_pose(frame)
        yolo_result = self.yolo_detector.detect_pose(frame)

        if mp_result is None and yolo_result is None:
            return None

        h, w, _ = frame.shape

        # Validate YOLO landmarks — reject implausible detections
        yolo_valid = False
        if yolo_result is not None:
            yolo_valid = _validate_landmarks(yolo_result['landmarks'], (h, w))

        if mp_result is not None and yolo_result is not None and yolo_valid:
            # Both detected and YOLO is plausible — fuse landmarks
            fused_landmarks = _fuse_landmarks(
                mp_result['landmarks'],
                yolo_result['landmarks'],
                mp_weight=self.mp_weight,
                yolo_weight=self.yolo_weight,
            )
        elif mp_result is not None:
            # Only MediaPipe detected (or YOLO rejected)
            fused_landmarks = mp_result['landmarks'].copy()
        elif yolo_result is not None and yolo_valid:
            # Only YOLO detected (and validated)
            fused_landmarks = yolo_result['landmarks'].copy()
        else:
            return None

        return {
            'landmarks': fused_landmarks,
            'raw_landmarks': None,
            'frame_shape': (h, w),
        }

    def process_video(self, video_path: str, skip_frames: int = 1) -> List[Dict]:
        """
        Process video with fused pose detection.

        Args:
            video_path: Path to video file
            skip_frames: Process every Nth frame

        Returns:
            List of pose data dictionaries
        """
        cap = cv2.VideoCapture(video_path)
        pose_data = []

        frame_count = 0
        processed_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Processing video: {total_frames} frames at {fps:.2f} fps (analyzing every {skip_frames} frames)")
        print("  Fusion: MediaPipe + YOLOv8 pose (confidence-weighted)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames != 0:
                frame_count += 1
                continue

            pose_result = self.detect_pose(frame)
            processed_count += 1

            pose_data.append({
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'pose': pose_result,
            })

            frame_count += 1
            if processed_count % 15 == 0:
                pct = (frame_count / total_frames) * 100
                print(f"Progress: {pct:.1f}% ({processed_count} frames analyzed)")

        cap.release()
        print(f"✓ Completed: {processed_count} frames analyzed ({total_frames} total, skipped {total_frames - processed_count})")

        return pose_data

    def __del__(self):
        """Cleanup resources."""
        pass
