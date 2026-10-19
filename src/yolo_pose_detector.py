"""YOLOv8 pose detection wrapper for swimming stroke analysis.

Maps COCO keypoints (17-point pose) to MediaPipe-compatible landmark format
for seamless fusion.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from ultralytics import YOLO

# COCO keypoint index to landmark name mapping
COCO_TO_LANDMARK = {
    0: 'nose',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
}

# All landmark names we need (subset of MediaPipe landmarks)
REQUIRED_LANDMARKS = [
    'nose', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]


class YOLOPoseDetector:
    """YOLOv8 pose detector that outputs MediaPipe-compatible landmarks."""

    def __init__(self, model_path: str = 'yolov8m-pose.pt', confidence_threshold: float = 0.4):
        """
        Initialize YOLO pose model.

        Args:
            model_path: Path to YOLO pose model weights
            confidence_threshold: Minimum confidence for pose detection
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect pose in a single frame using YOLO.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Dictionary with landmarks and metadata in MediaPipe format,
            or None if no pose detected.
        """
        h, w, _ = frame.shape

        # Run YOLO pose inference
        results = self.model(frame, verbose=False)

        if not results or len(results) == 0:
            return None

        # Take the first (highest confidence) detection
        result = results[0]
        if result.keypoints is None:
            return None

        keypoints = result.keypoints
        # Check if any keypoints were detected
        if len(keypoints.xy) == 0:
            return None

        keypoints_xy = keypoints.xy[0].cpu().numpy() if hasattr(keypoints.xy[0], 'cpu') else keypoints.xy[0]
        keypoints_conf = keypoints.conf[0].cpu().numpy() if hasattr(keypoints.conf[0], 'cpu') else keypoints.conf[0] if keypoints.conf is not None else None

        # Extract landmarks
        landmarks = {}
        for idx, name in COCO_TO_LANDMARK.items():
            if idx >= len(keypoints_xy):
                continue
            x_px, y_px = keypoints_xy[idx]
            # YOLO keypoints.xy returns pixel-space coordinates (not normalized)
            conf = float(keypoints_conf[idx]) if keypoints_conf is not None and idx < len(keypoints_conf) else 0.0
            # YOLO uses confidence per keypoint; treat as visibility
            landmarks[name] = {
                'x': float(x_px),
                'y': float(y_px),
                'z': 0.0,
                'visibility': conf,
            }

        # Check if we have enough landmarks (at least shoulders + hips + one arm)
        has_shoulders = 'left_shoulder' in landmarks and 'right_shoulder' in landmarks
        has_hips = 'left_hip' in landmarks and 'right_hip' in landmarks
        has_arm = 'left_elbow' in landmarks or 'right_elbow' in landmarks

        if not (has_shoulders and has_hips and has_arm):
            return None

        return {
            'landmarks': landmarks,
            'raw_landmarks': None,  # YOLO doesn't have a comparable raw format
            'frame_shape': (h, w),
        }

    def process_video(self, video_path: str, skip_frames: int = 1) -> List[Dict]:
        """
        Process video and extract pose data using YOLO.

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
        """Cleanup model resources."""
        pass
