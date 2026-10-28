"""Visualizes pose data and annotations on video."""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Optional
from src.video_processor import VideoProcessor
class Visualizer:
    """Creates annotated videos with pose overlays and metrics."""

    def __init__(self, lines_only: bool = False):
        """Initialize visualizer.

        Args:
            lines_only: If True, render pose skeleton and annotations on a blank
                        background instead of the original video frames.
        """
        self.lines_only = lines_only
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

        # Colors (BGR format)
        self.COLOR_SKELETON = (0, 255, 0)  # Green
        self.COLOR_CRITICAL = (0, 0, 255)  # Red
        self.COLOR_MODERATE = (0, 165, 255)  # Orange
        self.COLOR_MINOR = (255, 255, 0)  # Cyan
        self.COLOR_BOTH_ARMS = (255, 255, 0)  # Cyan — butterfly both-arm highlight
        self.COLOR_ENTRY_LINE = (255, 0, 255)  # Magenta — entry width
        self.COLOR_UNDULATION_TRACE = (255, 255, 0)  # Cyan — undulation trail
        self.COLOR_TEXT_BG = (0, 0, 0)  # Black
        self.COLOR_TEXT = (255, 255, 255)  # White

        # Undulation trace ring buffer (hip positions for last N frames)
        self._undulation_trace = []
        self._undulation_trace_maxlen = 30

        # Persistent color for elbow-diff line (keeps last state when no data)
        self._last_elbow_diff_color = (255, 255, 255)  # white fallback

        # ── Swimming-relevant skeleton connections (filtered from full MediaPipe) ──
        # Only draw connections between landmarks that are useful for stroke analysis.
        # This avoids garbage lines from face/hands/feet landmarks that are
        # underwater or incorrectly detected.
        self._swim_connections = [
            # Torso
            (11, 12),   # left_shoulder ↔ right_shoulder
            (11, 23),   # left_shoulder ↔ left_hip
            (12, 24),   # right_shoulder ↔ right_hip
            (23, 24),   # left_hip ↔ right_hip
            # Left arm
            (11, 13),   # left_shoulder ↔ left_elbow
            (13, 15),   # left_elbow ↔ left_wrist
            # Right arm
            (12, 14),   # right_shoulder ↔ right_elbow
            (14, 16),   # right_elbow ↔ right_wrist
            # Left leg
            (23, 25),   # left_hip ↔ left_knee
            (25, 27),   # left_knee ↔ left_ankle
            # Right leg
            (24, 26),   # right_hip ↔ right_knee
            (26, 28),   # right_knee ↔ right_ankle
        ]
        # Landmark indices that are relevant (swimming subset)
        self._swim_landmark_indices = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}

    def create_annotated_video(
        self,
        pose_data: List[Dict],
        output_path: str,
        analysis_results: Dict,
        original_video_path: str
    ) -> str:
        """
        Create annotated video with pose overlay and metrics.

        Frames are re-read directly from the original video (pose_data no longer
        stores frames) to avoid memory exhaustion on long videos.

        Args:
            pose_data: List of frame data with pose information (no 'frame' key)
            output_path: Path to save annotated video
            analysis_results: Results from stroke analyzer
            original_video_path: Path to original video for metadata

        Returns:
            Path to created video
        """
        if not pose_data:
            raise ValueError("No pose data to visualize")

        # Reset per-video state
        self._undulation_trace = []
        self._is_butterfly = ('undulation' in analysis_results.get('metrics', {}) and
                              analysis_results['metrics']['undulation'].get('undulation_amplitude') is not None)
        self._is_backstroke = ('body_line' in analysis_results.get('metrics', {}) and
                               (analysis_results['metrics']['body_line'].get('body_line_angle_deviation') is not None or
                                analysis_results['metrics']['body_line'].get('hip_drop') is not None))

        # Build a fast lookup: frame_number -> pose
        pose_lookup = {fd['frame_number']: fd['pose'] for fd in pose_data}
        frame_numbers = sorted(pose_lookup.keys())
        if not frame_numbers:
            raise ValueError('No frame numbers in pose_data')
        min_frame = frame_numbers[0]
        max_frame = frame_numbers[-1]

        # Get video properties
        video_info = VideoProcessor(original_video_path).get_video_info()
        total_frames = video_info['frame_count']

        # Create video writer
        writer = VideoProcessor.create_video_writer(
            output_path,
            video_info['fps'],
            video_info['width'],
            video_info['height']
        )

        # Re-open original video for frame-by-frame reading
        cap = cv2.VideoCapture(original_video_path)

        print(f"\nCreating annotated video...")
        print(f"Output: {output_path}")

        frame_idx = 0
        last_pose = None  # Forward-fill pose on frames that were skipped during analysis

        # Pre-create blank canvas once (same size as video)
        blank_canvas = np.zeros((video_info['height'], video_info['width'], 3), dtype=np.uint8)
        if self.lines_only:
            blank_canvas.fill(255)  # white background

        # Seek to first frame with pose data
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
        frame_idx = min_frame
        rendered_count = 0
        last_pose = pose_lookup[min_frame]

        while frame_idx <= max_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Use analyzed pose for this frame, or forward-fill from last known pose
            if frame_idx in pose_lookup:
                last_pose = pose_lookup[frame_idx]
            pose = last_pose

            # Decide which canvas to draw on
            if self.lines_only:
                canvas = blank_canvas.copy()
            else:
                canvas = frame

            # Draw pose if detected
            if pose is not None:
                canvas = self._draw_pose(canvas, pose)
                canvas = self._draw_metrics_overlay(canvas, pose, analysis_results)

            # Draw overall stats in corner
            canvas = self._draw_stats_panel(canvas, pose, analysis_results, frame_idx, total_frames)

            writer.write(canvas)
            rendered_count += 1
            frame_idx += 1

            if rendered_count % 30 == 0:
                print(f"Rendered {rendered_count}/{max_frame - min_frame + 1} frames")

        cap.release()
        writer.release()
        print(f"Completed: {output_path}")

        return output_path

    def create_annotated_video_4panel(
        self,
        pose_data: List[Dict],
        output_path: str,
        analysis_results: Dict,
        original_video_path: str
    ) -> str:
        """
        Create a 4-panel annotated video:
        - Top-left: original analyzed video (same as create_annotated_video)
        - Top-right: front view of the swimmer (x-y projection)
        - Bottom-left: top view (x-z projection, looking from above)
        - Bottom-right: side view (y-z projection, looking from the side)

        The 3D views keep the swimmer stationary and centered so technique
        can be analysed without camera perspective distortion.
        """
        if not pose_data:
            raise ValueError("No pose data to visualize")

        # Reset per-video state
        self._undulation_trace = []
        self._is_butterfly = ('undulation' in analysis_results.get('metrics', {}) and
                              analysis_results['metrics']['undulation'].get('undulation_amplitude') is not None)
        self._is_backstroke = ('body_line' in analysis_results.get('metrics', {}) and
                               (analysis_results['metrics']['body_line'].get('body_line_angle_deviation') is not None or
                                analysis_results['metrics']['body_line'].get('hip_drop') is not None))

        # Build pose lookup
        pose_lookup = {fd['frame_number']: fd['pose'] for fd in pose_data}
        frame_numbers = sorted(pose_lookup.keys())
        if not frame_numbers:
            raise ValueError('No frame numbers in pose_data')
        min_frame = frame_numbers[0]
        max_frame = frame_numbers[-1]

        # Get video properties
        video_info = VideoProcessor(original_video_path).get_video_info()
        orig_w = video_info['width']
        orig_h = video_info['height']
        fps = video_info['fps']

        # Output video is 2x width, 2x height (4 panels)
        out_w = orig_w * 2
        out_h = orig_h * 2

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

        # Re-open original video
        cap = cv2.VideoCapture(original_video_path)

        print(f"\nCreating 4-panel annotated video...")
        print(f"Output: {output_path}")

        # ── Estimate water plane from world landmarks ──
        # The water surface is approximated as the median y-coordinate
        # of all body landmarks across all detected frames.
        all_y = []
        for fn in frame_numbers:
            p = pose_lookup[fn]
            if p is not None and p.get('world_landmarks'):
                wld = p['world_landmarks']
                for name in ['nose', 'left_shoulder', 'right_shoulder',
                             'left_elbow', 'right_elbow',
                             'left_wrist', 'right_wrist',
                             'left_hip', 'right_hip',
                             'left_knee', 'right_knee',
                             'left_ankle', 'right_ankle']:
                    lm = wld.get(name)
                    if lm:
                        all_y.append(lm['y'])
        water_y = np.median(all_y) if all_y else 0.0
        print(f"Estimated water surface y = {water_y:.4f} (from {len(all_y)} landmarks)")

        # Seek to first frame with pose data
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
        frame_idx = min_frame
        rendered_count = 0
        last_pose = pose_lookup[min_frame]

        # ── 3D landmark names for skeleton connections ──
        landmark_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                          'left_shoulder', 'right_shoulder',
                          'left_elbow', 'right_elbow',
                          'left_wrist', 'right_wrist',
                          'left_hip', 'right_hip',
                          'left_knee', 'right_knee',
                          'left_ankle', 'right_ankle']
        # Swimming-relevant 3D connections (same as _swim_connections but using names)
        swim_connections_3d = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
        ]

        while frame_idx <= max_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Get pose
            if frame_idx in pose_lookup:
                last_pose = pose_lookup[frame_idx]
            pose = last_pose

            # ── Build 4-panel canvas ──
            # Top-left: original analyzed frame
            tl_panel = frame.copy()
            if pose is not None:
                tl_panel = self._draw_pose(tl_panel, pose)
                tl_panel = self._draw_metrics_overlay(tl_panel, pose, analysis_results)
            tl_panel = self._draw_stats_panel(tl_panel, pose, analysis_results, frame_idx, max_frame - min_frame + 1)

            panel_w = orig_w
            panel_h = orig_h

            # ── 3D views (side, top, front) using world landmarks ──
            # MediaPipe provides world_landmarks in a consistent 3D space (metres):
            #   x: left-right (horizontal position)
            #   y: up-down (vertical position, negative = up)
            #   z: forward-backward (depth, negative = forward toward camera)
            #
            # The three views project onto different planes:
            #   SIDE  (x-y plane)  — same perspective as the camera
            #   TOP   (x-z plane)  — looking down from above
            #   FRONT (z-y plane)  — looking from the front
            if pose is not None and pose.get('world_landmarks'):
                wld = pose['world_landmarks']
                pts = {}
                for name in ['nose', 'left_shoulder', 'right_shoulder',
                             'left_elbow', 'right_elbow',
                             'left_wrist', 'right_wrist',
                             'left_hip', 'right_hip',
                             'left_knee', 'right_knee',
                             'left_ankle', 'right_ankle']:
                    lm = wld.get(name)
                    if lm:
                        pts[name] = lm

                if pts:
                    # Centroid in world space
                    cx = sum(p['x'] for p in pts.values()) / len(pts)
                    cy = sum(p['y'] for p in pts.values()) / len(pts)
                    cz = sum(p['z'] for p in pts.values()) / len(pts)

                    # Bounding box extents (metres)
                    xs = [p['x'] for p in pts.values()]
                    ys = [p['y'] for p in pts.values()]
                    zs = [p['z'] for p in pts.values()]
                    sx = max(xs) - min(xs)  # body width
                    sy = max(ys) - min(ys)  # body height
                    sz = max(zs) - min(zs)  # body depth

                    # Scale so the larger dimension fills ~90% of the panel
                    scale_side  = min(panel_w / (sx + 0.001), panel_h / (sy + 0.001)) * 0.9
                    scale_top   = min(panel_w / (sx + 0.001), panel_h / (sz + 0.001)) * 0.9
                    scale_front = min(panel_w / (sz + 0.001), panel_h / (sy + 0.001)) * 0.9

                    # Helper to draw a skeleton view
                    def draw_skel(canvas, pts, connections, proj_fn,
                                  panel_w, panel_h):
                        """Project each connection and draw line+circle in skeleton colour."""
                        for src, dst in connections:
                            if src in pts and dst in pts:
                                p1 = pts[src]
                                p2 = pts[dst]
                                x1, y1 = proj_fn(p1)
                                x2, y2 = proj_fn(p2)
                                px1 = int(panel_w // 2 + x1)
                                py1 = int(panel_h // 2 + y1)
                                px2 = int(panel_w // 2 + x2)
                                py2 = int(panel_h // 2 + y2)
                                cv2.line(canvas, (px1, py1), (px2, py2),
                                         self.COLOR_SKELETON, 3)
                                cv2.circle(canvas, (px1, py1), 5,
                                           self.COLOR_SKELETON, -1)
                                cv2.circle(canvas, (px2, py2), 5,
                                           self.COLOR_SKELETON, -1)

                    # ── SIDE VIEW (x-y plane) — same perspective as camera ──
                    tr_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                    tr_panel.fill(255)
                    draw_skel(tr_panel, pts, swim_connections_3d,
                              lambda p: ((p['x'] - cx) * scale_side, (p['y'] - cy) * scale_side),
                              panel_w, panel_h)

                    # Draw water line in SIDE VIEW (horizontal line at water_y)
                    water_y_proj = int(panel_h // 2 + (water_y - cy) * scale_side)
                    water_x_proj_left = int(panel_w // 2 + (min(xs) - cx) * scale_side)
                    water_x_proj_right = int(panel_w // 2 + (max(xs) - cx) * scale_side)
                    # Draw dashed blue line for water surface
                    dash_len = 10
                    for t in range(0, 100, 2):
                        x = int(water_x_proj_left + (water_x_proj_right - water_x_proj_left) * t / 100)
                        if (t // dash_len) % 2 == 0:
                            tr_panel[water_y_proj, max(0, min(panel_w-1, x)), :] = [255, 128, 0]
                    # Also draw a more visible solid blue line segment
                    cv2.line(tr_panel,
                             (max(0, water_x_proj_left), water_y_proj),
                             (min(panel_w-1, water_x_proj_right), water_y_proj),
                             (255, 128, 0), 2)  # light blue
                    # Label in black for visibility
                    cv2.putText(tr_panel, 'WATER',
                                (max(0, water_x_proj_right + 10), water_y_proj + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                    # ── TOP VIEW (x-z plane) — looking down from above ──
                    bl_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                    bl_panel.fill(255)
                    draw_skel(bl_panel, pts, swim_connections_3d,
                              lambda p: ((p['x'] - cx) * scale_top, -(p['z'] - cz) * scale_top),
                              panel_w, panel_h)

                    # ── FRONT VIEW (z-y plane) — looking from the front ──
                    br_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                    br_panel.fill(255)
                    draw_skel(br_panel, pts, swim_connections_3d,
                              lambda p: ((p['z'] - cz) * scale_front, (p['y'] - cy) * scale_front),
                              panel_w, panel_h)

                    # Draw water line in FRONT VIEW (horizontal line at water_y)
                    water_y_proj_f = int(panel_h // 2 + (water_y - cy) * scale_front)
                    water_z_proj_top = int(panel_w // 2 + (max(zs) - cz) * scale_front)
                    water_z_proj_bottom = int(panel_w // 2 + (min(zs) - cz) * scale_front)
                    cv2.line(br_panel,
                             (max(0, water_z_proj_top), water_y_proj_f),
                             (min(panel_w-1, water_z_proj_bottom), water_y_proj_f),
                             (255, 128, 0), 2)
                    cv2.putText(br_panel, 'WATER',
                                (max(0, water_z_proj_bottom + 10), water_y_proj_f + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                    # ── Panel labels ──
                    label_font = cv2.FONT_HERSHEY_SIMPLEX
                    label_scale = 0.8
                    label_thick = 2
                    cv2.putText(tr_panel, 'SIDE VIEW', (12, 30),
                                label_font, label_scale, (0, 0, 0), label_thick)
                    cv2.putText(bl_panel, 'TOP VIEW', (12, 30),
                                label_font, label_scale, (0, 0, 0), label_thick)
                    cv2.putText(br_panel, 'FRONT VIEW', (12, 30),
                                label_font, label_scale, (0, 0, 0), label_thick)
                else:
                    tr_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                    tr_panel.fill(255)
                    bl_panel = tr_panel.copy()
                    br_panel = tr_panel.copy()
            else:
                tr_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                tr_panel.fill(255)
                bl_panel = tr_panel.copy()
                br_panel = tr_panel.copy()

            # ── Assemble 4-panel canvas ──
            canvas4 = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            canvas4[:orig_h, :orig_w] = tl_panel
            canvas4[:orig_h, orig_w:] = tr_panel
            canvas4[orig_h:, :orig_w] = bl_panel
            canvas4[orig_h:, orig_w:] = br_panel

            writer.write(canvas4)
            rendered_count += 1
            frame_idx += 1

            if rendered_count % 30 == 0:
                print(f"Rendered {rendered_count}/{max_frame - min_frame + 1} frames")

        cap.release()
        writer.release()
        print(f"Completed: {output_path}")

        return output_path

    def _draw_pose(self, frame: np.ndarray, pose: Dict) -> np.ndarray:
        """Draw pose skeleton on frame.

        Uses a filtered set of swimming-relevant connections (shoulders, arms,
        hips, legs) with visibility thresholds.  Avoids garbage lines from
        face/hands/feet landmarks that are underwater or incorrectly detected.

        When MediaPipe raw_landmarks are available (standard mode), uses the
        native MediaPipe drawing for the full skeleton.  When unavailable
        (precision/fusion mode), draws from the landmarks dict directly.

        For butterfly, both arms are highlighted with a different colour to
        emphasise simultaneous motion.  An entry-width line is drawn between
        the two wrists when both are visible.

        Interpolated poses (where pose detection failed) are drawn in grey
        to indicate they are estimations.
        """
        landmarks = pose.get('landmarks', {})
        VIS_THRESH = 0.3  # minimum visibility (lowered to accommodate precision-mode penalties)

        # Use grey color for interpolated poses
        is_interpolated = pose.get('interpolated', False)
        if is_interpolated:
            SKELETON_COLOR = (128, 128, 128)  # grey
            BOTH_ARMS_COLOR = (128, 128, 128)  # grey
        else:
            SKELETON_COLOR = self.COLOR_SKELETON
            BOTH_ARMS_COLOR = self.COLOR_BOTH_ARMS

        # ── Native MediaPipe skeleton (standard mode) ──
        if pose.get('raw_landmarks') is not None:
            self.mp_drawing.draw_landmarks(
                frame,
                pose['raw_landmarks'],
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=self.COLOR_SKELETON,
                    thickness=2,
                    circle_radius=3
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=self.COLOR_SKELETON,
                    thickness=2
                )
            )
        else:
            # ── Precision mode: draw filtered skeleton from landmarks dict ──
            name_to_idx = {
                'nose': 0,
                'left_shoulder': 11, 'right_shoulder': 12,
                'left_elbow': 13, 'right_elbow': 14,
                'left_wrist': 15, 'right_wrist': 16,
                'left_hip': 23, 'right_hip': 24,
                'left_knee': 25, 'right_knee': 26,
                'left_ankle': 27, 'right_ankle': 28,
            }
            # Draw connections
            for (src_idx, dst_idx) in self._swim_connections:
                src_name = next((k for k, v in name_to_idx.items() if v == src_idx), None)
                dst_name = next((k for k, v in name_to_idx.items() if v == dst_idx), None)
                if src_name is None or dst_name is None:
                    continue
                src_lm = landmarks.get(src_name)
                dst_lm = landmarks.get(dst_name)
                if (src_lm is not None and dst_lm is not None and
                    src_lm.get('visibility', 0) >= VIS_THRESH and
                    dst_lm.get('visibility', 0) >= VIS_THRESH):
                    x1, y1 = int(src_lm['x']), int(src_lm['y'])
                    x2, y2 = int(dst_lm['x']), int(dst_lm['y'])
                    cv2.line(frame, (x1, y1), (x2, y2), SKELETON_COLOR, 2)
            # Draw landmark dots
            for name in name_to_idx:
                lm = landmarks.get(name)
                if lm and lm.get('visibility', 0) >= VIS_THRESH:
                    x, y = int(lm['x']), int(lm['y'])
                    cv2.circle(frame, (x, y), 4, SKELETON_COLOR, -1)

        # ── Butterfly-specific overlays ──
        if self._is_backstroke:
            # Backstroke: highlight the recovering arm (straight arm above shoulder)
            # Also show body rotation indicator
            arm_segments = [
                ('left_shoulder', 'left_elbow'),
                ('left_elbow', 'left_wrist'),
                ('right_shoulder', 'right_elbow'),
                ('right_elbow', 'right_wrist'),
            ]
            for (src, dst) in arm_segments:
                src_lm = landmarks.get(src)
                dst_lm = landmarks.get(dst)
                if (src_lm and dst_lm and
                    src_lm.get('visibility', 0) > 0.3 and
                    dst_lm.get('visibility', 0) > 0.3):
                    x1, y1 = int(src_lm['x']), int(src_lm['y'])
                    x2, y2 = int(dst_lm['x']), int(dst_lm['y'])
                    cv2.line(frame, (x1, y1), (x2, y2), BOTH_ARMS_COLOR, 3)
                    cv2.circle(frame, (x1, y1), 5, BOTH_ARMS_COLOR, -1)
                    cv2.circle(frame, (x2, y2), 5, BOTH_ARMS_COLOR, -1)
            return frame

        if not self._is_butterfly:
            return frame

        # 1. Highlight visible arm segments in cyan
        arm_segments = [
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
        ]
        for (src, dst) in arm_segments:
            src_lm = landmarks.get(src)
            dst_lm = landmarks.get(dst)
            if (src_lm and dst_lm and
                src_lm.get('visibility', 0) > 0.3 and
                dst_lm.get('visibility', 0) > 0.3):
                x1, y1 = int(src_lm['x']), int(src_lm['y'])
                x2, y2 = int(dst_lm['x']), int(dst_lm['y'])
                cv2.line(frame, (x1, y1), (x2, y2), BOTH_ARMS_COLOR, 4)
                cv2.circle(frame, (x1, y1), 6, BOTH_ARMS_COLOR, -1)
                cv2.circle(frame, (x2, y2), 6, BOTH_ARMS_COLOR, -1)

        # 2. Entry-width line between wrists (magenta)
        left_wrist = landmarks.get('left_wrist')
        right_wrist = landmarks.get('right_wrist')
        if (left_wrist and right_wrist and
            left_wrist.get('visibility', 0) > 0.3 and
            right_wrist.get('visibility', 0) > 0.3):
            lx = int(left_wrist['x'])
            ly = int(left_wrist['y'])
            rx = int(right_wrist['x'])
            ry = int(right_wrist['y'])
            cv2.line(frame, (lx, ly), (rx, ry), self.COLOR_ENTRY_LINE, 2)

        # 3. Dolphin undulation trace — use whichever hip is visible
        left_hip = landmarks.get('left_hip')
        right_hip = landmarks.get('right_hip')

        hip_x, hip_y = None, None
        if left_hip and left_hip.get('visibility', 0) > 0.3:
            hip_x, hip_y = int(left_hip['x']), int(left_hip['y'])
        elif right_hip and right_hip.get('visibility', 0) > 0.3:
            hip_x, hip_y = int(right_hip['x']), int(right_hip['y'])

        if hip_x is not None:
            self._undulation_trace.append((hip_x, hip_y))
            if len(self._undulation_trace) > self._undulation_trace_maxlen:
                self._undulation_trace.pop(0)

            if len(self._undulation_trace) >= 2:
                for i in range(1, len(self._undulation_trace)):
                    alpha = i / len(self._undulation_trace)
                    color = tuple(int(c * alpha) for c in self.COLOR_UNDULATION_TRACE)
                    cv2.line(
                        frame,
                        self._undulation_trace[i - 1],
                        self._undulation_trace[i],
                        color,
                        2
                    )

        return frame

    def _draw_metrics_overlay(self, frame: np.ndarray, pose: Dict, analysis: Dict) -> np.ndarray:
        """Draw real-time metrics overlay on frame.

        For butterfly: both elbow angles are shown, plus a sync indicator
        between the two wrists.

        Skips overlay for interpolated poses (no reliable live metrics).
        """
        if pose.get('interpolated', False):
            return frame
        landmarks = pose['landmarks']
        metrics = analysis['metrics']
        is_bf = self._is_butterfly
        is_bk = self._is_backstroke

        # Choose optimal elbow range based on stroke type
        if is_bf:
            elbow_opt_min, elbow_opt_max, elbow_crit = (120, 160, 160)
        elif is_bk:
            elbow_opt_min, elbow_opt_max, elbow_crit = (120, 150, 160)
        else:
            elbow_opt_min, elbow_opt_max, elbow_crit = (80, 160, 120)

        # --- Elbow angles (left always, right too if butterfly) ---
        if metrics.get('elbow', {}).get('avg_angle') is not None:

            # Left elbow
            if (landmarks['left_shoulder']['visibility'] > 0.3 and
                landmarks['left_elbow']['visibility'] > 0.3 and
                landmarks['left_wrist']['visibility'] > 0.3):

                angle = self._calculate_angle(
                    landmarks['left_shoulder'],
                    landmarks['left_elbow'],
                    landmarks['left_wrist']
                )
                pos = (int(landmarks['left_elbow']['x']), int(landmarks['left_elbow']['y']))
                color = self._get_angle_color(angle, elbow_opt_min, elbow_opt_max, elbow_crit, reverse=False)
                self._draw_angle_annotation(frame, pos, angle, color)

            # Right elbow (butterfly and backstroke show both; freestyle shows left only)
            if (is_bf or is_bk) and (landmarks['right_shoulder']['visibility'] > 0.3 and
                landmarks['right_elbow']['visibility'] > 0.3 and
                landmarks['right_wrist']['visibility'] > 0.3):

                angle = self._calculate_angle(
                    landmarks['right_shoulder'],
                    landmarks['right_elbow'],
                    landmarks['right_wrist']
                )
                pos = (int(landmarks['right_elbow']['x']), int(landmarks['right_elbow']['y']))
                color = self._get_angle_color(angle, elbow_opt_min, elbow_opt_max, elbow_crit, reverse=False)
                self._draw_angle_annotation(frame, pos, angle, color)

        # --- Backstroke timing indicator ---
        if is_bk and metrics.get('synchronization', {}).get('phase_gap') is not None:
            left_wrist = landmarks.get('left_wrist')
            right_wrist = landmarks.get('right_wrist')
            if (left_wrist and right_wrist and
                left_wrist['visibility'] > 0.3 and
                right_wrist['visibility'] > 0.3):
                lx = left_wrist['x']
                rx = right_wrist['x']
                mid_x = int((lx + rx) / 2)
                mid_y = int((left_wrist['y'] + right_wrist['y']) / 2)
                timing_ok = metrics['synchronization'].get('timing_ok', False)
                label = "TIMING OK" if timing_ok else "TIMING OFF"
                color = self.COLOR_SKELETON if timing_ok else self.COLOR_CRITICAL
                self._draw_text(frame, label, (mid_x - 25, mid_y - 10),
                                scale=0.5, color=color, thickness=2)

        # --- Butterfly sync indicator (badge between wrists) ---
        if is_bf and metrics.get('synchronization', {}).get('sync_delta') is not None:
            left_wrist = landmarks.get('left_wrist')
            right_wrist = landmarks.get('right_wrist')
            if (left_wrist and right_wrist and
                left_wrist['visibility'] > 0.3 and
                right_wrist['visibility'] > 0.3):

                lx = left_wrist['x']
                rx = right_wrist['x']
                mid_x = int((lx + rx) / 2)
                mid_y = int((left_wrist['y'] + right_wrist['y']) / 2)

                sync_status = metrics['synchronization'].get('synchronized', False)
                label = "SYNC" if sync_status else "ASYNC"
                color = self.COLOR_SKELETON if sync_status else self.COLOR_CRITICAL

                self._draw_text(frame, label, (mid_x - 20, mid_y - 10),
                                scale=0.5, color=color, thickness=2)

        return frame

    def _draw_stats_panel(
        self,
        frame: np.ndarray,
        pose: Optional[Dict],
        analysis: Dict,
        current_frame: int,
        total_frames: int
    ) -> np.ndarray:
        """Draw stats panel in corner of frame.

        Uses a fixed set of lines so the layout never jumps.  Lines that have
        no data show "---" instead of disappearing.
        """
        h, w = frame.shape[:2]

        # Semi-transparent overlay
        overlay = frame.copy()
        panel_height = 420
        cv2.rectangle(overlay, (0, 0), (400, panel_height), self.COLOR_TEXT_BG, -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        metrics = analysis['metrics']
        landmarks = pose.get('landmarks', {}) if pose else {}

        # ── Build fixed line list ──
        lines = []
        is_bf = 'undulation' in metrics and metrics['undulation'].get('undulation_amplitude') is not None
        is_bk = 'body_line' in metrics and (metrics['body_line'].get('body_line_angle_deviation') is not None or
                                             metrics['body_line'].get('hip_drop') is not None)

        # Choose optimal elbow range based on stroke type
        if is_bf:
            elbow_opt_min, elbow_opt_max, elbow_crit = (120, 160, 160)
        elif is_bk:
            elbow_opt_min, elbow_opt_max, elbow_crit = (120, 150, 160)
        else:
            elbow_opt_min, elbow_opt_max, elbow_crit = (80, 160, 120)

        # 1. Title
        if is_bf:
            lines.append(("BUTTERFLY ANALYSIS", self.COLOR_TEXT, 0.6, 2))
        elif is_bk:
            lines.append(("BACKSTROKE ANALYSIS", self.COLOR_TEXT, 0.6, 2))
        else:
            lines.append(("FREESTYLE ANALYSIS", self.COLOR_TEXT, 0.6, 2))

        # 2. Live elbow angle (left)
        left_sh = landmarks.get('left_shoulder')
        left_el = landmarks.get('left_elbow')
        left_wr = landmarks.get('left_wrist')
        if (left_sh and left_el and left_wr and
            left_sh.get('visibility', 0) > 0.3 and
            left_el.get('visibility', 0) > 0.3 and
            left_wr.get('visibility', 0) > 0.3):
            live_angle = self._calculate_angle(left_sh, left_el, left_wr)
            color = self._get_angle_color(live_angle, elbow_opt_min, elbow_opt_max, elbow_crit, reverse=False)
            lines.append((f"Elbow(3D): {live_angle:.0f}deg", color, 0.5, 1))
        else:
            lines.append(("Elbow: ---", self.COLOR_TEXT, 0.5, 1))

        # 3. Head lift
        nose = landmarks.get('nose')
        if nose and nose.get('visibility', 0) > 0.3:
            sh_y = 0
            sh_count = 0
            for side in ('left_shoulder', 'right_shoulder'):
                s = landmarks.get(side)
                if s and s.get('visibility', 0) > 0.3:
                    sh_y += s['y']
                    sh_count += 1
            if sh_count > 0:
                sh_avg = sh_y / sh_count
                lift = nose['y'] - sh_avg
                color = self.COLOR_CRITICAL if lift > 0.15 else self.COLOR_TEXT
                lines.append((f"Head lift: {lift:.2f}", color, 0.5, 1))
            else:
                lines.append(("Head lift: ---", self.COLOR_TEXT, 0.5, 1))
        else:
            lines.append(("Head lift: ---", self.COLOR_TEXT, 0.5, 1))

        # 4. Keypoints visible count
        vis_count = sum(1 for lm in landmarks.values() if lm.get('visibility', 0) > 0.3)
        lines.append((f"Keypoints: {vis_count}/33", self.COLOR_TEXT, 0.5, 1))

        # 5. Live per-frame elbow diff (sync indicator for butterfly/backstroke)
        left_sh = landmarks.get('left_shoulder')
        left_el = landmarks.get('left_elbow')
        left_wr = landmarks.get('left_wrist')
        right_sh = landmarks.get('right_shoulder')
        right_el = landmarks.get('right_elbow')
        right_wr = landmarks.get('right_wrist')
        if (left_sh and left_el and left_wr and right_sh and right_el and right_wr and
            left_sh.get('visibility', 0) > 0.3 and left_el.get('visibility', 0) > 0.3 and
            left_wr.get('visibility', 0) > 0.3 and right_sh.get('visibility', 0) > 0.3 and
            right_el.get('visibility', 0) > 0.3 and right_wr.get('visibility', 0) > 0.3):
            left_angle = self._calculate_angle(left_sh, left_el, left_wr)
            right_angle = self._calculate_angle(right_sh, right_el, right_wr)
            diff = abs(left_angle - right_angle)
            color = self.COLOR_SKELETON if diff < 20 else self.COLOR_CRITICAL
            self._last_elbow_diff_color = color
            lines.append((f"Elbow diff: {diff:.0f}°", color, 0.5, 1))
        else:
            lines.append(("Elbow diff: ---", self._last_elbow_diff_color, 0.5, 1))

        # 6. Overall average elbow angle
        if metrics.get('elbow', {}).get('avg_angle') is not None:
            elbow_avg = metrics['elbow']['avg_angle']
            color = self._get_angle_color(elbow_avg, elbow_opt_min, elbow_opt_max, elbow_crit, reverse=False)
            lines.append((f"Avg elbow: {elbow_avg:.0f}deg", color, 0.4, 1))
        else:
            lines.append(("Avg elbow: ---", self.COLOR_TEXT, 0.4, 1))

        # ── Backstroke-specific lines ──
        if is_bk:
            # 7. Body rotation
            if metrics.get('rotation', {}).get('avg_rotation') is not None:
                rot = metrics['rotation']['avg_rotation']
                lines.append((f"Rotation: {rot:.0f}°", self.COLOR_TEXT, 0.5, 1))
            else:
                lines.append(("Rotation: ---", self.COLOR_TEXT, 0.5, 1))

            # 8. Arm timing
            if metrics.get('synchronization', {}).get('phase_gap') is not None:
                sync = metrics['synchronization']
                status = "ok" if sync['timing_ok'] else "off"
                color = self.COLOR_SKELETON if sync['timing_ok'] else self.COLOR_CRITICAL
                lines.append((f"Timing: {sync['phase_gap']:.2f} ({status})", color, 0.5, 1))
            else:
                lines.append(("Timing: ---", self.COLOR_TEXT, 0.5, 1))

            # 9. Recovery clearance
            if metrics.get('recovery', {}).get('avg_recovery_height') is not None:
                rec = metrics['recovery']['avg_recovery_height']
                color = self.COLOR_CRITICAL if rec > 0.25 or rec < 0.03 else self.COLOR_TEXT
                lines.append((f"Recovery: {rec:.2f}h", color, 0.5, 1))
            else:
                lines.append(("Recovery: ---", self.COLOR_TEXT, 0.5, 1))

            # 10. Body line
            if metrics.get('body_line', {}).get('body_line_angle_deviation') is not None:
                dev = metrics['body_line']['body_line_angle_deviation']
                color = self.COLOR_CRITICAL if dev > 15 else self.COLOR_TEXT
                lines.append((f"Body line: {dev:.1f}° dev", color, 0.5, 1))
            elif metrics.get('body_line', {}).get('hip_drop') is not None:
                hd = metrics['body_line']['hip_drop']
                color = self.COLOR_CRITICAL if hd > 0.03 else self.COLOR_TEXT
                lines.append((f"Hip drop: {hd:.3f}", color, 0.5, 1))
            else:
                lines.append(("Hip drop: ---", self.COLOR_TEXT, 0.5, 1))

            # 11. Entry width
            if metrics.get('entry', {}).get('avg_entry_width') is not None:
                ew = metrics['entry']['avg_entry_width']
                lines.append((f"Entry: {ew:.2f}w", self.COLOR_TEXT, 0.5, 1))
            else:
                lines.append(("Entry: ---", self.COLOR_TEXT, 0.5, 1))

        else:
            # ── Butterfly / Freestyle lines ──
            # 7. Butterfly recovery height
            if metrics.get('recovery', {}).get('avg_recovery_height') is not None:
                rec = metrics['recovery']['avg_recovery_height']
                color = self.COLOR_CRITICAL if rec > 0.25 else self.COLOR_TEXT
                lines.append((f"Recovery: {rec:.2f}h", color, 0.4, 1))
            else:
                lines.append(("Recovery: ---", self.COLOR_TEXT, 0.4, 1))

            # 8. Butterfly shoulder-hip phase
            if metrics.get('shoulder_hip_phase', {}).get('phase_lag') is not None:
                phase = metrics['shoulder_hip_phase']
                status = "wave" if not phase['in_phase'] else "no wave"
                color = self.COLOR_SKELETON if not phase['in_phase'] else self.COLOR_CRITICAL
                lines.append((f"Phase: {phase['phase_lag']:.2f} ({status}) {phase['cross_correlation']:.2f}", color, 0.35, 1))
            else:
                lines.append(("Phase: ---", self.COLOR_TEXT, 0.4, 1))

            # 9. Butterfly hip during breath
            if metrics.get('hip_during_breath', {}).get('hip_drop') is not None:
                hd = metrics['hip_during_breath']['hip_drop']
                color = self.COLOR_CRITICAL if hd > 0.06 else self.COLOR_TEXT
                lines.append((f"Hip drop: {hd:.3f}", color, 0.4, 1))
            else:
                lines.append(("Hip drop: ---", self.COLOR_TEXT, 0.4, 1))

            # 10. Butterfly coordination
            if metrics.get('coordination', {}).get('coordination_gap') is not None:
                coord = metrics['coordination']
                status = "ok" if coord['aligned'] else "mis"
                color = self.COLOR_SKELETON if coord['aligned'] else self.COLOR_CRITICAL
                lines.append((f"Coordination: {coord['coordination_gap']:.0f}f ({status})", color, 0.4, 1))
            else:
                lines.append(("Coordination: ---", self.COLOR_TEXT, 0.4, 1))

            # 11. Butterfly breathing timing
            if metrics.get('breathing_timing', {}).get('breath_to_entry_gap') is not None:
                bt = metrics['breathing_timing']
                status = "ok" if not bt['late_breathing'] else "late"
                color = self.COLOR_SKELETON if not bt['late_breathing'] else self.COLOR_CRITICAL
                lines.append((f"Breath: {bt['breath_to_entry_gap']:.0f}f ({status})", color, 0.4, 1))
            else:
                lines.append(("Breath: ---", self.COLOR_TEXT, 0.4, 1))

            # 12. Butterfly undulation
            if metrics.get('undulation', {}).get('undulation_amplitude') is not None:
                amp = metrics['undulation']['undulation_amplitude']
                lines.append((f"Undulation: {amp:.3f}", self.COLOR_TEXT, 0.5, 1))
            else:
                lines.append(("Undulation: ---", self.COLOR_TEXT, 0.5, 1))

            # 13. Butterfly arm synchronisation (video-wide + recent window)
            if metrics.get('synchronization', {}).get('sync_delta') is not None:
                delta = metrics['synchronization']['sync_delta']
                recent = metrics['synchronization'].get('recent_sync_delta')
                sync_status = metrics['synchronization'].get('synchronized', False)
                color = self.COLOR_SKELETON if sync_status else self.COLOR_CRITICAL
                if recent is not None:
                    lines.append((f"Sync: {delta:.1f}° (recent {recent:.1f}°)", color, 0.4, 1))
                else:
                    lines.append((f"Sync: {delta:.1f}°", color, 0.4, 1))
            else:
                lines.append(("Sync: ---", self.COLOR_TEXT, 0.4, 1))

            # 14. Butterfly entry width
            if metrics.get('entry', {}).get('avg_entry_width') is not None:
                ew = metrics['entry']['avg_entry_width']
                lines.append((f"Entry Width: {ew:.2f}x", self.COLOR_TEXT, 0.5, 1))
            else:
                lines.append(("Entry Width: ---", self.COLOR_TEXT, 0.5, 1))

        # 15. Stroke rate (shared by all)
        if metrics.get('stroke_rate', {}).get('spm') is not None:
            spm = metrics['stroke_rate']['spm']
            lines.append((f"Stroke Rate: {spm:.0f} SPM", self.COLOR_TEXT, 0.5, 1))
        else:
            lines.append(("Stroke Rate: ---", self.COLOR_TEXT, 0.5, 1))

        # ── Render all lines at fixed positions ──
        y_offset = 30
        line_height = 22
        for i, (text, color, scale, thickness) in enumerate(lines):
            if i == 0:
                self._draw_text(frame, text, (10, y_offset), scale=scale, color=color, thickness=thickness)
                y_offset += line_height + 8
            else:
                self._draw_text(frame, text, (10, y_offset), scale=scale, color=color, thickness=thickness)
                y_offset += line_height

        # Progress bar
        progress = current_frame / total_frames
        bar_width = 380
        bar_height = 10
        bar_x = 10
        bar_y = panel_height - 30

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)

        return frame

    def _draw_angle_annotation(
        self,
        frame: np.ndarray,
        position: tuple,
        angle: float,
        color: tuple
    ):
        """Draw angle annotation at position."""
        text = f"{angle:.0f}deg"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 2

        # Get text size for background
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)

        # Draw background
        bg_x1 = position[0] - 5
        bg_y1 = position[1] - text_h - 10
        bg_x2 = position[0] + text_w + 5
        bg_y2 = position[1] - 5

        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), self.COLOR_TEXT_BG, -1)

        # Draw text
        cv2.putText(frame, text, (position[0], position[1] - 10), font, scale, color, thickness)

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple,
        scale: float = 0.5,
        color: tuple = None,
        thickness: int = 1
    ):
        """Draw text with optional background."""
        if color is None:
            color = self.COLOR_TEXT

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, position, font, scale, color, thickness, cv2.LINE_AA)

    def _get_angle_color(
        self,
        angle: float,
        optimal_min: float,
        optimal_max: float,
        critical_threshold: float,
        reverse: bool = False
    ) -> tuple:
        """Get color based on angle quality."""
        if reverse:
            # For metrics where lower is worse
            if angle < critical_threshold:
                return self.COLOR_CRITICAL
            elif angle < optimal_min:
                return self.COLOR_MODERATE
            else:
                return self.COLOR_SKELETON
        else:
            # For metrics where higher is worse
            if optimal_min <= angle <= optimal_max:
                return self.COLOR_SKELETON
            elif angle > critical_threshold:
                return self.COLOR_CRITICAL
            else:
                return self.COLOR_MODERATE

    @staticmethod
    def _calculate_angle(point1: Dict, point2: Dict, point3: Dict, use_3d: bool = True) -> float:
        """Calculate angle between three points, optionally in 3D.

        When use_3d=True (default), uses (x, y, z) to compute the true 3D joint
        angle.  This accounts for foreshortening and body roll — critical for
        swimming where arms move toward/away from the camera.
        """
        if use_3d:
            p1 = np.array([point1.get('x', 0), point1.get('y', 0), point1.get('z', 0)])
            p2 = np.array([point2.get('x', 0), point2.get('y', 0), point2.get('z', 0)])
            p3 = np.array([point3.get('x', 0), point3.get('y', 0), point3.get('z', 0)])
        else:
            p1 = np.array([point1['x'], point1['y']])
            p2 = np.array([point2['x'], point2['y']])
            p3 = np.array([point3['x'], point3['y']])

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)
