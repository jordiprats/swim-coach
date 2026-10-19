"""Visualizes pose data and annotations on video."""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Dict, Optional
from src.video_processor import VideoProcessor
class Visualizer:
    """Creates annotated videos with pose overlays and metrics."""

    def __init__(self):
        """Initialize visualizer."""
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

        # Build a fast lookup: frame_number -> pose
        pose_lookup = {fd['frame_number']: fd['pose'] for fd in pose_data}

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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Use analyzed pose for this frame, or forward-fill from last known pose
            if frame_idx in pose_lookup:
                last_pose = pose_lookup[frame_idx]
            pose = last_pose

            # Draw pose if detected
            if pose is not None:
                frame = self._draw_pose(frame, pose)
                frame = self._draw_metrics_overlay(frame, pose, analysis_results)

            # Draw overall stats in corner
            frame = self._draw_stats_panel(frame, pose, analysis_results, frame_idx, total_frames)

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"Rendered {frame_idx}/{total_frames} frames")

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
        """
        landmarks = pose.get('landmarks', {})
        VIS_THRESH = 0.3  # minimum visibility (lowered to accommodate precision-mode penalties)

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
                    cv2.line(frame, (x1, y1), (x2, y2), self.COLOR_SKELETON, 2)
            # Draw landmark dots
            for name in name_to_idx:
                lm = landmarks.get(name)
                if lm and lm.get('visibility', 0) >= VIS_THRESH:
                    x, y = int(lm['x']), int(lm['y'])
                    cv2.circle(frame, (x, y), 4, self.COLOR_SKELETON, -1)

        # ── Butterfly-specific overlays ──
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
                cv2.line(frame, (x1, y1), (x2, y2), self.COLOR_BOTH_ARMS, 4)
                cv2.circle(frame, (x1, y1), 6, self.COLOR_BOTH_ARMS, -1)
                cv2.circle(frame, (x2, y2), 6, self.COLOR_BOTH_ARMS, -1)

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
        """
        landmarks = pose['landmarks']
        metrics = analysis['metrics']
        is_bf = self._is_butterfly

        # Choose optimal elbow range based on stroke type
        elbow_opt_min, elbow_opt_max, elbow_crit = (120, 160, 160) if is_bf else (80, 160, 120)

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

            # Right elbow (butterfly shows both; freestyle shows left only)
            if is_bf and (landmarks['right_shoulder']['visibility'] > 0.3 and
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

        # Choose optimal elbow range based on stroke type
        elbow_opt_min, elbow_opt_max, elbow_crit = (120, 160, 160) if is_bf else (80, 160, 120)

        # 1. Title
        if is_bf:
            lines.append(("BUTTERFLY ANALYSIS", self.COLOR_TEXT, 0.6, 2))
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

        # 5. Live per-frame elbow diff (sync indicator)
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

        # 15. Stroke rate
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
