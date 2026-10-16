#!/usr/bin/env python3
"""
Swim Stroke Analyzer — Main CLI Entry Point

Analyzes freestyle or butterfly swimming technique from video
and provides coaching-quality feedback.

Two modes:
  - Standard mode (default): fast, real-time-capable pipeline
  - Precision mode (--precision): trades speed for accuracy using:
      * Advanced temporal smoothing (Savitzky–Golay + One Euro + Kalman)
      * Stroke-cycle segmentation with phase-specific metrics
      * Body-relative normalization
      * Confidence propagation through all metrics
"""

import argparse
import sys
import os
from pathlib import Path

from src.pose_detector import PoseDetector
from src.fusion_pose_detector import FusionPoseDetector
from src.video_processor import VideoProcessor
from src.stroke_analyzer import StrokeAnalyzer
from src.butterfly_analyzer import ButterflyAnalyzer
from src.visualizer import Visualizer
from src.feedback_generator import FeedbackGenerator
from src.data_exporter import export_datapoints_csv
from src.pose_smoother import smooth_pose_data
from src.stroke_cycle_detector import StrokeCycleDetector


def _parse_time_str(s: str) -> float:
    """Parse a time string like '30s', '1m', '1m:30s', '90s' into seconds."""
    if s is None:
        return None
    s = s.strip()
    total = 0.0
    # Split on colon for minutes:seconds format
    if ':' in s:
        parts = s.split(':')
        for part in parts:
            part = part.strip()
            if part.endswith('m'):
                total += float(part[:-1]) * 60
            elif part.endswith('s'):
                total += float(part[:-1])
            else:
                # Plain number — treat as seconds unless it's the first part (minutes)
                if len(parts) <= 2:
                    # First part before colon = minutes, rest = seconds
                    if part == parts[0]:
                        total += float(part) * 60
                    else:
                        total += float(part)
        return total
    # Single value with or without unit
    if s.endswith('m'):
        return float(s[:-1]) * 60
    elif s.endswith('s'):
        return float(s[:-1])
    else:
        return float(s)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze swimming technique from video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick analysis (standard mode)
  python main.py video.mp4

  # Precision analysis (best accuracy, slower)
  python main.py video.mp4 --precision

  # Analyze butterfly
  python main.py video.mp4 --stroke butterfly --precision

  # Analyze a specific time range
  python main.py video.mp4 --from 30s --to 1m:30s --precision

  # Generate report only
  python main.py video.mp4 --report-only
        """
    )

    parser.add_argument(
        'video',
        help='Path to input video file'
    )

    parser.add_argument(
        '--stroke',
        choices=['freestyle', 'butterfly'],
        default='freestyle',
        help='Type of stroke to analyze (default: freestyle)'
    )

    parser.add_argument(
        '-o', '--output',
        help='Path for output video (default: input_analyzed.mp4)',
        default=None
    )

    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Generate text report only, skip video visualization'
    )

    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip text report, only generate annotated video'
    )

    parser.add_argument(
        '--analyze-every',
        type=int,
        default=1,
        help='Process every Nth frame (1 = every frame, 2 = every other). Default: 1'
    )

    parser.add_argument(
        '--precision',
        action='store_true',
        help='Enable precision mode: temporal smoothing + YOLO+MediaPipe fusion + '
             'stroke-cycle segmentation + body-relative normalization + confidence '
             'propagation. Slower but significantly more accurate coaching feedback.'
    )

    parser.add_argument(
        '--from',
        type=str,
        default=None,
        dest='from_time',
        help='Start time for analysis (e.g. "30s", "1m", "1m:30s"). '
             'Trims video before analysis.'
    )
    parser.add_argument(
        '--to',
        type=str,
        default=None,
        dest='to_time',
        help='End time for analysis (e.g. "1m:40s", "90s"). '
             'Trims video before analysis.'
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Set output path
    if args.output is None:
        args.output = VideoProcessor.generate_output_path(args.video)

    # Parse time range
    time_from = _parse_time_str(args.from_time) if args.from_time else None
    time_to = _parse_time_str(args.to_time) if args.to_time else None
    if time_from is not None and time_to is not None and time_from >= time_to:
        print(f"Error: --from ({args.from_time}={time_from:.1f}s) must be before --to ({args.to_time}={time_to:.1f}s)")
        sys.exit(1)

    print("=" * 60)
    print(f"SWIM STROKE ANALYZER — {args.stroke.upper()}")
    if args.precision:
        print("  MODE: PRECISION (accuracy-first, slower)")
    else:
        print("  MODE: STANDARD (speed-optimized)")
    if time_from is not None or time_to is not None:
        range_str = ""
        if time_from is not None:
            range_str += f"from {args.from_time} ({time_from:.1f}s)"
        if time_to is not None:
            if range_str:
                range_str += " "
            range_str += f"to {args.to_time} ({time_to:.1f}s)"
        print(f"  TIME RANGE: {range_str}")
    print("=" * 60)
    print(f"Input video: {args.video}")
    if not args.report_only:
        print(f"Output video: {args.output}")
    print("")

    try:
        # Step 1: Extract pose data from video
        print("Step 1/4: Detecting pose in video frames...")
        if args.precision:
            print("  Using fusion detector: MediaPipe + YOLOv8 pose")
            detector = FusionPoseDetector()
        else:
            detector = PoseDetector()
        pose_data = detector.process_video(args.video, skip_frames=args.analyze_every)

        if not pose_data:
            print("Error: Failed to process video")
            sys.exit(1)

        print(f"✓ Processed {len(pose_data)} frames")
        print("")

        # ── Time-range trimming ──
        if time_from is not None or time_to is not None:
            original_len = len(pose_data)
            filtered = []
            for pd in pose_data:
                t = pd['timestamp']
                if time_from is not None and t < time_from:
                    continue
                if time_to is not None and t > time_to:
                    continue
                filtered.append(pd)
            pose_data = filtered
            print(f"✓ Time range trimmed: {original_len} → {len(pose_data)} frames")
            if not pose_data:
                print("Error: No frames remain after time-range filtering")
                sys.exit(1)
            print("")

        # ── Precision mode: advanced temporal smoothing ──
        if args.precision:
            print("── PRECISION MODE ──")
            smooth_pose_data(pose_data)
            print("")

        # ── Always export datapoints CSV for offline debugging ──
        datapoints_path = os.path.splitext(args.output)[0] + '_datapoints.csv'
        export_datapoints_csv(pose_data, datapoints_path)
        print(f"✓ Datapoints saved to {datapoints_path}")
        print("")

        # Step 2: Analyze stroke mechanics
        stroke_label = args.stroke.capitalize()
        print(f"Step 2/4: Analyzing {stroke_label} stroke mechanics...")

        if args.stroke == 'butterfly':
            analyzer = ButterflyAnalyzer()
        else:
            analyzer = StrokeAnalyzer(use_cycle_detection=args.precision)

        analysis_results = analyzer.analyze_video(pose_data)

        if 'error' in analysis_results:
            print(f"Error: {analysis_results['error']}")
            sys.exit(1)

        print("✓ Analysis complete")
        print("")

        # Step 3: Generate feedback report
        if not args.no_report:
            print("Step 3/4: Generating feedback report...")
            feedback = FeedbackGenerator()
            report = feedback.generate_report(analysis_results, args.stroke)
            print("✓ Report generated")
            print("")
            print(report)
            print("")

            # Save report to file
            report_path = os.path.splitext(args.output)[0] + '_report.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {report_path}")
            print("")
        else:
            print("Step 3/4: Skipping report generation")
            print("")

        # Step 4: Create annotated video
        if not args.report_only:
            print("Step 4/4: Creating annotated video...")
            visualizer = Visualizer()
            output_video = visualizer.create_annotated_video(
                pose_data,
                args.output,
                analysis_results,
                args.video
            )
            print(f"✓ Video saved to: {output_video}")

            # Best-effort H.264 re-encode for browser playback
            if VideoProcessor.reencode_for_browser(output_video):
                print("✓ Re-encoded to H.264 for browser compatibility")
            else:
                print("  (ffmpeg not found — skipping browser re-encode)")
            print("")
        else:
            print("Step 4/4: Skipping video generation")
            print("")

        print("=" * 60)
        print(f"{args.stroke.upper()} ANALYSIS COMPLETE")
        print("=" * 60)

        # Quick summary
        feedback = FeedbackGenerator()
        summary = feedback.generate_summary(analysis_results, args.stroke)
        print(summary)
        print("")

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
