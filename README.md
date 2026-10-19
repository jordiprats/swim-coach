# Swim Stroke Analyzer — CLI Tool

Analyze freestyle or butterfly swimming technique from video and get coaching-quality feedback.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic analysis (freestyle)
python main.py path/to/video.mp4

# Butterfly analysis
python main.py video.mp4 --stroke butterfly

# Precision mode (slower, more accurate)
python main.py video.mp4 --precision

# Custom output path
python main.py video.mp4 -o output/analysis.mp4

# Report only (skip video)
python main.py video.mp4 --report-only

# Video only (skip report)
python main.py video.mp4 --no-report

# Analyze a specific time range
python main.py video.mp4 --from 30s --to 1m:30s
```

## How it works

1. **Pose Detection** — AI (MediaPipe / YOLOv8) identifies 33 body points per frame
2. **Metric Calculation** — Computes angles, positions, movement patterns
3. **Issue Detection** — Compares your metrics to optimal ranges
4. **Feedback Generation** — Creates prioritized, actionable tips
5. **Video Creation** — Overlays skeleton, angles, metrics on your video

## Supported Strokes

- Freestyle
- Butterfly

## Requirements

- Python 3.8+
- A video of you swimming (15–60 seconds, side view)
