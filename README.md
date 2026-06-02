![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-AI-orange)
![ADAS](https://img.shields.io/badge/ADAS-Automotive-red)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Perception-purple)
![Road Safety](https://img.shields.io/badge/Road-Safety-success)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen) 
# InstaSafe AI

Real-time road safety monitoring using YOLOv8 and centroid tracking. Detects dangerous driving behaviour — sudden movements, lane cuts, collision risks, and wrong-direction travel — and overlays live alerts on the video feed.

[![GitHub](https://img.shields.io/badge/GitHub-YOUR_USERNAME%2Finsta-safe-blue?logo=github)](https://github.com/YOUR_USERNAME/insta-safe)

## What it does

- Tracks vehicles and pedestrians frame by frame using centroid-based tracking
- Analyses each object's trajectory for speed, acceleration, instability, and sudden changes
- Detects lane cuts, pedestrian crossings, and wrong-direction movement
- Predicts collision risk by projecting future positions
- Scores overall scene chaos and displays a live alert panel

## Requirements

- Python 3.10+
- A webcam or video file

## Installation

```bash
git clone https://github.com/vishal9519-vis/insta-safe.git
cd insta-safe
pip install -r requirements.txt
```

The YOLOv8 nano model (`yolov8n.pt`) downloads automatically on first run if it is not already in the `models/` folder.

## Running

```bash
python main.py
```

Press **Q** to quit.

## Configuration

All settings are in `config/config.py`:

| Setting | Default | Description |
|---|---|---|
| `VIDEO_SOURCE` | `0` | Webcam index or path to a video file |
| `FRAME_WIDTH` | `1280` | Capture width |
| `FRAME_HEIGHT` | `720` | Capture height |
| `CONFIDENCE` | `0.45` | YOLO detection confidence threshold |
| `SUDDEN_SPEED_THRESHOLD` | `18` | Pixel-per-frame speed change to trigger sudden movement |
| `RISK_HIGH` | `70` | Risk score threshold for DANGER alert |
| `CHAOS_HIGH` | `65` | Scene chaos score threshold for CHAOTIC label |

To use a video file instead of a webcam, set `VIDEO_SOURCE = "path/to/video.mp4"`.

## Project structure

```
insta-safe/
├── main.py                    # Entry point
├── requirements.txt
├── config/
│   └── config.py              # All tunable parameters
├── modules/
│   ├── tracker.py             # Centroid tracker
│   ├── trajectory_analyzer.py
│   ├── lane_analyzer.py
│   ├── chaos_scorer.py
│   ├── predictor.py
│   └── alert_engine.py
├── models/                    # Place yolov8n.pt here (auto-downloaded)
└── assets/                    # Optional: test videos, screenshots
```

## Alerts

| Alert | Trigger |
|---|---|
| SUDDEN MOVEMENT | Speed change exceeds threshold in last few frames |
| HIGH SPEED | Sustained speed above 1.5× sudden threshold |
| ERRATIC PATH | Instability score above 25 |
| SUDDEN STOP | Negative acceleration below −12 |
| COLLISION RISK | Projected positions of two objects converge within 60px |
| LANE CUT | Object reverses horizontal direction sharply |
| PEDESTRIAN CROSSING | Person moves more horizontally than vertically |
| WRONG DIRECTION | Object moves against expected traffic side |

## Autor
