![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-AI-orange)
![ADAS](https://img.shields.io/badge/ADAS-Automotive-red)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Perception-purple)
![Road Safety](https://img.shields.io/badge/Road-Safety-success)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Tests](https://img.shields.io/badge/Tests-Pytest-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

# InstaSafe AI

Real-time road safety monitoring using YOLOv8 and centroid tracking. Detects dangerous driving behaviour, sudden movements, lane cuts, collision risks, and wrong-direction travel while providing live alerts and risk assessment.

---

## Overview

InstaSafe AI is an intelligent road safety monitoring system designed to improve traffic awareness through computer vision and AI. The system processes live video streams, tracks vehicles and pedestrians, predicts potential risks, and generates real-time alerts.

### Key Features

- Real-time object detection using YOLOv8
- Vehicle and pedestrian tracking
- Collision risk prediction
- Lane cut detection
- Wrong direction detection
- Pedestrian crossing analysis
- Scene chaos scoring
- Live alert dashboard
- Real-time risk monitoring
- Lightweight and fast processing

---

## Technologies Used

- Python 3.10+
- YOLOv8
- OpenCV
- NumPy
- Ultralytics
- Computer Vision
- AI-Based Risk Assessment
- ADAS Concepts

---

## Requirements

- Python 3.10+
- Webcam or Video Input
- Windows / Linux / macOS

---

## Installation

Clone the repository:

```bash
git clone https://github.com/vishal9519-vis/insta-safe.git
cd insta-safe
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The YOLOv8 Nano model (`yolov8n.pt`) will automatically download during the first execution if it is not already available.

---

## Running the Project

```bash
python main.py
```

Press **Q** to quit.

---

## Running Tests

Execute all tests:

```bash
pytest
```

Install pytest if required:

```bash
pip install pytest
pytest
```

Run a specific test file:

```bash
pytest tests/test_tracker.py
```

---

## Configuration

All configuration parameters are available in:

```bash
config/config.py
```

| Setting | Default | Purpose |
|----------|----------|----------|
| VIDEO_SOURCE | 0 | Webcam or video file |
| FRAME_WIDTH | 1280 | Video width |
| FRAME_HEIGHT | 720 | Video height |
| CONFIDENCE | 0.45 | Detection confidence |
| SUDDEN_SPEED_THRESHOLD | 18 | Sudden movement threshold |
| RISK_HIGH | 70 | High risk alert threshold |
| CHAOS_HIGH | 65 | Scene chaos threshold |

Example:

```python
VIDEO_SOURCE = "video.mp4"
```

---

## Project Structure

```text
insta-safe/
│
├── main.py
├── requirements.txt
│
├── config/
│   └── config.py
│
├── modules/
│   ├── tracker.py
│   ├── trajectory_analyzer.py
│   ├── lane_analyzer.py
│   ├── chaos_scorer.py
│   ├── predictor.py
│   └── alert_engine.py
│
├── tests/
│   ├── test_tracker.py
│   ├── test_predictor.py
│   └── test_alert_engine.py
│
├── models/
│   └── yolov8n.pt
│
├── assets/
│   ├── screenshots/
│   └── videos/
│
└── README.md
```

---

## Alert Types

| Alert | Description |
|---------|------------|
| SUDDEN MOVEMENT | Rapid speed change detected |
| HIGH SPEED | Vehicle exceeds safe threshold |
| ERRATIC PATH | Unstable movement pattern |
| SUDDEN STOP | Hard braking detected |
| COLLISION RISK | Potential future collision |
| LANE CUT | Sudden lane change |
| PEDESTRIAN CROSSING | Pedestrian crossing vehicle path |
| WRONG DIRECTION | Opposite traffic movement |

---

## Applications

- Smart Cities
- Traffic Monitoring
- Road Safety Systems
- Intelligent Transportation Systems
- Autonomous Vehicle Research
- ADAS Development
- Computer Vision Projects
- AI Surveillance

---

## Future Enhancements

- Multi-camera support
- Vehicle classification
- Traffic density analytics
- Cloud dashboard integration
- Driver behaviour analysis
- Number plate recognition
- Edge AI deployment
- Smart traffic signal integration

---

## Screenshots

Add screenshots inside:

```text
assets/screenshots/
```

Example:

```md
### Dashboard

![Dashboard](assets/screenshots/dashboard.png)

### Risk Monitoring

![Risk](assets/screenshots/risk_monitor.png)
```

---

## Performance

- Real-Time Processing
- Lightweight Architecture
- Low Hardware Requirements
- Fast Object Tracking
- Scalable Design

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch
3. Commit changes
4. Push changes
5. Open a Pull Request

---

## Author

### Vishal Pitla

B.Tech CSE (Artificial Intelligence)

GitHub:
https://github.com/vishal9519-vis

LinkedIn:
https://www.linkedin.com/

---

## License

This project is released under the MIT License.

---
