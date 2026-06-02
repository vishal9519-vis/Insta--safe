import os

VIDEO_SOURCE  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
TARGET_FPS    = 30

YOLO_MODEL     = "yolov8n.pt"
CONFIDENCE     = 0.45
IOU_THRESHOLD  = 0.4

TRAFFIC_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    15: "cat",
    16: "dog",
    19: "cow",
}

MAX_DISAPPEARED = 30
MAX_DISTANCE    = 100
TRAJECTORY_LEN  = 20

SUDDEN_SPEED_THRESHOLD = 18
DIRECTION_CHANGE_DEG   = 45
RISK_HIGH   = 70
CHAOS_HIGH  = 65
CHAOS_MED   = 35

GREEN  = (0, 200, 60)
ORANGE = (0, 140, 255)
RED    = (0, 0, 230)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)

SHOW_TRAJECTORY = True
SHOW_LABELS     = True
