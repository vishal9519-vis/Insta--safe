import cv2
import time
import sys
import os

# Ensure project root is on path regardless of where script is run from
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import (
    VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    YOLO_MODEL, CONFIDENCE, IOU_THRESHOLD, TRAFFIC_CLASSES,
    SHOW_TRAJECTORY, SHOW_LABELS,
    GREEN, ORANGE, RED, WHITE,
)
from modules.tracker             import CentroidTracker
from modules.trajectory_analyzer import TrajectoryAnalyzer
from modules.lane_analyzer       import LaneAnalyzer
from modules.chaos_scorer        import ChaosScorer
from modules.predictor           import RiskPredictor
from modules.alert_engine        import AlertEngine

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


def load_model():
    """Load YOLO model — downloads automatically if not present."""
    model_path = os.path.join(PROJECT_ROOT, "models", YOLO_MODEL)
    if not os.path.exists(model_path):
        print(f"[INFO] Model not found locally — downloading {YOLO_MODEL} (auto-cached by ultralytics)...")
        return YOLO(YOLO_MODEL)   # ultralytics downloads to its cache
    return YOLO(model_path)


def get_color(alert_level):
    return RED if alert_level == "DANGER" else ORANGE if alert_level == "WARNING" else GREEN


def draw_trajectory(frame, trajectory, color):
    for i in range(1, len(trajectory)):
        alpha = i / len(trajectory)
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, trajectory[i - 1], trajectory[i], c, 1)


def draw_alerts_panel(frame, recent_alerts, scene_alert, chaos_score, chaos_label, chaos_color):
    h, w = frame.shape[:2]
    panel_x = w - 340
    cv2.rectangle(frame, (panel_x, 0), (w, 160), (20, 20, 20), -1)

    cv2.putText(frame, f"CHAOS: {chaos_label} ({chaos_score})",
                (panel_x + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, chaos_color, 2)

    if scene_alert:
        cv2.putText(frame, scene_alert["text"],
                    (panel_x + 8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, scene_alert["color"], 1)

    y = 72
    for alert in recent_alerts[:4]:
        cv2.putText(frame, f"[{alert['time']}] {alert['text']}",
                    (panel_x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, alert["color"], 1)
        y += 20


def main():
    model = load_model()

    tracker       = CentroidTracker()
    traj_analyzer = TrajectoryAnalyzer()
    lane_analyzer = LaneAnalyzer()
    chaos_scorer  = ChaosScorer()
    predictor     = RiskPredictor()
    alert_engine  = AlertEngine()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"Cannot open video source: {VIDEO_SOURCE}")
        print("Tip: set VIDEO_SOURCE = 0 for webcam, or a path to a video file in config/config.py")
        sys.exit(1)

    frame_delay = 1.0 / TARGET_FPS
    prev_time   = time.time()

    print("InstaSafe AI running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or frame read failed.")
            break

        h, w = frame.shape[:2]

        results    = model(frame, conf=CONFIDENCE, iou=IOU_THRESHOLD, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in TRAFFIC_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections.append({
                "cx": cx, "cy": cy,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label": TRAFFIC_CLASSES[cls_id],
                "conf":  float(box.conf[0]),
            })

        tracked    = tracker.update(detections)
        motion_map = {}
        risk_map   = {}
        all_alerts = []

        for obj_id, obj in tracked.items():
            traj  = obj["trajectory"]
            label = obj["label"]

            motion = traj_analyzer.analyze(traj)
            lane   = lane_analyzer.analyze_lane_behavior(traj, label, w)
            risk   = predictor.predict(obj_id, motion, tracked, frame.shape)
            alerts = alert_engine.process(obj_id, label, risk, lane)

            motion_map[obj_id] = motion
            risk_map[obj_id]   = risk
            all_alerts.extend(alerts)

        chaos_score, _  = chaos_scorer.compute(list(tracked.keys()), motion_map, risk_map)
        chaos_label, chaos_color = chaos_scorer.get_chaos_label(chaos_score)
        scene_alert     = alert_engine.get_scene_alert(chaos_score)

        for obj_id, obj in tracked.items():
            cx, cy = obj["cx"], obj["cy"]
            label  = obj["label"]
            risk   = risk_map.get(obj_id, {})
            alert_level = risk.get("alert_level", "SAFE")
            color  = get_color(alert_level)

            best_box, min_dist = None, float("inf")
            for det in detections:
                d = abs(det["cx"] - cx) + abs(det["cy"] - cy)
                if d < min_dist:
                    min_dist = d
                    best_box = det

            if best_box:
                cv2.rectangle(frame,
                              (best_box["x1"], best_box["y1"]),
                              (best_box["x2"], best_box["y2"]), color, 2)

            if SHOW_TRAJECTORY and len(obj["trajectory"]) > 1:
                pts = [(int(p[0]), int(p[1])) for p in obj["trajectory"]]
                draw_trajectory(frame, pts, color)

            if SHOW_LABELS:
                score = risk.get("risk_score", 0)
                tag   = f"{label} #{obj_id} [{score}]"
                cv2.putText(frame, tag, (cx - 40, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        draw_alerts_panel(frame, alert_engine.get_recent_alerts(),
                          scene_alert, chaos_score, chaos_label, chaos_color)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

        cv2.imshow("InstaSafe AI", frame)

        elapsed = time.time() - now
        wait    = max(1, int((frame_delay - elapsed) * 1000))
        if cv2.waitKey(wait) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
