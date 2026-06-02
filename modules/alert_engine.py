import time
from collections import deque
from config.config import RISK_HIGH


class AlertEngine:
    def __init__(self):
        self.alert_history   = deque(maxlen=10)
        self.last_alert_time = {}
        self.cooldown        = 2.5

    def process(self, obj_id, label, risk_data, lane_data):
        alerts = []
        now    = time.time()

        # Risk-flag alerts (sudden movement, high speed, erratic path, etc.)
        for flag in risk_data.get("risk_flags", []):
            key = (obj_id, flag)
            if now - self.last_alert_time.get(key, 0) < self.cooldown:
                continue
            alert = self._build(flag, obj_id, label)
            if alert:
                alerts.append(alert)
                self.last_alert_time[key] = now
                self.alert_history.appendleft(alert)

        # Lane-behaviour alerts
        lane_conditions = [
            (lane_data.get("lane_cut_risk"),    f"LANE CUT — {label} ID:{obj_id}",          "DANGER",  (0, 0, 230),   "lane_cut"),
            (lane_data.get("crossing_risk"),    f"PEDESTRIAN CROSSING — ID:{obj_id}",        "WARNING", (0, 140, 255), "crossing"),
            (lane_data.get("wrong_direction"),  f"WRONG DIRECTION — {label} ID:{obj_id}",    "DANGER",  (0, 0, 230),   "wrong_dir"),
        ]

        for cond, text, level, color, tag in lane_conditions:
            key = (obj_id, tag)
            if cond and now - self.last_alert_time.get(key, 0) >= self.cooldown:
                a = {"text": text, "level": level, "color": color, "time": time.strftime("%H:%M:%S")}
                alerts.append(a)
                self.last_alert_time[key] = now
                self.alert_history.appendleft(a)

        return alerts

    def get_scene_alert(self, score):
        if score >= 80:
            return {"text": "EXTREME ROAD CHAOS", "level": "DANGER",  "color": (0, 0, 230)}
        if score >= 65:
            return {"text": "HIGH TRAFFIC CHAOS", "level": "WARNING", "color": (0, 60, 200)}
        return None

    def get_recent_alerts(self):
        return list(self.alert_history)

    def _build(self, flag, obj_id, label):
        types = {
            "SUDDEN_MOVEMENT": ("SUDDEN MOVEMENT", "DANGER",  (0, 0, 230)),
            "HIGH_SPEED":      ("HIGH SPEED",       "WARNING", (0, 140, 255)),
            "ERRATIC_PATH":    ("ERRATIC PATH",     "WARNING", (0, 140, 255)),
            "SUDDEN_STOP":     ("SUDDEN STOP",      "DANGER",  (0, 0, 230)),
            "COLLISION_RISK":  ("COLLISION RISK",   "DANGER",  (0, 0, 200)),
        }
        if flag not in types:
            return None
        tp, lv, col = types[flag]
        return {
            "text":  f"{tp} — {label} ID:{obj_id}",
            "level": lv,
            "color": col,
            "time":  time.strftime("%H:%M:%S"),
        }
