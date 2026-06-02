import numpy as np
from collections import deque
from config.config import CHAOS_HIGH, CHAOS_MED


class ChaosScorer:
    def __init__(self):
        self.history = deque(maxlen=30)

    def compute(self, tracked, motion_map, risk_map):
        if not tracked:
            self.history.append(0)
            return 0, {}

        speeds    = []
        high_risk = 0
        erratic   = 0
        sudden    = 0

        for oid in tracked:
            if oid in motion_map:
                m = motion_map[oid]
                speeds.append(m["speed"])
                if m["is_sudden"]:
                    sudden += 1
                if m["instability"] > 20:
                    erratic += 1
            if oid in risk_map and risk_map[oid]["alert_level"] == "DANGER":
                high_risk += 1

        n         = len(tracked)
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        raw       = min(n * 3 + high_risk * 12 + avg_speed * 0.8 + erratic * 7, 100)
        self.history.append(int(raw))
        smoothed = int(np.mean(self.history))

        return smoothed, {
            "n_objects":  n,
            "n_high_risk": high_risk,
            "avg_speed":  round(avg_speed, 1),
            "n_sudden":   sudden,
        }

    def get_chaos_label(self, score):
        if score >= CHAOS_HIGH:
            return "CHAOTIC",  (0, 0, 230)
        elif score >= CHAOS_MED:
            return "MODERATE", (0, 140, 255)
        return "NORMAL", (0, 200, 60)
