import math
from config.config import SUDDEN_SPEED_THRESHOLD, RISK_HIGH


class RiskPredictor:
    WEIGHTS = {
        "sudden_movement":  35,
        "high_speed":       20,
        "high_instability": 25,
        "collision_path":   40,
        "rapid_decel":      20,
    }

    def predict(self, obj_id, motion_data, all_tracked, frame_shape):
        score = 0
        flags = []

        if motion_data["is_sudden"]:
            score += self.WEIGHTS["sudden_movement"]
            flags.append("SUDDEN_MOVEMENT")

        if motion_data["speed"] > SUDDEN_SPEED_THRESHOLD * 1.5:
            score += self.WEIGHTS["high_speed"]
            flags.append("HIGH_SPEED")

        if motion_data["instability"] > 25:
            score += self.WEIGHTS["high_instability"]
            flags.append("ERRATIC_PATH")

        if motion_data["acceleration"] < -12:
            score += self.WEIGHTS["rapid_decel"]
            flags.append("SUDDEN_STOP")

        if self._collision_likely(obj_id, all_tracked):
            score += self.WEIGHTS["collision_path"]
            flags.append("COLLISION_RISK")

        score = min(score, 100)
        alert = "DANGER" if score >= RISK_HIGH else "WARNING" if score >= 40 else "SAFE"

        label_map = {
            "SUDDEN_MOVEMENT": "Sudden movement",
            "HIGH_SPEED":      "High speed",
            "ERRATIC_PATH":    "Erratic path",
            "SUDDEN_STOP":     "Sudden stop",
            "COLLISION_RISK":  "Collision risk",
        }
        prediction = " | ".join(label_map[f] for f in flags if f in label_map) or "Normal"

        return {
            "risk_score":  score,
            "risk_flags":  flags,
            "alert_level": alert,
            "prediction":  prediction,
        }

    def _collision_likely(self, obj_id, all_tracked):
        if obj_id not in all_tracked:
            return False
        traj_a = all_tracked[obj_id]["trajectory"]
        if len(traj_a) < 3:
            return False
        future_a = self._project(traj_a)
        for other_id, other in all_tracked.items():
            if other_id == obj_id:
                continue
            traj_b = other["trajectory"]
            if len(traj_b) < 3:
                continue
            if math.dist(future_a, self._project(traj_b)) < 60:
                return True
        return False

    def _project(self, traj, steps=5):
        if len(traj) < 2:
            return traj[-1]
        recent = traj[-min(4, len(traj)):]
        dx = sum(recent[i][0] - recent[i-1][0] for i in range(1, len(recent))) / (len(recent) - 1)
        dy = sum(recent[i][1] - recent[i-1][1] for i in range(1, len(recent))) / (len(recent) - 1)
        return (traj[-1][0] + dx * steps, traj[-1][1] + dy * steps)
