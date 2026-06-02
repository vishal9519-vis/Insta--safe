import math
import numpy as np

from config.config import SUDDEN_SPEED_THRESHOLD


class TrajectoryAnalyzer:
    def analyze(self, trajectory):
        if len(trajectory) < 2:
            return self._empty()

        positions = np.array(trajectory, dtype=np.float32)
        return {
            "speed": round(self._speed(positions), 2),
            "direction_deg": round(self._direction(positions), 1),
            "acceleration": round(self._acceleration(positions), 2),
            "instability": round(self._instability(positions), 2),
            "is_sudden": self._sudden(positions),
            "traj_length": len(trajectory),
        }

    def _speed(self, p):
        recent = p[-5:]
        if len(recent) < 2:
            return 0.0

        dists = np.sqrt(((np.diff(recent, axis=0)) ** 2).sum(axis=1))
        return float(dists.mean())

    def _direction(self, p):
        if len(p) < 2:
            return 0.0

        dx = p[-1][0] - p[-2][0]
        dy = p[-1][1] - p[-2][1]
        return math.degrees(math.atan2(-dy, dx)) % 360

    def _acceleration(self, p):
        if len(p) < 4:
            return 0.0

        dists = np.sqrt(((np.diff(p, axis=0)) ** 2).sum(axis=1))
        if len(dists) < 4:
            return 0.0

        early_speed = float(np.mean(dists[: max(1, len(dists) // 2)]))
        recent_speed = float(np.mean(dists[-3:]))
        return recent_speed - early_speed

    def _instability(self, p):
        if len(p) < 4:
            return 0.0

        diffs = np.diff(p, axis=0)
        angles = [math.degrees(math.atan2(-d[1], d[0])) % 360 for d in diffs]

        changes = []
        for i in range(1, len(angles)):
            diff = abs(angles[i] - angles[i - 1])
            changes.append(min(diff, 360 - diff))

        if not changes:
            return 0.0

        return float(np.mean(changes))

    def _sudden(self, p):
        if len(p) < 4:
            return False

        dists = np.sqrt(((np.diff(p, axis=0)) ** 2).sum(axis=1))
        if len(dists) < 3:
            return False

        return bool((float(np.max(dists)) - float(np.min(dists))) > SUDDEN_SPEED_THRESHOLD * 0.6)

    def _empty(self):
        return {
            "speed": 0.0,
            "direction_deg": 0.0,
            "acceleration": 0.0,
            "instability": 0.0,
            "is_sudden": False,
            "traj_length": 0,
        }
