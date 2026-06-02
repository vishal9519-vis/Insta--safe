import numpy as np
import math
from config.config import SUDDEN_SPEED_THRESHOLD


class TrajectoryAnalyzer:
    def analyze(self, trajectory):
        if len(trajectory) < 2:
            return self._empty()
        positions = np.array(trajectory, dtype=np.float32)
        return {
            "speed":         round(self._speed(positions), 2),
            "direction_deg": round(self._direction(positions), 1),
            "acceleration":  round(self._acceleration(positions), 2),
            "instability":   round(self._instability(positions), 2),
            "is_sudden":     self._sudden(positions),
            "traj_length":   len(trajectory),
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
        dists    = np.sqrt(((np.diff(p, axis=0)) ** 2).sum(axis=1))
        baseline = dists[-3] if len(dists) >= 3 else dists[0]
        return float(dists[-1] - baseline)

    def _instability(self, p):
        if len(p) < 4:
            return 0.0
        diffs  = np.diff(p, axis=0)
        angles = [math.degrees(math.atan2(-d[1], d[0])) for d in diffs]
        changes = [
            min(abs(angles[i] - angles[i - 1]), 360 - abs(angles[i] - angles[i - 1]))
            for i in range(1, len(angles))
        ]
        return float(np.std(changes)) if changes else 0.0

    def _sudden(self, p):
        if len(p) < 3:
            return False
        recent = p[-4:]
        dists  = np.sqrt(((np.diff(recent, axis=0)) ** 2).sum(axis=1))
        if len(dists) < 2:
            return False
        return abs(float(dists[-1]) - float(dists[0])) > SUDDEN_SPEED_THRESHOLD * 0.6

    def _empty(self):
        return {
            "speed": 0.0, "direction_deg": 0.0,
            "acceleration": 0.0, "instability": 0.0,
            "is_sudden": False, "traj_length": 0,
        }
