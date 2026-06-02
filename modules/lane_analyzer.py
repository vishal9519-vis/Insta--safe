import numpy as np


class LaneAnalyzer:
    def analyze_lane_behavior(self, trajectory, label, frame_width):
        result = {
            "lane_cut_risk":   False,
            "wrong_direction": False,
            "crossing_risk":   False,
            "lateral_movement": 0.0,
        }
        if len(trajectory) < 5:
            return result

        p = np.array(trajectory)
        result["lateral_movement"] = round(float(p[-1][0] - p[-5][0]), 1)
        result["lane_cut_risk"]    = self._lane_cut(p)
        result["wrong_direction"]  = self._wrong_dir(p, frame_width)

        if label == "person":
            result["crossing_risk"] = self._crossing(p)

        return result

    def _lane_cut(self, p):
        if len(p) < 6:
            return False
        prev_dx = p[-5][0] - p[-6][0]
        curr_dx = p[-1][0] - p[-2][0]
        if abs(prev_dx) < 3 or abs(curr_dx) < 3:
            return False
        return (prev_dx > 0 and curr_dx < -8) or (prev_dx < 0 and curr_dx > 8)

    def _crossing(self, p):
        if len(p) < 4:
            return False
        horizontal = abs(p[-1][0] - p[-4][0])
        vertical   = abs(p[-1][1] - p[-4][1])
        return horizontal > vertical * 1.2 and horizontal > 15

    def _wrong_dir(self, p, fw):
        if len(p) < 5 or fw == 0:
            return False
        avg_dx             = float(p[-1][0] - p[-5][0])
        moving_right_on_left = p[-1][0] < fw * 0.4 and avg_dx > 20
        moving_left_on_right = p[-1][0] > fw * 0.6 and avg_dx < -20
        return moving_right_on_left or moving_left_on_right
