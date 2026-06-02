"""
tests/test_instasafe.py
=======================
Pytest test suite for InstaSafe AI — covers all core modules:
  - CentroidTracker
  - TrajectoryAnalyzer
  - LaneAnalyzer
  - ChaosScorer
  - RiskPredictor
  - AlertEngine

Run:
    pip install pytest
    pytest tests/test_instasafe.py -v
"""

import math
import sys
import os
import time

import pytest

# ── make sure the project root is importable ──────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.tracker             import CentroidTracker
from modules.trajectory_analyzer import TrajectoryAnalyzer
from modules.lane_analyzer       import LaneAnalyzer
from modules.chaos_scorer        import ChaosScorer
from modules.predictor           import RiskPredictor
from modules.alert_engine        import AlertEngine


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_detection(cx=100, cy=200, label="car", conf=0.9):
    return {"cx": cx, "cy": cy, "x1": cx-20, "y1": cy-20,
            "x2": cx+20, "y2": cy+20, "label": label, "conf": conf}


def straight_trajectory(n=10, step=5, start=(100, 300)):
    """Returns a trajectory moving straight right."""
    return [(start[0] + i * step, start[1]) for i in range(n)]


def zigzag_trajectory(n=12, amplitude=30, start=(100, 300)):
    """Returns a very erratic trajectory (lane-cutting / high instability)."""
    pts = []
    for i in range(n):
        x = start[0] + i * 10
        y = start[1] + (amplitude if i % 2 == 0 else -amplitude)
        pts.append((x, y))
    return pts


def stopped_trajectory(n=10, pos=(100, 200)):
    """Returns a trajectory of stationary points (no movement)."""
    return [pos] * n


# ─────────────────────────────────────────────────────────────────────────────
# CentroidTracker
# ─────────────────────────────────────────────────────────────────────────────

class TestCentroidTracker:

    def test_register_on_first_detection(self):
        tracker = CentroidTracker()
        detections = [make_detection(100, 200)]
        tracked = tracker.update(detections)
        assert len(tracked) == 1

    def test_tracks_same_object_across_frames(self):
        tracker = CentroidTracker()
        tracked1 = tracker.update([make_detection(100, 200)])
        tracked2 = tracker.update([make_detection(105, 205)])
        id1 = list(tracked1.keys())[0]
        id2 = list(tracked2.keys())[0]
        assert id1 == id2, "Same object should keep the same ID"

    def test_new_id_for_distant_object(self):
        tracker = CentroidTracker()
        tracker.update([make_detection(100, 200)])
        tracked2 = tracker.update([make_detection(900, 800)])
        # Original object may have disappeared; either 1 or 2 objects total
        assert len(tracked2) >= 1

    def test_disappears_after_max_frames(self):
        from config.config import MAX_DISAPPEARED
        tracker = CentroidTracker()
        tracker.update([make_detection(100, 200)])
        for _ in range(MAX_DISAPPEARED + 2):
            tracked = tracker.update([])   # no detections
        assert len(tracked) == 0, "Object should be deregistered after max disappearance"

    def test_trajectory_grows(self):
        tracker = CentroidTracker()
        for i in range(5):
            tracker.update([make_detection(100 + i * 5, 200)])
        obj = list(tracker.objects.items())[0]
        obj_id = obj[0]
        assert len(tracker.trajectories[obj_id]) == 5

    def test_trajectory_capped_at_max_len(self):
        from config.config import TRAJECTORY_LEN
        tracker = CentroidTracker()
        for i in range(TRAJECTORY_LEN + 10):
            tracker.update([make_detection(100 + i * 2, 200)])
        obj_id = list(tracker.objects.keys())[0]
        assert len(tracker.trajectories[obj_id]) <= TRAJECTORY_LEN

    def test_multiple_objects_tracked_independently(self):
        tracker = CentroidTracker()
        dets = [make_detection(100, 200, "car"), make_detection(500, 600, "truck")]
        tracked = tracker.update(dets)
        assert len(tracked) == 2

    def test_empty_detections_returns_empty(self):
        tracker = CentroidTracker()
        result = tracker.update([])
        assert result == {}

    def test_label_stored_correctly(self):
        tracker = CentroidTracker()
        tracked = tracker.update([make_detection(100, 200, label="bus")])
        obj = list(tracked.values())[0]
        assert obj["label"] == "bus"


# ─────────────────────────────────────────────────────────────────────────────
# TrajectoryAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class TestTrajectoryAnalyzer:

    def setup_method(self):
        self.analyzer = TrajectoryAnalyzer()

    def test_empty_trajectory_returns_zeros(self):
        result = self.analyzer.analyze([])
        assert result["speed"] == 0.0
        assert result["is_sudden"] is False

    def test_single_point_returns_zeros(self):
        result = self.analyzer.analyze([(100, 200)])
        assert result["speed"] == 0.0

    def test_straight_movement_has_low_instability(self):
        traj = straight_trajectory(n=15, step=5)
        result = self.analyzer.analyze(traj)
        assert result["instability"] < 10, "Straight path should have low instability"

    def test_zigzag_has_high_instability(self):
        traj = zigzag_trajectory(n=15)
        result = self.analyzer.analyze(traj)
        assert result["instability"] > 10, "Erratic path should have high instability"

    def test_stationary_speed_is_zero(self):
        traj = stopped_trajectory(n=10)
        result = self.analyzer.analyze(traj)
        assert result["speed"] == 0.0

    def test_moving_object_has_positive_speed(self):
        traj = straight_trajectory(n=10, step=10)
        result = self.analyzer.analyze(traj)
        assert result["speed"] > 0

    def test_direction_eastward_is_near_zero(self):
        traj = [(i * 10, 100) for i in range(10)]
        result = self.analyzer.analyze(traj)
        # Moving right → angle near 0°
        assert result["direction_deg"] < 10 or result["direction_deg"] > 350

    def test_direction_northward_is_near_90(self):
        traj = [(100, i * -10) for i in range(10)]   # y decreases → moving up
        result = self.analyzer.analyze(traj)
        assert 80 < result["direction_deg"] < 100

    def test_sudden_flag_set_on_large_speed_change(self):
        # Start slow, then big jump
        traj = [(i * 2, 100) for i in range(5)] + [(100 + i * 40, 100) for i in range(5)]
        result = self.analyzer.analyze(traj)
        assert result["is_sudden"] is True

    def test_acceleration_positive_on_speedup(self):
        slow = [(i * 1, 100) for i in range(5)]
        fast = [(5 + i * 20, 100) for i in range(5)]
        result = self.analyzer.analyze(slow + fast)
        assert result["acceleration"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# LaneAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class TestLaneAnalyzer:

    def setup_method(self):
        self.analyzer = LaneAnalyzer()
        self.frame_width = 1280

    def test_short_trajectory_returns_all_false(self):
        traj = [(100, 200), (101, 201)]
        result = self.analyzer.analyze_lane_behavior(traj, "car", self.frame_width)
        assert result["lane_cut_risk"] is False
        assert result["wrong_direction"] is False
        assert result["crossing_risk"] is False

    def test_no_lane_cut_on_straight_path(self):
        traj = straight_trajectory(n=10, step=5)
        result = self.analyzer.analyze_lane_behavior(traj, "car", self.frame_width)
        assert result["lane_cut_risk"] is False

    def test_lane_cut_detected_on_sharp_lateral_reversal(self):
        # Move steadily right then suddenly hard left
        traj = [(100 + i * 10, 300) for i in range(6)] + \
               [(160 - i * 15, 300) for i in range(4)]
        result = self.analyzer.analyze_lane_behavior(traj, "car", self.frame_width)
        assert result["lane_cut_risk"] is True

    def test_pedestrian_crossing_detected(self):
        # Person moving mostly horizontally
        traj = [(50 + i * 20, 300 + i * 2) for i in range(8)]
        result = self.analyzer.analyze_lane_behavior(traj, "person", self.frame_width)
        assert result["crossing_risk"] is True

    def test_crossing_not_triggered_for_car(self):
        traj = [(50 + i * 20, 300 + i * 2) for i in range(8)]
        result = self.analyzer.analyze_lane_behavior(traj, "car", self.frame_width)
        assert result["crossing_risk"] is False

    def test_wrong_direction_left_side_moving_right(self):
        # Object on the LEFT side (<40% of width) moving right
        traj = [(100 + i * 25, 300) for i in range(8)]  # 100 < 512 (40% of 1280)
        result = self.analyzer.analyze_lane_behavior(traj, "car", self.frame_width)
        assert result["wrong_direction"] is True

    def test_no_wrong_direction_on_normal_path(self):
        # Object in the center moving gently right
        traj = [(600 + i * 5, 300) for i in range(8)]
        result = self.analyzer.analyze_lane_behavior(traj, "car", self.frame_width)
        assert result["wrong_direction"] is False

    def test_lateral_movement_computed_correctly(self):
        traj = [(100 + i * 10, 300) for i in range(8)]
        result = self.analyzer.analyze_lane_behavior(traj, "car", self.frame_width)
        # last point x=170, point[-5] x=120  → diff = 50
        assert result["lateral_movement"] == pytest.approx(50.0)


# ─────────────────────────────────────────────────────────────────────────────
# ChaosScorer
# ─────────────────────────────────────────────────────────────────────────────

class TestChaosScorer:

    def setup_method(self):
        self.scorer = ChaosScorer()

    def _dummy_motion(self, speed=5.0, sudden=False, instability=5.0, accel=0.0):
        return {
            "speed": speed, "is_sudden": sudden,
            "instability": instability, "acceleration": accel,
            "direction_deg": 0.0, "traj_length": 10,
        }

    def test_no_objects_returns_zero(self):
        score, _ = self.scorer.compute([], {}, {})
        assert score == 0

    def test_many_high_risk_objects_gives_high_score(self):
        n = 10
        tracked   = list(range(n))
        motion_map = {i: self._dummy_motion(speed=30, sudden=True, instability=30) for i in range(n)}
        risk_map   = {i: {"alert_level": "DANGER"} for i in range(n)}
        score, _ = self.scorer.compute(tracked, motion_map, risk_map)
        assert score > 50

    def test_calm_scene_gives_low_score(self):
        tracked    = [0, 1]
        motion_map = {0: self._dummy_motion(), 1: self._dummy_motion()}
        risk_map   = {0: {"alert_level": "SAFE"}, 1: {"alert_level": "SAFE"}}
        score, _ = self.scorer.compute(tracked, motion_map, risk_map)
        assert score < 35

    def test_score_capped_at_100(self):
        n = 20
        tracked    = list(range(n))
        motion_map = {i: self._dummy_motion(speed=100, sudden=True, instability=50) for i in range(n)}
        risk_map   = {i: {"alert_level": "DANGER"} for i in range(n)}
        score, _ = self.scorer.compute(tracked, motion_map, risk_map)
        assert score <= 100

    def test_chaos_label_normal(self):
        label, color = self.scorer.get_chaos_label(10)
        assert label == "NORMAL"
        assert color == (0, 200, 60)

    def test_chaos_label_moderate(self):
        label, _ = self.scorer.get_chaos_label(50)
        assert label == "MODERATE"

    def test_chaos_label_chaotic(self):
        label, _ = self.scorer.get_chaos_label(80)
        assert label == "CHAOTIC"

    def test_smoothing_over_history(self):
        tracked    = [0]
        motion_map = {0: self._dummy_motion(speed=100, sudden=True, instability=50)}
        risk_map   = {0: {"alert_level": "DANGER"}}

        # First call — raw spike
        s1, _ = self.scorer.compute(tracked, motion_map, risk_map)

        # Calm frames after the spike
        calm_motion  = {0: self._dummy_motion()}
        calm_risk    = {0: {"alert_level": "SAFE"}}
        for _ in range(5):
            s2, _ = self.scorer.compute([0], calm_motion, calm_risk)

        # Score should have dropped but not be zero immediately (smoothing)
        assert s2 < s1


# ─────────────────────────────────────────────────────────────────────────────
# RiskPredictor
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskPredictor:

    def setup_method(self):
        self.predictor = RiskPredictor()

    def _motion(self, sudden=False, speed=5, instability=5, accel=0):
        return {
            "is_sudden": sudden, "speed": speed,
            "instability": instability, "acceleration": accel,
            "direction_deg": 0.0, "traj_length": 10,
        }

    def _tracked(self, obj_id, traj):
        return {obj_id: {"trajectory": traj, "label": "car", "cx": traj[-1][0], "cy": traj[-1][1]}}

    def test_safe_motion_gives_low_score(self):
        traj = straight_trajectory(n=10)
        tracked = self._tracked(0, traj)
        result = self.predictor.predict(0, self._motion(), tracked, (720, 1280, 3))
        assert result["risk_score"] < 40
        assert result["alert_level"] == "SAFE"

    def test_sudden_movement_adds_to_score(self):
        traj = straight_trajectory(n=10)
        tracked = self._tracked(0, traj)
        result = self.predictor.predict(0, self._motion(sudden=True), tracked, (720, 1280, 3))
        assert "SUDDEN_MOVEMENT" in result["risk_flags"]
        assert result["risk_score"] >= 35

    def test_high_speed_adds_to_score(self):
        from config.config import SUDDEN_SPEED_THRESHOLD
        traj = straight_trajectory(n=10)
        tracked = self._tracked(0, traj)
        fast_motion = self._motion(speed=SUDDEN_SPEED_THRESHOLD * 2)
        result = self.predictor.predict(0, fast_motion, tracked, (720, 1280, 3))
        assert "HIGH_SPEED" in result["risk_flags"]

    def test_erratic_path_adds_to_score(self):
        traj = straight_trajectory(n=10)
        tracked = self._tracked(0, traj)
        result = self.predictor.predict(0, self._motion(instability=30), tracked, (720, 1280, 3))
        assert "ERRATIC_PATH" in result["risk_flags"]

    def test_rapid_decel_adds_to_score(self):
        traj = straight_trajectory(n=10)
        tracked = self._tracked(0, traj)
        result = self.predictor.predict(0, self._motion(accel=-15), tracked, (720, 1280, 3))
        assert "SUDDEN_STOP" in result["risk_flags"]

    def test_danger_level_on_high_score(self):
        from config.config import SUDDEN_SPEED_THRESHOLD
        traj = straight_trajectory(n=10)
        tracked = self._tracked(0, traj)
        motion = self._motion(sudden=True, speed=SUDDEN_SPEED_THRESHOLD * 2,
                              instability=30, accel=-15)
        result = self.predictor.predict(0, motion, tracked, (720, 1280, 3))
        assert result["alert_level"] == "DANGER"

    def test_collision_risk_two_converging_objects(self):
        # Two trajectories heading toward the same point
        traj_a = [(100 + i * 10, 300) for i in range(6)]
        traj_b = [(250 - i * 10, 300) for i in range(6)]
        tracked = {
            0: {"trajectory": traj_a, "label": "car", "cx": traj_a[-1][0], "cy": traj_a[-1][1]},
            1: {"trajectory": traj_b, "label": "car", "cx": traj_b[-1][0], "cy": traj_b[-1][1]},
        }
        result = self.predictor.predict(0, self._motion(), tracked, (720, 1280, 3))
        assert "COLLISION_RISK" in result["risk_flags"]

    def test_score_capped_at_100(self):
        from config.config import SUDDEN_SPEED_THRESHOLD
        traj = straight_trajectory(n=10)
        tracked = self._tracked(0, traj)
        motion = self._motion(sudden=True, speed=SUDDEN_SPEED_THRESHOLD * 10,
                              instability=50, accel=-20)
        result = self.predictor.predict(0, motion, tracked, (720, 1280, 3))
        assert result["risk_score"] <= 100

    def test_prediction_text_is_string(self):
        traj = straight_trajectory(n=10)
        tracked = self._tracked(0, traj)
        result = self.predictor.predict(0, self._motion(), tracked, (720, 1280, 3))
        assert isinstance(result["prediction"], str)


# ─────────────────────────────────────────────────────────────────────────────
# AlertEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertEngine:

    def setup_method(self):
        self.engine = AlertEngine()

    def _risk(self, flags=None, level="SAFE", score=0):
        return {"risk_flags": flags or [], "alert_level": level, "risk_score": score}

    def _lane(self, cut=False, crossing=False, wrong=False):
        return {"lane_cut_risk": cut, "crossing_risk": crossing,
                "wrong_direction": wrong, "lateral_movement": 0.0}

    def test_no_alerts_on_safe_motion(self):
        alerts = self.engine.process(0, "car", self._risk(), self._lane())
        assert alerts == []

    def test_sudden_movement_generates_alert(self):
        risk = self._risk(flags=["SUDDEN_MOVEMENT"], level="DANGER", score=35)
        alerts = self.engine.process(0, "car", risk, self._lane())
        assert any("SUDDEN" in a["text"].upper() for a in alerts)

    def test_lane_cut_generates_alert(self):
        alerts = self.engine.process(0, "car", self._risk(), self._lane(cut=True))
        assert any("LANE" in a["text"].upper() for a in alerts)

    def test_pedestrian_crossing_alert(self):
        alerts = self.engine.process(1, "person", self._risk(), self._lane(crossing=True))
        assert any("PEDESTRIAN" in a["text"].upper() or "CROSSING" in a["text"].upper() for a in alerts)

    def test_wrong_direction_alert(self):
        alerts = self.engine.process(2, "car", self._risk(), self._lane(wrong=True))
        assert any("WRONG" in a["text"].upper() for a in alerts)

    def test_cooldown_prevents_duplicate_alerts(self):
        risk = self._risk(flags=["SUDDEN_MOVEMENT"], level="DANGER", score=35)
        alerts1 = self.engine.process(0, "car", risk, self._lane())
        alerts2 = self.engine.process(0, "car", risk, self._lane())  # immediate repeat
        assert len(alerts2) == 0, "Cooldown should suppress repeated alert"

    def test_get_recent_alerts_returns_list(self):
        risk = self._risk(flags=["HIGH_SPEED"], level="WARNING", score=40)
        self.engine.process(0, "car", risk, self._lane())
        recent = self.engine.get_recent_alerts()
        assert isinstance(recent, list)
        assert len(recent) >= 1

    def test_scene_alert_none_at_low_score(self):
        assert self.engine.get_scene_alert(20) is None

    def test_scene_alert_warning_at_moderate(self):
        alert = self.engine.get_scene_alert(66)
        assert alert is not None
        assert alert["level"] == "WARNING"

    def test_scene_alert_danger_at_extreme(self):
        alert = self.engine.get_scene_alert(85)
        assert alert is not None
        assert alert["level"] == "DANGER"

    def test_alert_history_bounded(self):
        risk = self._risk(flags=["SUDDEN_MOVEMENT"], level="DANGER", score=40)
        # Exhaust cooldown by using different object IDs
        for i in range(15):
            self.engine.process(i, "car", risk, self._lane())
        assert len(self.engine.get_recent_alerts()) <= 10

    def test_multiple_flags_generate_multiple_alerts(self):
        risk = self._risk(flags=["SUDDEN_MOVEMENT", "COLLISION_RISK"], level="DANGER", score=75)
        alerts = self.engine.process(5, "car", risk, self._lane())
        assert len(alerts) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Integration smoke-test (no camera / YOLO required)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    """
    Runs one frame of the full pipeline using mocked detections,
    without touching the camera or YOLO.
    """

    def test_full_pipeline_single_frame(self):
        tracker       = CentroidTracker()
        traj_analyzer = TrajectoryAnalyzer()
        lane_analyzer = LaneAnalyzer()
        chaos_scorer  = ChaosScorer()
        predictor     = RiskPredictor()
        alert_engine  = AlertEngine()

        frame_w = 1280

        detections = [
            make_detection(200, 400, "car"),
            make_detection(800, 400, "truck"),
        ]

        tracked = tracker.update(detections)
        motion_map, risk_map = {}, {}

        for obj_id, obj in tracked.items():
            traj   = obj["trajectory"]
            motion = traj_analyzer.analyze(traj)
            lane   = lane_analyzer.analyze_lane_behavior(traj, obj["label"], frame_w)
            risk   = predictor.predict(obj_id, motion, tracked, (720, 1280, 3))
            alert_engine.process(obj_id, obj["label"], risk, lane)

            motion_map[obj_id] = motion
            risk_map[obj_id]   = risk

        score, details = chaos_scorer.compute(list(tracked.keys()), motion_map, risk_map)

        assert isinstance(score, int)
        assert 0 <= score <= 100
        assert "n_objects" in details

    def test_multi_frame_tracker_stability(self):
        tracker = CentroidTracker()
        ids_seen = set()
        for frame in range(10):
            det = [make_detection(100 + frame * 5, 300)]
            tracked = tracker.update(det)
            for oid in tracked:
                ids_seen.add(oid)

        # Only 1 unique ID should have been assigned throughout
        assert len(ids_seen) == 1
