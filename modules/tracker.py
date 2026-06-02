import numpy as np
from collections import OrderedDict
from config.config import MAX_DISAPPEARED, MAX_DISTANCE, TRAJECTORY_LEN


class CentroidTracker:
    def __init__(self):
        self.next_id     = 0
        self.objects     = OrderedDict()
        self.disappeared = OrderedDict()
        self.labels      = OrderedDict()
        self.trajectories = OrderedDict()

    def register(self, cx, cy, label):
        self.objects[self.next_id]      = (cx, cy)
        self.disappeared[self.next_id]  = 0
        self.labels[self.next_id]       = label
        self.trajectories[self.next_id] = [(cx, cy)]
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]
        del self.labels[obj_id]
        del self.trajectories[obj_id]

    def update(self, detections):
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > MAX_DISAPPEARED:
                    self.deregister(obj_id)
            return {}

        new_centroids = np.array([[d["cx"], d["cy"]] for d in detections])

        if len(self.objects) == 0:
            for det in detections:
                self.register(det["cx"], det["cy"], det["label"])
        else:
            existing_ids      = list(self.objects.keys())
            existing_centroids = np.array(list(self.objects.values()))
            D    = self._distance_matrix(existing_centroids, new_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > MAX_DISTANCE:
                    continue
                obj_id = existing_ids[row]
                cx, cy = new_centroids[col]
                self.objects[obj_id]      = (cx, cy)
                self.disappeared[obj_id]  = 0
                self.labels[obj_id]       = detections[col]["label"]
                self.trajectories[obj_id].append((cx, cy))
                if len(self.trajectories[obj_id]) > TRAJECTORY_LEN:
                    self.trajectories[obj_id].pop(0)
                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(len(existing_ids))) - used_rows:
                obj_id = existing_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > MAX_DISAPPEARED:
                    self.deregister(obj_id)

            for col in set(range(len(detections))) - used_cols:
                det = detections[col]
                self.register(det["cx"], det["cy"], det["label"])

        return {
            obj_id: {
                "id":         obj_id,
                "cx":         cx,
                "cy":         cy,
                "label":      self.labels[obj_id],
                "trajectory": self.trajectories[obj_id],
            }
            for obj_id, (cx, cy) in self.objects.items()
        }

    def _distance_matrix(self, A, B):
        A    = A.astype(np.float32)
        B    = B.astype(np.float32)
        diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))
