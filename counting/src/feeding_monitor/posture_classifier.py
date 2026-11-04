"""Wraps the pretrained RandomForest posture classifier (see
posture_features.py's module docstring) behind the same simple interface
as feeding_classifier.classify(), so scripts/compare_classifiers.py can run
both over the same detections.

predict_batch() exists because sklearn's per-call Python overhead makes
predicting one row at a time (naively looping .predict([row])) dominate
runtime once there are more than a few thousand detections -- always prefer
it over calling predict()/predict_proba() in a loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .pose_tracker import Detection
from .posture_features import FEATURE_NAMES, Zone, extract_features


@dataclass
class MLPostureModel:
    model: object
    feature_names: list[str]

    @classmethod
    def load(cls, path: str | Path) -> "MLPostureModel":
        bundle = joblib.load(path)
        return cls(model=bundle["model"], feature_names=bundle["feature_names"])

    def _feature_frame(self, dets: list[Detection], zones: list[Zone]) -> pd.DataFrame:
        rows = [dict(zip(FEATURE_NAMES, extract_features(det, zones).as_row())) for det in dets]
        return pd.DataFrame(rows, columns=self.feature_names)

    def predict(self, det: Detection, zones: list[Zone], threshold: float = 0.5) -> bool:
        return bool(self.predict_batch([det], zones, threshold)[0])

    def predict_proba(self, det: Detection, zones: list[Zone]) -> float:
        return self.predict_proba_batch([det], zones)[0]

    def predict_batch(self, dets: list[Detection], zones: list[Zone], threshold: float = 0.5) -> list[bool]:
        """threshold < 0.5 is more lenient (calls "eating" more readily) --
        the model's own .predict() is a fixed 0.5 cut, so a lower threshold
        goes through predict_proba_batch instead."""
        if not dets:
            return []
        if threshold == 0.5:
            X = self._feature_frame(dets, zones)
            return [bool(v) for v in self.model.predict(X)]
        return [p >= threshold for p in self.predict_proba_batch(dets, zones)]

    def predict_proba_batch(self, dets: list[Detection], zones: list[Zone]) -> list[float]:
        """Probability of the 'eating' class per detection, for thresholding/
        diagnostics beyond the model's default 0.5 cut."""
        if not dets:
            return []
        X = self._feature_frame(dets, zones)
        proba = self.model.predict_proba(X)
        classes = list(self.model.classes_)
        idx = classes.index(1) if 1 in classes else len(classes) - 1
        return [float(p[idx]) for p in proba]
