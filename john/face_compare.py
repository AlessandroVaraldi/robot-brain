#!/usr/bin/env python3
"""Is the face in front of the robot the same one as a moment ago?

This answers "same or different", nothing else: no names, no identities.
Vectors live in RAM, inside one encounter (EncounterTracker.face_ref), and are
never written anywhere.

Only a large, frontal face is compared: with the head turned, the same person
scores far below the threshold.  A face that does not qualify gives None, and
the tracker then falls back on time alone.

The threshold is OpenCV's default for SFace; check it on the robot's camera
with eval/validate_faces.py.
"""

from __future__ import annotations


import cv2
import numpy as np

from paths import MODELS  # noqa: E402


def yaw_of(row):
    """Offset of the nose from the eyes' midpoint along the line of the eyes,
    in eye-distances.  0 is frontal.  Along the eyes, not across the picture:
    measured across, a head tilted 15 degrees and looking straight at the
    camera was taken for a head turned away, and half the faces were lost."""
    re_, le, no = (np.array(row[4 + 2 * i:6 + 2 * i], dtype=float) for i in range(3))
    eyes = le - re_
    d = float(np.linalg.norm(eyes))
    return float(abs((no - (re_ + le) / 2) @ eyes) / d ** 2) if d else 9.9


class FaceComparer:
    def __init__(self, threshold: float = 0.363, min_width: float = 60.0,
                 max_yaw: float = 0.15):
        self.threshold = threshold
        self.min_width = min_width
        self.max_yaw = max_yaw
        self.det = cv2.FaceDetectorYN.create(
            str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6, 0.3, 20)
        self.rec = cv2.FaceRecognizerSF.create(
            str(MODELS / "face_recognition_sface_2021dec.onnx"), "")

    def analyse(self, jpeg: bytes) -> dict:
        """The largest face: its width, rotation, and its vector if usable."""
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"faces": 0, "vec": None, "why": "unreadable image"}
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        if faces is None or len(faces) == 0:
            return {"faces": 0, "vec": None, "why": "no face"}
        row = max(faces, key=lambda r: r[2])
        out = {"faces": len(faces), "width": float(row[2]), "yaw": yaw_of(row), "vec": None}
        if out["width"] < self.min_width:
            out["why"] = f"small face ({out['width']:.0f}px)"
        elif out["yaw"] > self.max_yaw:
            out["why"] = f"head turned ({out['yaw']:.2f})"
        else:
            out["vec"] = self.rec.feature(self.rec.alignCrop(img, row)).copy()
            out["why"] = ""
        return out

    def embed(self, jpeg: bytes):
        return self.analyse(jpeg)["vec"]

    def score(self, a, b) -> float:
        return float(self.rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))

    def same(self, a, b) -> bool:
        return self.score(a, b) >= self.threshold
