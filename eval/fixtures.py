"""Test inputs made from saved frames: what the eye would see."""

import base64
import io
import os
import sys
import tempfile
from pathlib import Path

from paths import FACE_KEYS, FRAMES


def encode(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def board_on_person(text, frac=0.42, size=(640, 480)):
    """Paste the board onto a frame where someone is facing the camera.

    Someone must be in the frame: on an empty scene the eye rightly reports
    that nobody is engaging it, and nothing gets routed.
    """
    from PIL import Image, ImageDraw, ImageFont
    scene = Image.open(FRAMES / "pose1.jpg").convert("RGB").resize(size)
    w = int(size[0] * frac)
    h = int(w * 0.62)
    b = Image.new("RGB", (w, h), (246, 246, 243))
    d = ImageDraw.Draw(b)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(11, w // 14))
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([2, 2, w - 3, h - 3], outline=(80, 80, 80), width=3)
    words, line, lines = text.split(), "", []
    for word in words:
        t = (line + " " + word).strip()
        if d.textlength(t, font=font) > w - 14:
            lines.append(line)
            line = word
        else:
            line = t
    lines.append(line)
    step_y = w // 12
    y = h // 2 - step_y * len(lines) // 2
    for ln in lines:
        d.text(((w - d.textlength(ln, font=font)) / 2, y), ln, fill=(20, 20, 30), font=font)
        y += step_y
    scene.paste(b, (int(size[0] * 0.04), int(size[1] * 0.46)))
    buf = io.BytesIO()
    scene.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


class Faces:
    """Face keys (pi/face_key, with the software chip) and made-up faces: a
    person is a random vector, a frame of them a noisy copy of it, as far
    apart as two photos of one person are for SFace."""

    def __init__(self, seed=0):
        import numpy as np
        sys.path.insert(0, str(FACE_KEYS))
        from face_key.chip import SoftChip
        from face_key.store import FaceKeys, admin_keypair
        self.np, self.rng, self.people = np, np.random.default_rng(seed), {}
        self.keys = FaceKeys(SoftChip(key=os.urandom(32), burst=10**6), admin_keypair()[1],
                             Path(tempfile.mkdtemp()) / "faces.sqlite")

    def frame(self, who):
        np = self.np
        face = self.people.setdefault(who, self.rng.standard_normal(128))
        v = face / np.linalg.norm(face) + 0.8 * self.rng.standard_normal(128) / np.sqrt(128)
        return v / np.linalg.norm(v)

    @staticmethod
    def same(a, b):
        """The tracker's same-face check, for these faces."""
        return float(a @ b) > 0.3

    def memory_key(self, who):
        """The key to `who`'s memory, as their face would give it (None if forgotten)."""
        found = self.keys.recognise([self.frame(who), self.frame(who)])
        return found and (found.id, found.memory_key)
