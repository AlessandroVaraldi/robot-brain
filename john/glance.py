"""Tricks that need the picture, not just the words on the board.

  count     "HOW MANY OF US ARE THERE?": faces counted by the face detector
            (the one face_compare.py uses), not guessed by a language model
  drawing   "GUESS WHAT I DREW": the vision model is asked one narrow question
            about the board - what is drawn on it - apart from its usual job

Each returns a tricks.Fact for the reply, or None.  Without OpenCV or the face
model, counting is off; the drawing needs only the vision model.
"""

from __future__ import annotations

import base64
import re

from paths import prompt
from tricks import Fact

COUNT_Q = re.compile(r"\bhow\s+many\s+(?:of\s+us|people|persons|faces)\b", re.I)
HERE_NOW = re.compile(r"\b(of\s+us|see|here|there|in\s+front|in\s+the\s+room|around|can\s+you)\b",
                      re.I)
NOT_NOW = re.compile(r"\b(met|meet|today|yesterday|before|know|remember)\b", re.I)
DRAW_Q = re.compile(r"\b(?:guess\s+(?:what\s+)?(?:i\s+)?(?:drew|draw|drawing|this|the\s+drawing)|"
                    r"what\s+(?:did\s+i|am\s+i|have\s+i)\s+(?:draw|drew|drawn|drawing)|"
                    r"what\s+is\s+(?:this|my|the)\s+drawing|what\s+did\s+i\s+draw)\b", re.I)
DRAW_PROMPT = prompt("drawing")
NUMBERS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
_detector = None


def count_faces(image_b64: str) -> int | None:
    """Faces in the frame, or None when the detector is not available."""
    global _detector
    try:
        import cv2
        import numpy as np
        from paths import MODELS
    except ImportError:
        return None
    model = MODELS / "face_detection_yunet_2023mar.onnx"
    if not model.exists():
        return None
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.6, 0.3, 50)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(image_b64), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    _detector.setInputSize((img.shape[1], img.shape[0]))
    _, faces = _detector.detect(img)
    return 0 if faces is None else len(faces)


def asks_count(writing: str) -> bool:
    return bool(COUNT_Q.search(writing) and HERE_NOW.search(writing)
                and not NOT_NOW.search(writing))


def count_fact(n: int) -> Fact:
    if n == 0:
        return Fact("Right now you cannot make out any face in front of you.", "")
    word = NUMBERS[n] if n < len(NUMBERS) else str(n)
    # Impersonal: copied word for word, "you count 2 faces" sounded like theirs.
    return Fact(f"Faces turned towards you right now: {n}.",
                f"{n} / {word}", f"I can see {word} of you.")


def a(noun: str) -> str:
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


def drawing_fact(guess: str, other: str) -> Fact:
    guess, other = guess.strip().lower(), other.strip().lower()
    blank = ("nothing", "none", "blank", "empty", "no drawing", "text", "writing", "n/a")
    if guess in blank or guess.startswith("nothing"):
        guess = ""
    if other in blank:
        other = ""
    if not guess:
        return Fact("You cannot make out a drawing on the board.", "")
    maybe = f" - or perhaps {a(other)}" if other and other != guess else ""
    return Fact(f"You looked at the drawing: it looks like {a(guess)}{maybe}.", guess,
                f"It looks like {a(guess)} to me.")


def glance(writing: str, image_b64: str | None, ask_eye) -> Fact | None:
    """ask_eye(prompt, image_b64) -> dict: one question to the vision model."""
    if not image_b64 or not writing:
        return None
    if asks_count(writing):
        n = count_faces(image_b64)
        return None if n is None else count_fact(n)
    if DRAW_Q.search(writing):
        out = ask_eye(DRAW_PROMPT, image_b64)
        return drawing_fact(str(out.get("guess", "") or ""), str(out.get("other", "") or ""))
    return None
