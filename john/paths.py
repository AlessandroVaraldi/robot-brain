"""Where things are, for the system and for the scripts in tests/ and eval/."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FRAMES = DATA / "frames"          # saved camera frames, for tests and the fake bridge
MODELS = DATA / "models"          # face detection and recognition (ONNX)
LFW = DATA / "lfw"                # Labeled Faces in the Wild, to check face_compare
LIVE_LOGS = DATA / "live_logs"    # one file per live session (one_mind.py --log)
MEMORY_DB = ROOT / "memory" / "memory.sqlite"   # the robot memory: not in git
DEBUG_MEMORY = ROOT / "memory" / "debug"       # a separate memory, keys in the clear (--debug)
FACE_KEYS = ROOT / "pi"                         # face_key: faces as keys (import path)
PROMPTS = ROOT / "john" / "prompts"   # every instruction to a model, one file each


def prompt(file: str, /, **fill) -> str:
    """The text of john/prompts/<file>.txt, with ${key} filled from `fill`."""
    from string import Template
    text = (PROMPTS / f"{file}.txt").read_text(encoding="utf-8")
    text = text[:-1] if text.endswith("\n") else text
    return Template(text).substitute(**fill) if fill else text
