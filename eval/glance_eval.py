#!/usr/bin/env python3
"""The tricks that look (glance.py), through the whole pipeline: the eye reads
the board, the reply stage runs the trick, John answers.  The pictures are
made here: doodles drawn in code on a board held by someone, and groups of
faces from LFW with a board asking how many there are.  Nothing is sent to
the robot.

    python3 eval/glance_eval.py            # 3 repetitions
    python3 eval/glance_eval.py --n 1
"""

import argparse
import base64
import io
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "john"))
import glance  # noqa: E402
import suite  # noqa: E402
import tricks  # noqa: E402
import two_stage as T  # noqa: E402
from paths import FRAMES, LFW  # noqa: E402

W, H = 300, 200          # the drawing area on the board


def house(d):
    d.rectangle([90, 90, 210, 180], outline=0, width=4)
    d.line([80, 92, 150, 30, 220, 92], fill=0, width=4)
    d.rectangle([135, 130, 165, 180], outline=0, width=4)


def sun(d):
    d.ellipse([115, 65, 185, 135], outline=0, width=4)
    for k in range(8):
        a = k * math.pi / 4
        d.line([150 + 45 * math.cos(a), 100 + 45 * math.sin(a),
                150 + 75 * math.cos(a), 100 + 75 * math.sin(a)], fill=0, width=4)


def tree(d):
    d.rectangle([138, 120, 162, 190], outline=0, width=4)
    d.ellipse([90, 20, 210, 130], outline=0, width=4)


def smiley(d):
    d.ellipse([90, 30, 210, 170], outline=0, width=4)
    d.ellipse([120, 70, 132, 82], fill=0)
    d.ellipse([168, 70, 180, 82], fill=0)
    d.arc([115, 80, 185, 145], 20, 160, fill=0, width=4)


def fish(d):
    d.ellipse([70, 70, 200, 140], outline=0, width=4)
    d.polygon([200, 105, 250, 70, 250, 140], outline=0, width=4)
    d.ellipse([95, 92, 107, 104], fill=0)


def star(d):
    pts = []
    for k in range(10):
        r = 85 if k % 2 == 0 else 35
        a = -math.pi / 2 + k * math.pi / 5
        pts += [150 + r * math.cos(a), 105 + r * math.sin(a)]
    d.polygon(pts, outline=0, width=4)


def heart(d):
    d.arc([95, 40, 155, 100], 150, 360, fill=0, width=4)
    d.arc([145, 40, 205, 100], 180, 30, fill=0, width=4)
    d.line([99, 84, 150, 170, 201, 84], fill=0, width=4)


def flower(d):
    for k in range(6):
        a = k * math.pi / 3
        x, y = 150 + 32 * math.cos(a), 75 + 32 * math.sin(a)
        d.ellipse([x - 20, y - 20, x + 20, y + 20], outline=0, width=4)
    d.ellipse([135, 60, 165, 90], outline=0, width=4)
    d.line([150, 115, 150, 195], fill=0, width=4)


def umbrella(d):
    d.chord([60, 30, 240, 150], 180, 360, outline=0, width=4)
    d.line([150, 90, 150, 170], fill=0, width=4)
    d.arc([150, 150, 180, 190], 0, 180, fill=0, width=4)


def car(d):
    d.rectangle([60, 100, 240, 150], outline=0, width=4)
    d.line([100, 100, 125, 65, 185, 65, 205, 100], fill=0, width=4)
    d.ellipse([85, 135, 125, 175], outline=0, width=4)
    d.ellipse([175, 135, 215, 175], outline=0, width=4)


DRAWINGS = [(house, ("house", "home", "hut", "cabin", "building", "barn")), (sun, ("sun",)),
            (tree, ("tree",)), (smiley, ("smil", "face", "emoji", "happy")), (fish, ("fish",)),
            (star, ("star",)), (heart, ("heart",)), (flower, ("flower", "daisy")),
            (umbrella, ("umbrella",)), (car, ("car", "vehicle", "truck", "van", "bus"))]


def jpeg(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def board(text, doodle=None, size=(W + 40, H + 80)):
    from PIL import Image, ImageDraw, ImageFont
    b = Image.new("RGB", size, (246, 246, 243))
    d = ImageDraw.Draw(b)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    d.rectangle([2, 2, size[0] - 3, size[1] - 3], outline=(80, 80, 80), width=3)
    d.text(((size[0] - d.textlength(text, font=font)) / 2, 12), text, fill=(20, 20, 30), font=font)
    if doodle:
        art = Image.new("L", (W, H), 255)
        doodle(ImageDraw.Draw(art))
        b.paste(art.convert("RGB"), (20, 60))
    return b


def drawing_frame(doodle):
    from PIL import Image
    scene = Image.open(FRAMES / "pose1.jpg").convert("RGB").resize((640, 480))
    scene.paste(board("GUESS WHAT I DREW", doodle), (30, 150))
    return jpeg(scene)


def group_frame(k, rng):
    from PIL import Image
    people = [p for p in (LFW / "lfw").iterdir() if p.is_dir()]
    scene = Image.new("RGB", (640, 480), (190, 190, 185))
    size = 150
    x0 = (640 - k * size - (k - 1) * 8) // 2
    for i, p in enumerate(rng.sample(people, k)):
        face = Image.open(sorted(p.glob("*.jpg"))[0]).convert("RGB").crop((50, 40, 200, 210))
        scene.paste(face.resize((size, int(size * 1.13))), (x0 + i * (size + 8), 20))
    b = board("HOW MANY OF US ARE THERE?", size=(420, 90))
    scene.paste(b, (110, 330))
    return jpeg(scene)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--only", choices=("drawings", "groups"))
    a = ap.parse_args()
    T.warmup()
    rec, mem = suite.CTX["stranger"]
    if a.only != "groups":
        drawings(a, rec)
    if a.only != "drawings":
        groups(a, rec)


def drawings(a, rec):
    print("drawings (the eye's guess / the reply)")
    guessed = replied = total = 0
    for doodle, names in DRAWINGS:
        img = drawing_frame(doodle)
        g_ok = r_ok = 0
        outs = []
        for _ in range(a.n):
            eye = T.ask_eye(glance.DRAW_PROMPT, img)
            g_ok += any(n in str(eye.get("guess", "")).lower() for n in names)
            act, seen = T.step(img, dict(rec))
            said = (act.get("text") or act.get("action", "")).lower()
            r_ok += any(n in said for n in names)
            outs.append(f"{eye.get('guess')!r} / {said[:60]!r}")
        guessed, replied, total = guessed + g_ok, replied + r_ok, total + a.n
        print(f"  {doodle.__name__:9} eye {g_ok}/{a.n}  reply {r_ok}/{a.n}   {outs[0]}", flush=True)
    print(f"  total: eye {guessed}/{total}, reply {replied}/{total}")


def groups(a, rec):
    print("\ngroups (faces counted / the reply)")
    rng = random.Random(1)
    right = total = 0
    for k in (1, 2, 3, 4):
        for _ in range(2):                          # two different groups of each size
            img = group_frame(k, rng)
            n = glance.count_faces(img)
            for _ in range(a.n):
                act, seen = T.step(img, dict(rec))
                said = (act.get("text") or act.get("action", "")).lower()
                ok = n is not None and tricks.said(glance.count_fact(n), said)
                right, total = right + ok, total + 1
            print(f"  {k} faces: detector {n}   reply {said[:70]!r}", flush=True)
    print(f"  total: reply says what the detector counted {right}/{total}")


if __name__ == "__main__":
    main()
