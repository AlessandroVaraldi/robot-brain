# John

The mind of John, a small humanoid robot in a university lab. People talk to
it by writing on a whiteboard; it answers aloud, can make a few hand gestures,
and, if they agree, remembers them by their face across days.

Everything runs locally: two models served by [Ollama](https://ollama.com) on a
GPU machine, which talks over HTTP to a bridge on the robot's Raspberry Pi
(camera, speech, hand).

## How it works

```
robot bridge (Pi)          GPU machine
─────────────────          ───────────────────────────────────────────────────
/status, /snapshot.jpg ──► gate ──► encounters ──► eye ──► reasoning ──► check
/speak, /arm          ◄───────────────────────────────────────────────── action
                                        │
                                        ▼
                                     memory (SQLite)
```

- **Gate** (`one_mind.py`). No model runs unless a face is in view or the scene
  changed; an empty room costs nothing.
- **Encounters** (`encounters.py`). Who the robot is talking to. A new name counts
  only once it has been read on two frames; then the robot asks "May I remember
  your face?", and only a yes keeps anything beyond the encounter. A face it
  remembers is greeted by name and gets what the robot knows about them, and
  nobody else does; "FORGET MY FACE" forgets them, and "UPDATE MY NAME TO ..."
  changes their name once they confirm it.
- **Faces** (`pi/face_key/`). A face is kept only as a lock that the same face
  opens, with a key held by a chip on the robot: no faces are stored, and no
  names in the clear. The name is something known about a face, not who the
  person is. An admin can list the names and delete people by name.
- **Reading in full**. A board the eye has not finished reading ("...", a
  trailing comma, a phrase that stops on "your" or "I'm") is not answered: the
  next frame reads it whole. A board read half and then whole is answered as
  one: the reply is told which part is new.
- **Eye** (`two_stage.py`, `qwen3-vl:4b-instruct`). Reads the whiteboard, says
  whether someone is engaging the robot, and routes: a named gesture goes
  straight to the hand, anything else to the reasoning stage.
- **Reasoning** (`two_stage.py`, `qwen3:14b`). One short spoken reply, from what
  the person wrote, what the robot knows about them, and — only when the writing
  is about the robot — what it knows about itself. It can also pick a gesture
  when asked to move.
- **Check**. Before a reply to a question is said, a second question to the
  same model asks whether it states something the robot cannot know (about the
  person, a shared past, the world around it). An invented fact about the
  person is replaced at once by "You haven't told me that."; for other
  questions the reply is asked for once more, and replaced only if it invents
  again. Other people the robot knows are acknowledged by name, but nothing
  about them is shared. Writing that tries to rename John never reaches the
  model, and a reply that recites its instructions is not said.
- **Memory** (`memory_store.py`). When an encounter with someone it remembers
  ends, the robot writes notes about them, encrypted under the key their face
  gives. With nobody in front of it, it takes its time and reasons
  over which old facts are no longer true; if someone arrives it stops and tries
  again later, and a person who comes back before their notes are written gets
  a fast write-up at once. Sensitive details and anything addressed to the robot
  are never kept.

## Tricks

- **Calendar**: the day of the week of any date, someone's age, and the days left to a birthday.
- **Arithmetic**: long products, powers, square roots, primes and prime factors, worked out exactly.
- **Memory**: a list given to it, said back in order, backwards, sorted or one item at a time.
- **Anagrams**: a word, or the person's name, rearranged into real English words.
- **Counting**: how many people are in front of it, by the faces it sees.
- **Drawings**: a guess at what someone drew on the whiteboard.
- **Blindfold chess**: a whole game against Stockfish, moves written on the board, no chessboard seen.

## Layout

```
john/       the system
  one_mind.py       live loop: gate, bridge I/O, logging
  two_stage.py      eye, reasoning, checks, record and notes
  encounters.py     who the robot is talking to, archiving
  tricks.py         calendar, arithmetic, memory, anagrams
  glance.py         tricks that look: counting faces, guessing a drawing
  chess_table.py    blindfold chess
  memory_store.py   people and encounters (SQLite)
  face_compare.py   optional: is this the same face as a moment ago?
  self_memory.py    operator tool: facts the lab tells the robot about itself
  inspect_memory.py debug tool: everything a debug memory keeps, decrypted
  read_live.py      print a live session log
  prompts/          every instruction given to a model, one text file each
  paths.py          where data, prompts and memory live
pi/face_key/  faces as keys: locks, the key chip and its firmware, the admin tool
tests/      deterministic tests, no models
eval/       measurements with the real models, robot not involved
data/       frames, face models, logs (not in git, see data/README.md)
memory/     the robot's memory (not in git)
```

## Running

Requirements: Python 3.10+, Ollama with the two models, and for some tools
`numpy`, `opencv-python` and `Pillow`. Anagrams need `wordfreq`, and are
better with a word list in `data/words.txt`. Chess needs `python-chess` and a
Stockfish binary (in `data/stockfish/`, on the PATH, or at `ONE_STOCKFISH`).
`pi/face_key` needs `bchlib` and `cryptography`.

```
ollama pull qwen3-vl:4b-instruct
ollama pull qwen3:14b

python3 john/one_mind.py --dry-run --max-seconds 60   # decide, send nothing to the robot
python3 john/one_mind.py --face-check                 # live, remembering faces
python3 john/read_live.py                             # what happened in the last session
```

The bridge address is `BRIDGE` in `one_mind.py`. Use `--dry-run` for anything
that is not meant to move the robot.

People are remembered only with `--face-check`, and only once there is an
admin key: `python3 -m face_key.admin keygen <file>` (from `pi/`, on a machine
that is not the robot) prints a public key to put in `memory/admin.pub`.
`--chip /dev/ttyACM0` uses the key chip; without it a software stand-in is used,
which protects nothing.

`--debug` runs on a memory of its own (`memory/debug/`), makes a test admin key
there, writes every key in the clear and logs what is otherwise only kept
encrypted; `python3 john/inspect_memory.py` then shows all of it. Never point it
at the memory of real people.

```
python3 john/one_mind.py --dry-run --face-check --debug
python3 john/inspect_memory.py
python3 eval/fake_bridge_run.py --face-check --debug --reappear-seconds 5   # no robot needed
```

Each behaviour above can be switched off for comparison with an environment
variable set to `0`: `ONE_CHECK`, `ONE_CHECK_WIDE`, `ONE_CHECK_RETRY`,
`ONE_SPEAKER_LINE`, `ONE_OTHERS_LINE`, `ONE_CAREFUL_SUMMARY`, `ONE_CLOCK`,
`ONE_BODY_IN_SELF`, `ONE_TEXT_GESTURES`, `ONE_TRICKS`, `ONE_CHESS`,
`ONE_WAIT_UNFINISHED`, `ONE_CONTINUATION`, `ONE_RENAME_GUARD`, `ONE_RECITAL_GUARD`. `ONE_TEXT_MODEL` picks another
reasoning model.

## Testing

```
for t in tests/test_*.py; do python3 $t; done      # no models, seconds

python3 eval/test_multiday.py        # several simulated days, real models
python3 eval/fake_bridge_run.py      # the live loop against a fake bridge
python3 eval/robust.py --model qwen3:14b --body --clock --check --tag now
python3 eval/robust_report.py eval/suite_out/robust_now_*.json
python3 eval/notes.py                # end-of-encounter notes
python3 eval/face_key_lfw.py         # face keys on LFW
python3 eval/greet_eval.py           # the greeting for a recognised face
```

`robust.py` runs the answer suite (`eval/suite.py`) under four wordings of the
same system prompt, and a case counts as solid only if it holds under all of
them: with small models one added sentence can flip unrelated answers, so every
change is judged on the whole suite, not on the case it was made for. The checks
are keyword-based and coarse on purpose; the raw outputs are saved to
`eval/suite_out/` to be read.
