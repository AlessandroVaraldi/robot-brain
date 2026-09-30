#!/usr/bin/env python3
"""chess_table.py: a game from start to result, with a stand-in engine that
plays scripted moves (or the first legal one) and reports a set evaluation.
Needs python-chess; no Stockfish, no models."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "john"))
import chess  # noqa: E402
import chess_table as ct  # noqa: E402

RESULTS = []


def check(ok, label):
    RESULTS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")


class FakeEngine:
    def __init__(self, script=(), score=0):
        self.script, self.eval, self.closed = list(script), score, False

    def best(self, board):
        while self.script:
            move = chess.Move.from_uci(self.script.pop(0))
            if move in board.legal_moves:
                return move
        return sorted(board.legal_moves, key=lambda m: m.uci())[0]

    def score(self, board, colour):
        return self.eval

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def table(script=(), score=0):
    eng = FakeEngine(script, score)
    clock = Clock()
    return ct.ChessTable(make_engine=lambda: eng, clock=clock), eng, clock


def main():
    for w in ("LET'S PLAY CHESS", "DO YOU WANT TO PLAY CHESS?", "CHESS?", "SHALL WE PLAY A GAME OF CHESS"):
        check(ct.ChessTable(make_engine=lambda: None).starts(w), f"starts: {w!r}")
    for w in ("I LIKE CHESS", "WHAT IS CHESS?", "HELLO", "LET'S PLAY"):
        check(not ct.ChessTable(make_engine=lambda: None).starts(w), f"does not start: {w!r}")

    t = ct.ChessTable(make_engine=lambda: None)
    check("not installed" in t.handle("LET'S PLAY CHESS") and not t.active(), "no engine: says so, no game")

    # --- they take black: John opens
    t, eng, clock = table(script=["e2e4"])
    check(t.handle("LET'S PLAY CHESS").endswith("white or black?"), "asks for a colour")
    say = t.handle("BLACK")
    check(say == "I have white. I play pawn from e2 to e4. Your move.", f"John opens: {say!r}")

    # --- they take white: moves in the notations, repeats, illegal and non-moves
    t, eng, clock = table(script=["e7e5", "b8c6", "g8f6"])
    t.handle("CHESS?")
    check(t.handle("WHITE") == "You have white. Your move.", "white: waits")
    say = t.handle("e2 e4")
    check(say == "You played pawn from e2 to e4. I play pawn from e7 to e5. Your move.", f"from-to: {say!r}")
    check(t.handle("e2 e4") == "", "the same board again: nothing said")
    say = t.handle("NF3")
    check(say.startswith("You played knight from g1 to f3."), f"all-caps, one reading: {say!r}")
    say = t.handle("e2 e5")
    check(say == "There is no piece of yours on e2. Your move." and t.state == "play",
          f"illegal: said, the game goes on: {say!r}")
    say = t.handle("HELLO JOHN")
    check(say == "That is not a move. Do you want to end the game?" and t.state == "quit?", f"not a move: {say!r}")
    say = t.handle("NO")
    check(say.startswith("Then we play on. My last move was knight from b8 to c6") and t.state == "play",
          f"not ending: {say!r}")
    say = t.handle("HELLO JOHN")
    check(say == "That is not a move. Do you want to end the game?", f"not a move: {say!r}")
    say = t.handle("Bc4")
    check(say.startswith("You played bishop from f1 to c4.") and t.state == "play",
          f"a legal move answers the question too: {say!r}")
    n = len(t.board.move_stack)
    check(t.handle("UNDO LAST MOVE") == "Undone. Your move again." and len(t.board.move_stack) == n - 2,
          "undo: their move and John's answer")
    t.handle("hello")
    check(t.handle("YES") == "All right, the game is over." and not t.active() and eng.closed,
          "ending: game over, engine closed")

    # --- the three kinds of wrong writing, from the starting position
    t, eng, clock = table()
    t.handle("CHESS?"), t.handle("WHITE")
    for w, want in (("e7 e5", "The piece on e7 is not yours. Your move."),
                    ("e2 e5", "Pawn from e2 to e5 is not legal here. Your move."),
                    ("g1 g3", "Knight from g1 to g3 is not legal here. Your move."),
                    ("Qh5", "Qh5 is not legal here. Your move."),
                    ("O-O", "Castling is not legal here. Your move.")):
        t.misses = 0                       # one at a time: the run of them is tested below
        say = t.handle(w)
        check(say == want and t.state == "play", f"illegal {w!r}: {say!r}")
    for w in ("e9", "Nf0", "Kz4", "e2 e9", "E2 I4"):
        t.misses = 0
        say = t.handle(w)
        check(say == "I can't read that move. Please write it again." and t.state == "play",
              f"unreadable {w!r}: {say!r}")
    for w in ("HELLO", "WHAT IS MY NAME?", "2+2", "ROOM 42", "I LIKE IT", "BE", "THANKS"):
        say = t.handle(w)
        check(say == "That is not a move. Do you want to end the game?", f"not a move {w!r}: {say!r}")
        t.handle("NO")                     # back to the game for the next one
    check(len(t.board.move_stack) == 0, "nothing wrong was played")

    # --- a run of moves that cannot be played or read: the third asks about ending
    t, eng, clock = table(script=["e7e5"])
    t.handle("CHESS?"), t.handle("WHITE")
    t.handle("e2 e5"), t.handle("e9")
    say = t.handle("Qh5")
    check(say == "Qh5 is not legal here. Do you want to end the game?" and t.state == "quit?",
          f"third in a row: asks: {say!r}")
    t.handle("NO")
    say = t.handle("e9")
    check(say == "I can't read that move. Please write it again." and t.state == "play",
          f"after NO the count starts again: {say!r}")
    t.handle("e2 e5")
    t.handle("e2 e4")                      # a legal move in between
    say = t.handle("e2 e3")                # e2 is empty now: illegal, the first of a new run
    check(t.state == "play" and "end the game" not in say, f"a legal move starts the count again: {say!r}")
    t.handle("e9")
    say = t.handle("Nf0")
    check(say == "I can't read that move. Do you want to end the game?", f"unreadable, third: {say!r}")

    # --- ambiguous standard notation: two knights can reach d2
    t, eng, clock = table()
    t.handle("CHESS?"), t.handle("WHITE")
    t.board = chess.Board("4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1")
    say = t.handle("Nd2")
    check("two ways" in say and t.state == "play", f"ambiguous: asks to rewrite: {say!r}")
    check(t.handle("f3 d2").startswith("You played knight from f3 to d2."), "from-to settles it")

    # --- capitals matter in standard notation
    t, eng, clock = table()
    t.handle("CHESS?"), t.handle("WHITE")
    t.board = chess.Board("4k3/8/8/2p5/1P6/8/8/4K3 w - - 0 1")
    check(t.handle("bxc5").startswith("You played pawn from b4 takes on c5"), "bxc5: the pawn")

    # --- draws offered
    t, eng, clock = table(score=20)
    t.handle("CHESS?"), t.handle("WHITE"), t.handle("e2 e4")
    say = t.handle("DRAW?")
    check(say.startswith("No, I'd rather play on.") and t.state == "play", f"level game: declined: {say!r}")
    eng.eval = -500
    say = t.handle("I OFFER A DRAW")
    check(say == "I accept the draw. Shall we play again?" and t.state == "rematch?", f"lost game: accepted: {say!r}")

    # --- resigning, rematch with colours swapped on confirmation
    t, eng, clock = table()
    t.handle("CHESS?"), t.handle("WHITE"), t.handle("e2 e4")
    check(t.handle("I RESIGN") == "You resign, so I win. Do you want a rematch?", "resign: John won, offers a rematch")
    check(t.handle("YES") == "Shall we swap colours? You would have black.", "rematch: asks to swap")
    say = t.handle("YES")
    check(say.startswith("I have white. I play") and t.robot_white, f"swapped: John has white: {say!r}")
    t.handle("I GIVE UP")
    check(t.handle("NO") == "All right. Thank you for the game." and not t.active(), "no rematch")

    # --- John mates (fool's mate)
    t, eng, clock = table(script=["e7e5", "d8h4"])
    t.handle("CHESS?"), t.handle("WHITE"), t.handle("f2 f3")
    say = t.handle("g2 g4")
    check(say.endswith("I play queen from d8 to h4, checkmate. I win. Do you want a rematch?"),
          f"John mates, offers a rematch: {say!r}")

    # --- they mate (scholar's mate): John asks for its rematch
    t, eng, clock = table(script=["e7e5", "b8c6", "g8f6"])
    t.handle("CHESS?"), t.handle("WHITE")
    for m in ("e2 e4", "f1 c4", "d1 h5"):
        t.handle(m)
    say = t.handle("Qxf7#")
    check(say == "You played queen from h5 takes on f7, checkmate. You win, well played. "
                 "Will you give me a rematch?", f"they mate, John asks for a rematch: {say!r}")
    check(t.handle("WHAT IS MY NAME?") is None and not t.active(), "after the game, other writing is not the table's")

    # --- check said aloud
    t, eng, clock = table(script=["e7e5", "f8b4"])
    t.handle("CHESS?"), t.handle("WHITE"), t.handle("e2 e4")
    say = t.handle("d2 d4")
    check("bishop from f8 to b4, check." in say, f"check said: {say!r}")

    # --- a long silence
    t, eng, clock = table()
    t.handle("CHESS?"), t.handle("WHITE")
    clock.t += ct.ASK_IDLE_S + 1
    check(t.idle().startswith("Are we still playing?") and t.idle() == "", "asks once after a pause")
    clock.t += ct.DROP_IDLE_S
    check(not t.active() and eng.closed, "dropped after a long silence, engine closed")

    # --- through two_stage.play_chess and Mind: nothing of the game is kept
    import two_stage as T
    from encounters import EncounterTracker, Mind
    from memory_store import MemoryStore
    T.TABLE, eng, clock = table(script=["e7e5"])
    check(T.play_chess({"writing": "WHAT IS MY NAME?", "engaged": True}) is None, "no game: not the table's")
    act = T.play_chess({"writing": "LET'S PLAY CHESS", "engaged": True})
    check(act["chess"] and act["text"].endswith("white or black?"), "a game starts through step")
    check(T.play_chess({"writing": "", "engaged": True}) == {"action": "none", "chess": True},
          "during a game, nothing written: John says nothing")

    frames = iter(["WHITE", "e2 e4", "MY NAME IS ALESSANDRO"])

    def decide(image, record, memory, is_new=None):
        seen = {"writing": next(frames), "engaged": True}
        return T.play_chess(seen, is_new) or {"action": "speak", "text": "normal"}, seen

    now = [0.0]
    store = MemoryStore(Path(tempfile.mkdtemp()) / "m.sqlite")
    mind = Mind(store, EncounterTracker(timeout_s=5), decide, lambda n, o, t: ("", ""),
                lambda s, r, m: "", clock=lambda: now[0])
    for _ in range(3):
        now[0] += 1
        act = mind.on_frame(True, "img")
    enc = mind.tracker.active
    check(act.get("chess") and "not a move" in act["text"], "a name written mid-game is not handled")
    check(enc.turns == [] and not enc.record.get("person_name") and store.people() == 0,
          "no turns, no name, nothing stored")

    # after a game, other writing reaches the normal reply once, not as a repeat
    T.TABLE, eng, clock = table()
    for w in ("CHESS?", "WHITE", "e2 e4", "I RESIGN"):
        T.TABLE.handle(w)
    marks = []
    T.look = lambda img: {"writing": "WHAT IS YOUR NAME?", "engaged": True, "route": "answer"}
    T.answer_act = lambda *a, **k: ("John.", "")
    act, _ = T.step("img", {"person_name": "", "person_notes": "", "pending": ""},
                    is_new=lambda s: marks.append(1) or len(marks) == 1)
    check(act.get("text") == "John." and len(marks) == 1,
          f"the table lets go: answered, the repeat check asked once: {act!r}")

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
