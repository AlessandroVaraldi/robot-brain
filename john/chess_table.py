"""Blindfold chess: John plays a game written on the whiteboard, move by move.

Someone proposes a game, John asks for a colour, then plays: it repeats the
move it read, answers with its own in full words ("knight from g1 to f3"),
says "check" and announces the result, then offers a rematch.  During a game
it does nothing else, and nothing about the game is kept: the game belongs to
the table, not to the encounter - players look at the board, not the camera -
and after a long silence it is dropped.

Everything said here is a fixed sentence; no language model is involved.
The engine is Stockfish through python-chess, both optional: without them
John says it cannot play.
"""

from __future__ import annotations

import glob
import os
import random
import re
import shutil
import time

try:
    import chess
    import chess.engine
except ImportError:                    # python-chess is optional
    chess = None

ASK_IDLE_S = 600                       # no move for this long: ask once if we still play
DROP_IDLE_S = 1200                     # no move for this long: the game is dropped
DRAW_ACCEPT_CP = -300                  # accept a draw only when this bad for John
MAX_MISSES = 3                         # moves in a row that cannot be played or read:
                                       # then ask whether to end the game
MOVE_TIME_S = 1.0

START = re.compile(r"\b(play|game|match|let'?s|shall|want|up\s+for)\b.*\bchess\b|"
                   r"\bchess\b.*\b(play|game|match)\b|^\W*chess\W*$", re.I)
YES = re.compile(r"^\W*(yes|yeah|yep|sure|ok|okay|of\s+course|why\s+not|y)\b", re.I)
NO = re.compile(r"^\W*(no|nope|nah|not\s+now|n)\b", re.I)
STOP = re.compile(r"^\W*(stop|cancel|no\s+chess|never\s*mind)\b", re.I)
WHITE = re.compile(r"\bwhite\b", re.I)
BLACK = re.compile(r"\bblack\b", re.I)
EITHER = re.compile(r"\b(you\s+choose|either|any|random|whatever|don'?t\s+mind)\b", re.I)
UNDO = re.compile(r"\bundo\b", re.I)
RESIGN = re.compile(r"\b(i\s+resign|resign|i\s+give\s+up|give\s+up)\b", re.I)
DRAW = re.compile(r"\b(draw|offer\s+a\s+draw|remis)\b", re.I)
PIECES = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}


def engine_path() -> str | None:
    """ONE_STOCKFISH, else a Stockfish unpacked in data/stockfish, else the PATH."""
    if os.environ.get("ONE_STOCKFISH"):
        return os.environ["ONE_STOCKFISH"]
    from paths import DATA
    found = sorted(glob.glob(str(DATA / "stockfish" / "**" / "stockfish*"), recursive=True))
    found = [p for p in found if os.path.isfile(p) and os.access(p, os.X_OK)]
    return found[0] if found else shutil.which("stockfish")


class Stockfish:
    """Full strength, a fixed time per move, and small enough not to get in the
    way of the rest: two threads, a small hash table.  One process per game."""

    def __init__(self, path: str):
        self.e = chess.engine.SimpleEngine.popen_uci(path)
        self.e.configure({"Threads": 2, "Hash": 64, "Skill Level": 20})

    def best(self, board):
        return self.e.play(board, chess.engine.Limit(time=MOVE_TIME_S)).move

    def score(self, board, colour) -> int:
        """Centipawns from `colour`'s side; a mate counts as 10000."""
        info = self.e.analyse(board, chess.engine.Limit(time=0.3))
        return info["score"].pov(colour).score(mate_score=10000)

    def close(self):
        try:
            self.e.quit()
        except Exception:
            pass


def new_engine():
    if chess is None:
        return None
    path = engine_path()
    return Stockfish(path) if path else None


# ---------------------------------------------------------------- reading a move

# The shape of a move, squares allowed to be wrong ("e9", "Nf0", "Kz4"): most
# likely a move misread, not a message.  Real words never fit it.
MOVE_SHAPE = re.compile(r"[KQRBN]?[a-z]?\d?x?[a-z]\d{1,2}(=?[QRBN])?|[a-z]\d{1,2}-?x?[a-z]\d{1,2}[qrbn]?",
                        re.I)


def parse_move(board, text: str):
    """(move, why, what).  A legal move with why ""; or None and why:
      "illegal"     a well-formed move that cannot be played here; `what` says
                    why, to be said aloud ("There is no piece of yours on e2.")
      "unreadable"  shaped like a move but no such move exists ("e9", "Nf0")
      "ambiguous"   two legal readings
      "not a move"  anything else
    From-to squares in any case ("e2 e4", "E2-E4", "e7e8q"); standard notation
    as written ("Nf3", "bxc4", "O-O"), where capitals matter; an all-capitals
    "NF3" only if it has one legal reading."""
    t = re.sub(r"\b(check(mate)?|mate)\b", "", text or "", flags=re.I)
    t = re.sub(r"[!?+#.,;]+", " ", t).strip()
    if not t:
        return None, "not a move", ""
    m = re.fullmatch(r"([a-h])\s*([1-8])\s*(?:-|x|to|\s)?\s*([a-h])\s*([1-8])\s*(?:=|\s)?\s*([qrbn])?",
                     t, re.I)
    if m:
        a, b, c, d, promo = m.groups()
        uci = f"{a}{b}{c}{d}".lower() + (promo or "").lower()
        move = chess.Move.from_uci(uci)
        if not promo and board.piece_type_at(move.from_square) == chess.PAWN and d in "18":
            move = chess.Move.from_uci(uci + "q")          # promotes to a queen unless said
        if move in board.legal_moves:
            return move, "", ""
        return None, "illegal", why_illegal(board, move)
    if re.fullmatch(r"[o0]\s*-?\s*[o0](\s*-?\s*[o0])?", t, re.I):
        t = "O-O-O" if len(re.findall(r"[o0]", t, re.I)) == 3 else "O-O"
    t = t.replace(" ", "")
    illegal = ""
    for reading in dict.fromkeys([t] + ([t[0] + t[1:].lower(), t.lower()] if t.isupper() else [])):
        try:
            return board.parse_san(reading), "", ""
        except chess.AmbiguousMoveError:
            return None, "ambiguous", ""
        except chess.IllegalMoveError:
            illegal = illegal or reading
        except ValueError:
            pass
    if illegal:
        name = "Castling" if illegal.startswith("O-O") else illegal
        return None, "illegal", f"{name} is not legal here."
    if MOVE_SHAPE.fullmatch(t):
        return None, "unreadable", ""
    return None, "not a move", ""


def why_illegal(board, move) -> str:
    """Said aloud: the move as read and what is wrong with it, when that is
    plain - a misread shows at once."""
    a, b = chess.square_name(move.from_square), chess.square_name(move.to_square)
    piece = board.piece_at(move.from_square)
    if piece is None:
        return f"There is no piece of yours on {a}."
    if piece.color != board.turn:
        return f"The piece on {a} is not yours."
    return f"{PIECES[piece.piece_type].capitalize()} from {a} to {b} is not legal here."


def spoken(board, move) -> str:
    """The move in words, said from the board before it: "knight from g1 to
    f3", "pawn from e4 takes on d5", "castles kingside"; then ", check" or
    ", checkmate" as it lands."""
    if board.is_castling(move):
        text = "castles kingside" if chess.square_file(move.to_square) == 6 else "castles queenside"
    else:
        piece = PIECES[board.piece_type_at(move.from_square)]
        a, b = chess.square_name(move.from_square), chess.square_name(move.to_square)
        if board.is_en_passant(move):
            text = f"{piece} from {a} takes on {b} en passant"
        elif board.is_capture(move):
            text = f"{piece} from {a} takes on {b}"
        else:
            text = f"{piece} from {a} to {b}"
        if move.promotion:
            text += f" and promotes to a {PIECES[move.promotion]}"
    after = board.copy(stack=False)
    after.push(move)
    return text + (", checkmate" if after.is_checkmate() else ", check" if after.is_check() else "")


# ---------------------------------------------------------------- the table

class ChessTable:
    """The one game at the table.  handle() takes what was written and returns
    what John says, or None when the writing is not for the game."""

    def __init__(self, make_engine=new_engine, clock=time.time, rng=None):
        self.make_engine, self.clock, self.rng = make_engine, clock, rng or random.Random()
        self.reset()

    def reset(self):
        if getattr(self, "engine", None):
            self.engine.close()
        self.engine = None
        self.board = None
        self.robot_white = False
        self.state = "off"             # off, colour, play, quit?, rematch?, swap?
        self.last_move_written = ""
        self.misses = 0
        self.last_at = 0.0
        self.asked_idle = False

    # -- when the table is in charge -------------------------------------------
    def active(self) -> bool:
        if self.state != "off" and self.clock() - self.last_at > DROP_IDLE_S:
            self.reset()                   # abandoned: dropped without a word
        return self.state != "off"

    def starts(self, writing: str) -> bool:
        return bool(START.search(writing or ""))

    def idle(self) -> str:
        """Someone is there and wrote nothing: once, after a long pause, ask."""
        if (self.state in ("play", "quit?") and not self.asked_idle
                and self.clock() - self.last_at > ASK_IDLE_S):
            self.asked_idle = True
            return "Are we still playing? " + self.whose_turn()
        return ""

    # -- one board ----------------------------------------------------------------
    def handle(self, writing: str) -> str | None:
        w = (writing or "").strip()
        if self.state == "off":
            if not self.starts(w):
                return None
            if chess is None or (self.engine is None and not self.start_engine()):
                return "I can't play chess right now: my chess engine is not installed."
            self.state = "colour"
            self.touch()
            return "Let's play chess. Do you want white or black?"
        norm = re.sub(r"\W+", "", w.lower())
        if self.state in ("play", "quit?") and norm and norm == self.last_move_written:
            return ""                      # the board with the move just played, again
        self.touch()
        return getattr(self, "on_" + self.state.rstrip("?"))(w)

    def touch(self):
        self.last_at = self.clock()
        self.asked_idle = False

    def start_engine(self) -> bool:
        try:
            self.engine = self.make_engine()
        except Exception:
            self.engine = None
        return self.engine is not None

    # -- states -------------------------------------------------------------------
    def on_colour(self, w):
        if STOP.match(w) or NO.match(w):
            self.reset()
            return "All right, no chess."
        if WHITE.search(w) and not BLACK.search(w):
            human_white = True
        elif BLACK.search(w) and not WHITE.search(w):
            human_white = False
        elif EITHER.search(w):
            human_white = self.rng.random() < 0.5
        else:
            return "Do you want white or black? Write STOP if you'd rather not play."
        return self.begin(robot_white=not human_white)

    def begin(self, robot_white: bool) -> str:
        self.board = chess.Board()
        self.robot_white = robot_white
        self.state = "play"
        self.last_move_written = ""
        self.misses = 0
        if self.engine is None and not self.start_engine():
            self.reset()
            return "I can't play chess right now: my chess engine is not installed."
        if robot_white:
            return "I have white. " + self.robot_moves()
        return "You have white. Your move."

    def on_play(self, w):
        if UNDO.search(w):
            return self.undo()
        if RESIGN.search(w):
            return self.end("You resign, so I win.", robot_won=True)
        if DRAW.search(w):
            if self.engine.score(self.board, self.robot_colour()) <= DRAW_ACCEPT_CP:
                return self.end("I accept the draw.", draw=True)
            return "No, I'd rather play on. " + self.whose_turn()
        move, why, what = parse_move(self.board, w)
        if move is None:
            # Only what is plainly not a move asks about ending the game; a move
            # that cannot be played or read is said so, and the game goes on.
            if why == "not a move":
                self.state, self.misses = "quit?", 0
                return "That is not a move. Do you want to end the game?"
            said = {"ambiguous": "I can read that two ways. Please write it as from and to "
                                 "squares, like e2 e4.",
                    "unreadable": "I can't read that move. Please write it again."
                    }.get(why, f"{what} Your move.")
            self.misses += 1
            if self.misses >= MAX_MISSES:      # a long run of them: maybe they want to stop
                self.state, self.misses = "quit?", 0
                return said.rsplit(" Your move.", 1)[0].rsplit(" Please write", 1)[0] + (
                    " Do you want to end the game?")
            self.state = "play"
            return said
        self.state, self.misses = "play", 0
        self.last_move_written = re.sub(r"\W+", "", w.lower())
        said = f"You played {spoken(self.board, move)}. "
        self.board.push(move)
        if self.board.is_game_over(claim_draw=True):
            return said + self.result()
        return said + self.robot_moves()

    def on_quit(self, w):
        if YES.match(w) or STOP.match(w):
            self.reset()
            return "All right, the game is over."
        if NO.match(w):
            self.state, self.misses = "play", 0
            return "Then we play on. " + self.whose_turn()
        return self.on_play(w)             # a legal move is an answer too

    def on_rematch(self, w):
        if YES.match(w):
            self.state = "swap?"
            return f"Shall we swap colours? You would have {'white' if self.robot_white else 'black'}."
        if NO.match(w) or STOP.match(w):
            self.reset()
            return "All right. Thank you for the game."
        self.reset()                       # something else: the table lets go
        return None

    def on_swap(self, w):
        if YES.match(w):
            return self.begin(robot_white=not self.robot_white)
        if NO.match(w):
            return self.begin(robot_white=self.robot_white)
        self.reset()
        return None

    # -- moves and endings ---------------------------------------------------------
    def robot_colour(self):
        return chess.WHITE if self.robot_white else chess.BLACK

    def robot_moves(self) -> str:
        move = self.engine.best(self.board)
        said = f"I play {spoken(self.board, move)}."
        self.board.push(move)
        if self.board.is_game_over(claim_draw=True):
            return said + " " + self.result()
        return said + " Your move."

    def whose_turn(self) -> str:
        if self.board is None:
            return ""
        if self.board.turn == self.robot_colour():
            return self.robot_moves()
        last = self.board.peek() if self.board.move_stack else None
        if last is not None:
            before = self.board.copy()
            before.pop()
            if before.turn == self.robot_colour():
                return f"My last move was {spoken(before, last)}. Your move."
        return "Your move."

    def undo(self) -> str:
        """Their last move, and John's answer to it if there was one."""
        b = self.board
        theirs = len(b.move_stack) - (1 if self.robot_white else 0)   # moves they made, at least
        if theirs < 1:
            return "There is no move of yours to undo. Your move."
        if b.turn != self.robot_colour():                  # John answered: take that back too
            b.pop()
        b.pop()
        self.last_move_written = ""
        self.misses = 0
        return "Undone. Your move again."

    def result(self) -> str:
        b = self.board
        if b.is_checkmate():
            robot_won = b.turn != self.robot_colour()
            # The move itself already ended in ", checkmate".
            return self.end("I win." if robot_won else "You win, well played.", robot_won=robot_won)
        if b.is_stalemate():
            why = "stalemate"
        elif b.is_insufficient_material():
            why = "insufficient material"
        elif b.can_claim_fifty_moves():
            why = "the fifty-move rule"
        else:
            why = "threefold repetition"
        return self.end(f"It's a draw by {why}.", draw=True)

    def end(self, text: str, robot_won: bool = False, draw: bool = False) -> str:
        """The result aloud, then a rematch: for them if John won, for John if it
        lost."""
        self.state = "rematch?"
        if draw:
            return f"{text} Shall we play again?"
        if robot_won:
            return f"{text} Do you want a rematch?"
        return f"{text} Will you give me a rematch?"
