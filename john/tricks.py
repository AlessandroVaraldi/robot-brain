"""Things John works out exactly, where a language model would guess.

The writing is searched for something to work out.  If there is one, the
result is computed here and handed to the reply as a fact to say; the model
only puts it in its own words.  Each trick returns a Fact, or None when the
writing does not call for it.

  calendar     the day of the week of any date, how old someone is, how many
               days to a birthday
  arithmetic   + - * / and powers, square roots, primes and prime factors
  sequence     a list to remember, said back in order, backwards, sorted, or
               one item at a time, for as long as the encounter lasts
  anagram      real English words from the letters of a word or of their name
               (needs the wordfreq package; without it the trick is off)
"""

from __future__ import annotations

import ast
import datetime as dt
import math
import operator
import re
from dataclasses import dataclass


@dataclass
class Fact:
    text: str        # the fact, as the model is given it
    say: str         # what the reply must contain; "a | b | c": all, in this order
    spoken: str = "" # said instead when the reply loses the result; text if empty
    never: str = ""  # what the reply must not contain ("a / b": any of them)

    def fallback(self) -> str:
        return self.spoken or self.text


def work_out(writing: str, today: dt.date | None = None) -> Fact | None:
    """The first trick that applies to this writing, or None."""
    today = today or dt.date.today()
    rest, fact = calendar(writing or "", today)
    return fact or arithmetic(rest)


def said(fact: Fact, reply: str) -> bool:
    """Does the reply carry the fact?  "a | b": both, in this order; "3 / three":
    either.  Digits are compared without separators: "7,006,652" says 7006652."""
    flat = re.sub(r"(?<=\d)[,\s](?=\d)", "", (reply or "").lower())
    if fact.never and any(alt in flat for alt in fact.never.lower().split(" / ")):
        return False
    at = 0
    for part in fact.say.lower().split(" | "):
        hits = [(i, len(alt)) for alt in part.split(" / ") for i in [flat.find(alt, at)] if i >= 0]
        if not hits:
            return False
        i, n = min(hits)
        at = i + n
    return True


# ---------------------------------------------------------------- calendar

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], 1)}
MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
MONTHS["sept"] = 9
MONTH = r"(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
DAY = r"(\d{1,2})(?:st|nd|rd|th)?"
YEAR = r"(\d{4})"
DATE_PATTERNS = [
    # 12 MARCH 1990, 12TH OF MARCH, 12 MAR 1990
    (re.compile(rf"\b{DAY}\s+(?:of\s+)?{MONTH}(?:,?\s+{YEAR})?\b", re.I), ("d", "m", "y")),
    # MARCH 12, 1990 / MARCH 12TH
    (re.compile(rf"\b{MONTH}\s+{DAY}(?:,?\s+{YEAR})?\b", re.I), ("m", "d", "y")),
    # 1990-03-12
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), ("y", "m", "d")),
    # 12/03/1990, 12.03.90: day first unless that is impossible
    (re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{2}|\d{4})\b"), ("a", "b", "y")),
]
BIRTH = re.compile(r"\b(born|birthday|birth)\b", re.I)


def find_date(writing: str):
    """(a date, or (month, day) when no year is given; the matched span), or
    (None, None)."""
    for pattern, order in DATE_PATTERNS:
        for m in pattern.finditer(writing):
            parts = dict(zip(order, m.groups()))
            try:
                if "a" in parts:
                    a, b = int(parts["a"]), int(parts["b"])
                    day, month = (a, b) if b <= 12 else (b, a)
                else:
                    day = int(parts["d"])
                    month = (int(parts["m"]) if parts["m"].isdigit()
                             else MONTHS[parts["m"].lower()])
                year = parts.get("y")
                if year is None:
                    dt.date(2000, month, day)          # a leap year: 29 February is valid
                    return (month, day), m.span()
                year = int(year) + (0 if len(year) == 4 else 1900 if int(year) > 30 else 2000)
                return dt.date(year, month, day), m.span()
            except (ValueError, KeyError):
                continue
    return None, None


def calendar(writing: str, today: dt.date):
    """(the writing without the date, Fact or None)."""
    date, span = find_date(writing)
    if date is None:
        return writing, None
    rest = writing[:span[0]] + " " + writing[span[1]:]
    if isinstance(date, dt.date):
        if date.year < 1583:                       # before the Gregorian calendar
            return rest, None
        weekday = date.strftime("%A")
        when = f"{date.day} {date.strftime('%B')} {date.year}"
        verb = "was" if date < today else "is" if date == today else "will be"
        text = spoken = f"{when} {verb} a {weekday}."
        if BIRTH.search(writing) and date <= today:
            age = today.year - date.year - ((today.month, today.day) < (date.month, date.day))
            text += f" Age of someone born then: {age}."
            spoken = f"{when} was a {weekday}, so you are {age} now."
        return rest, Fact(text, weekday, spoken)
    month, day = date
    nxt = next_occurrence(month, day, today)
    weekday = nxt.strftime("%A")
    days = (nxt - today).days
    text = (f"{day} {nxt.strftime('%B')} is today, a {weekday}." if days == 0 else
            f"The next {day} {nxt.strftime('%B')}, in {nxt.year}, is a {weekday}, in {days} "
            f"day{'s' if days != 1 else ''}.")
    return rest, Fact(text, weekday)


def next_occurrence(month: int, day: int, today: dt.date) -> dt.date:
    for year in range(today.year, today.year + 9):
        try:
            d = dt.date(year, month, day)
        except ValueError:                         # 29 February outside a leap year
            continue
        if d >= today:
            return d
    raise ValueError("no such day")


# ---------------------------------------------------------------- arithmetic

WORDS = [(r"\bsquare\s+root\s+of\b", " sqrt "), (r"√", " sqrt "),
         (r"\bto\s+the\s+power\s+of\b", "**"), (r"\^", "**"),
         (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"),
         (r"\bdivided\s+by\b", "/"), (r"÷", "/"), (r"\btimes\b", "*"),
         (r"\bmultiplied\s+by\b", "*"), (r"×", "*"), (r"(?<=\d)\s*[xX]\s*(?=\d)", "*"),
         (r"\bplus\b", "+"), (r"\bminus\b", "-")]
# A calculation is worked out only when it is asked for, or is all that is
# written: "THINK OF A NUMBER 1-100" is not a subtraction.
ASKING = re.compile(r"\?|=|^\s*(what|how much|calculate|compute|work out|solve)\b", re.I)
# Any number as it may be written: 7, -7, 0.5, 1/2, 1e6, 1,000,003, SEVEN, PI.
NUM = (r"(-?\d[\d,]*(?:\.\d+)?(?:e\d+)?(?:/\d+)?|pi|π|zero|one|two|three|four|five|six|"
       r"seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
       r"eighteen|nineteen|twenty)")
# "Not prime" carries the word "prime" too.
NOT_PRIME = "not prime / not a prime / isn't prime / isn't a prime / is not / no,"
PRIME_Q = re.compile(rf"\bis\s+{NUM}\s+(?:a\s+)?prime\b|\b{NUM}\s+is\s+(?:a\s+)?prime\b", re.I)
FACTOR_Q = re.compile(rf"\b(?:prime\s+factors?\s+of|factori[sz]e|factor)\s+{NUM}", re.I)
SQRT_Q = re.compile(rf"\bsqrt\s*\(?\s*{NUM}\s*\)?", re.I)
WORD_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
PRIME_MAX = 10 ** 18                # checked exactly up to here; above it is too big to say
FACTOR_MAX = 10 ** 12               # factored in full up to here
EXPR = re.compile(r"[\d.]+(?:\s*(?:\*\*|[-+*/])\s*\(?\s*[\d.]+\s*\)?)+")
MAX_DIGITS = 30
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow}


def arithmetic(writing: str) -> Fact | None:
    text = writing or ""
    for pattern, repl in WORDS:
        text = re.sub(pattern, repl, text, flags=re.I)
    m = FACTOR_Q.search(text)
    if m:
        return factor_fact(m.group(1))
    m = PRIME_Q.search(text)
    if m:
        return prime_fact(m.group(1) or m.group(2))
    m = SQRT_Q.search(text)
    if m:
        return sqrt_fact(m.group(1))
    alone = re.fullmatch(r"\s*" + EXPR.pattern + r"\s*=?\s*", text)
    if not ASKING.search(writing or "") and not alone:
        return None
    m = EXPR.search(text)
    if not m:
        return None
    expr = m.group(0).strip()
    try:
        value = evaluate(ast.parse(expr, mode="eval").body)
    except (ValueError, ZeroDivisionError, OverflowError, SyntaxError, TypeError):
        return None
    shown = number(value)
    if shown is None:
        return None
    return Fact(f"{pretty(expr)} = {shown}.", shown)


def evaluate(node):
    """Numbers and + - * / ** only: nothing else in the writing is ever run."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = evaluate(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        a, b = evaluate(node.left), evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 100 or abs(a) > 10 ** 6):
            raise ValueError("too big to say")
        if (isinstance(node.op, ast.Div) and isinstance(a, int) and isinstance(b, int)
                and b and a % b == 0):
            return a // b
        return OPS[type(node.op)](a, b)
    raise ValueError("not arithmetic")


def number(value) -> str | None:
    """As said aloud: whole numbers in full, the rest to 6 significant digits."""
    if isinstance(value, float) and value.is_integer() and abs(value) < 10 ** 15:
        value = int(value)
    if isinstance(value, int):
        return str(value) if len(str(abs(value))) <= MAX_DIGITS else None
    if not math.isfinite(value):
        return None
    return f"{value:.6g}"


def pretty(expr: str) -> str:
    signs = {"*": " × ", "/": " ÷ ", "**": "^"}
    return re.sub(r"\s*(\*\*|[-+*/])\s*",
                  lambda m: signs.get(m.group(1), f" {m.group(1)} "), expr)


def read_number(s: str):
    """(value, as said): an int when the number is whole, else a float; π as
    math.pi.  None if it cannot be read."""
    t = (s or "").strip().lower()
    if t in ("pi", "π"):
        return math.pi, "π"
    if t in WORD_NUMBERS:
        return WORD_NUMBERS[t], t
    t = t.replace(",", "")
    if re.fullmatch(r"-?\d+", t):
        return int(t), t                    # whole numbers exactly: a float loses digits
    try:
        if "/" in t:
            a, b = t.split("/")
            value = int(a) / int(b)
        else:
            value = float(t)
    except (ValueError, ZeroDivisionError):
        return None
    if value.is_integer() and abs(value) < 10 ** 30:
        return int(value), str(int(value))
    return value, s.strip()


def not_a_whole_number(shown: str, what: str) -> Fact:
    return Fact(f"{shown} is not a whole number, and only whole numbers {what}.", "whole")


def prime_fact(s: str) -> Fact | None:
    read = read_number(s)
    if read is None:
        return None
    n, shown = read
    if not isinstance(n, int):
        return not_a_whole_number(shown, "can be prime")
    if n < 0:
        return Fact(f"{n} is not prime: prime numbers are positive whole numbers.", "not")
    if n < 2:
        return Fact(f"{n} is not prime: a prime has exactly two divisors, 1 and itself.", "not")
    if n > PRIME_MAX:
        return Fact(f"{n} is too big for you to check whether it is prime.", "too big")
    if is_prime(n):
        return Fact(f"{n} is a prime number.", "prime", never=NOT_PRIME)
    if n <= FACTOR_MAX:
        return Fact(f"{n} is not prime: it is {' × '.join(map(str, factors(n)))}.", "not")
    small = next((p for p in range(2, 10 ** 6) if n % p == 0), None)
    return Fact(f"{n} is not prime" + (f": it is divisible by {small}." if small else "."), "not")


def factor_fact(s: str) -> Fact | None:
    read = read_number(s)
    if read is None:
        return None
    n, shown = read
    if not isinstance(n, int):
        return not_a_whole_number(shown, "have prime factors")
    if n < 2:
        return Fact(f"{n} has no prime factors: only whole numbers from 2 up have them.", "no prime")
    if n > FACTOR_MAX:
        return Fact(f"{n} is too big for you to factor.", "too big")
    fs = factors(n)
    if len(fs) == 1:
        return Fact(f"{n} is prime: its only prime factor is itself.", "prime", never=NOT_PRIME)
    return Fact(f"{n} = {' × '.join(map(str, fs))}.", str(fs[-1]))


def sqrt_fact(s: str) -> Fact | None:
    read = read_number(s)
    if read is None:
        return None
    x, shown = read
    if abs(x) > 10 ** 24:
        return Fact(f"{shown} is too big for you to take its square root.", "too big")
    if x < 0:
        r = math.sqrt(-x)
        r_shown = str(int(r)) if r.is_integer() else f"{r:.6g}"
        return Fact(f"{shown} has no square root among ordinary numbers: its square roots "
                    f"are the imaginary ±{r_shown}i.", "imaginary")
    if isinstance(x, int) and math.isqrt(x) ** 2 == x:
        r = math.isqrt(x)
        return Fact(f"The square root of {x} is {r}, exactly.", str(r))
    r = math.sqrt(x)
    r_shown = f"{r:.6g}"
    if r * r == x:
        return Fact(f"The square root of {shown} is {r_shown}, exactly.", r_shown)
    return Fact(f"The square root of {shown} is about {r_shown}.", r_shown)


def is_prime(n: int) -> bool:
    """Exact for every n below 3.3e24 (Miller-Rabin with these fixed bases)."""
    if n < 2:
        return False
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in bases:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d, r = d // 2, r + 1
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def factors(n: int) -> list[int]:
    out, p = [], 2
    while p * p <= n:
        while n % p == 0:
            out.append(p)
            n //= p
        p += 1 if p == 2 else 2
    return out + ([n] if n > 1 else [])


# ---------------------------------------------------------------- sequence

REMEMBER = re.compile(r"^\W*(also\s+|and\s+)?(?:please\s+)?(?:remember|memori[sz]e)\b"
                      r"(?:\s+(?:this|these|the\s+following|that))?\W*(.*)$", re.I | re.S)
MAX_ITEMS = 30
ORDINALS = {w: i for i, w in enumerate(
    ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
     "ninth", "tenth", "eleventh", "twelfth"], 1)}
NTH = re.compile(r"\b(" + "|".join(ORDINALS) + r"|last|(\d{1,2})(?:st|nd|rd|th))\b"
                 r"(?:\s+(?:one|item|thing|word|number))?", re.I)
ABOUT_LIST = re.compile(r"\b(remember|list|sequence|items?|things|words|numbers|ones?|"
                        r"them|it|back)\b", re.I)
BACKWARDS = re.compile(r"\b(backwards?|in\s+reverse|reversed?)\b", re.I)
SORTED = re.compile(r"\b(sorted|sort\s+them|in\s+order|alphabetical(?:ly)?|smallest\s+to\s+"
                    r"(?:largest|biggest))\b", re.I)
COUNT = re.compile(r"\bhow\s+many\s+(?:things|items|words|numbers|of\s+them|did\s+i)\b", re.I)
RECALL = re.compile(r"\b(what\s+did\s+i\s+(?:ask|tell)\s+you\s+to\s+remember|repeat\s+(?:it|them|"
                    r"the\s+list)|say\s+(?:it|them)\s+back|what\s+(?:were|are)\s+they|"
                    r"the\s+list|recite)\b", re.I)


# Words that make a phrase a sentence rather than a list of things.
SENTENCE_WORDS = set("""the a an is are was were be to of that this my your his her it
and or but not in on at for with i you he she we they""".split())


def items_of(text: str) -> list[str]:
    """What is to be remembered: comma-separated; else single words and numbers;
    else, if it reads as a sentence, the sentence as one item - "THAT ALESSANDRO IS
    A THIEF" was said back as "Alessandro, Is, A, Thief"."""
    words = text.split()
    if re.search(r",|;", text):
        parts = re.split(r",|;|\band\b", text)
    elif any(w.lower().strip(".!?") in SENTENCE_WORDS for w in words[1:]):
        parts = [re.sub(r"^\s*that\s+", "", text, flags=re.I)]
    else:
        parts = words
    # Capitalised, not shouted: a speech engine may spell out a word in capitals.
    out = [" ".join(w.capitalize() for w in p.strip(" .!?:-\"'").split()) for p in parts]
    if len(parts) == 1 and len(words) > 1 and not re.search(r",|;", text) and out:
        out = [out[0][:1] + out[0][1:].lower()]      # a sentence: "Alessandro is a thief"
    return [p for p in out if p][:MAX_ITEMS]


def sequence(writing: str, held: list[str]) -> tuple[list[str], Fact | None]:
    """(the list to keep from now on, Fact or None).  A board starting with
    REMEMBER starts a list, ALSO REMEMBER adds to it; the questions after that
    are about the list held."""
    w = writing or ""
    m = REMEMBER.match(w)
    if m and items_of(m.group(2)):
        new = items_of(m.group(2))
        kept = (held + new)[:MAX_ITEMS] if m.group(1) else new
        text = (f"They gave you {len(new)} thing{'s' if len(new) != 1 else ''} to remember"
                f"{' in addition' if m.group(1) else ''}: {', '.join(new)}. Holding "
                f"{len(kept)} in all. Tell them you have got it, without repeating them.")
        return kept, Fact(text, "")
    if not held:
        # Asked for a list never given: said so, or the model made one up ("You
        # asked me to remember the anagram of LISTEN").
        if re.search(r"\bremember\b", w, re.I) and (RECALL.search(w) or BACKWARDS.search(w)
                                                    or COUNT.search(w) or NTH.search(w)):
            return held, Fact("They have not given you anything to remember yet.", "",
                              "You haven't given me anything to remember yet.")
        return held, None
    n = len(held)
    shown = lambda xs: ", ".join(xs)  # noqa: E731
    if BACKWARDS.search(w):
        back = held[::-1]
        return held, Fact(f"The list they gave you, backwards: {shown(back)}.",
                          " | ".join(back).lower(), f"Backwards: {shown(back)}.")
    if SORTED.search(w):
        order = sorted(held, key=lambda x: (0, float(x), "") if re.fullmatch(r"-?\d+(\.\d+)?", x)
                       else (1, 0.0, x))
        return held, Fact(f"The list they gave you, sorted: {shown(order)}.",
                          " | ".join(order).lower(), f"Sorted: {shown(order)}.")
    if COUNT.search(w):
        return held, Fact(f"They gave you {n} thing{'s' if n != 1 else ''} to remember.", str(n))
    m = NTH.search(w)
    if m and ABOUT_LIST.search(w):
        word = m.group(1).lower()
        i = n if word == "last" else ORDINALS.get(word) or int(m.group(2))
        if 1 <= i <= n:
            return held, Fact(f"Item {i} of the {n} they gave you: {held[i - 1]}.",
                              held[i - 1].lower(), f"Number {i} was {held[i - 1]}.")
        return held, Fact(f"They gave you only {n} things, so there is no number {i}.", str(n))
    if RECALL.search(w):
        return held, Fact(f"The list they gave you, in order: {shown(held)}.",
                          " | ".join(held).lower(), f"In order: {shown(held)}.")
    return held, None


# ---------------------------------------------------------------- anagram

ANAGRAM = re.compile(r"\banagrams?\b(?:\s+(?:of|for|from|with))?\s+(?:the\s+word\s+)?"
                     r"(my\s+name|[a-z][a-z' -]{1,30}?)\s*[?.!]*$", re.I)
VOCABULARY = 50000                  # the most frequent English words considered
MIN_ZIPF = 3.0                      # rarer than this is not a word people know
# Words John must never say, whatever letters they come from.  Kept as hashes
# so the list does not read as one.
NEVER = {
    "02ac484597c8", "037b3e936d13", "03913c546d46", "0391a1e58f32", "08bc5beda7a9",
    "0b77230a89e4", "11dbf66d28b6", "19a38662e23e", "1ac5f681171f", "1f45855e4097",
    "1f956b5138bf", "21554666d275", "22bf5d4a65ff", "23c1bf668c18", "266f83d202fa",
    "2e71777dff35", "2f61cb7837b8", "320d1a474a0d", "35ed5406781e", "3844f1150f73",
    "3852a12677f0", "38cbc7bbe54a", "38d0f91a99c5", "39f6f95327b3", "3b19ecd69b49",
    "3bf858ffe8f9", "3e83b13d99bf", "46e6f4054939", "4b8cfc115af4", "4dbc8c31da4f",
    "56ece01521bf", "57456e092ee2", "59033478180d", "5c9b0c957784", "5eb965dd8c80",
    "63aef6ff8e1e", "66280fb19d5c", "66e7a97c5557", "68bb04bd54b8", "6a3578663cb2",
    "6b7b1987ddad", "71bcdde68808", "737de7673447", "7ac78dd9d9bb", "7f50fd4afd66",
    "7fd86b25e099", "819d7c152e96", "82da4c33e3a5", "842df0e20f51", "85fe8de475bc",
    "867268472cd8", "8b7cef62e842", "8c4947e96c7c", "8f595011a395", "921f208a404d",
    "926dee392274", "a5cec7af5f7a", "a9c241cebb7c", "ab14d94055e7", "ac04b70e6de3",
    "b5af50a4c265", "b6928c296eb2", "bbbb7904a751", "bcee59cecbc4", "bf5afc18dfbc",
    "bff272e9d673", "c0049442a7ca", "c016fad84319", "c177922cb771", "c22d4a0c9612",
    "c71230fc13c6", "c80f5bc166cd", "c8645c4a303b", "c976720ffb80", "cecafb4d7ac2",
    "d64a57b064fd", "d7eb2aa54ec8", "dcd6732d222b", "dfb04e6fdc8e", "e3a82186438a",
    "e49524050d4b", "ebbe2e8ed1f6", "f1358a077206", "f1ca6ecc6865", "f73127d74a6a",
    "f8a17e958f70", "f9c8390832e5",
}
_words = None


def never(word: str) -> bool:
    import hashlib
    return hashlib.sha1(word.encode()).hexdigest()[:12] in NEVER


def vocabulary():
    """{sorted letters: [words, most common first]}, built once; empty without
    wordfreq.  Word frequencies alone let in abbreviations and names ("UCLA",
    "Dior"): with a word list in data/words.txt, only its lowercase entries
    count as words."""
    global _words
    if _words is None:
        _words = {}
        try:
            import wordfreq
        except ImportError:
            return _words
        from paths import DATA
        listed = DATA / "words.txt"
        known = ({w.strip() for w in listed.open(errors="ignore") if w.strip().islower()}
                 if listed.exists() else None)
        for w in wordfreq.top_n_list("en", VOCABULARY):
            if (w.isalpha() and w.isascii() and (len(w) > 1 or w in ("a", "i"))
                    and (known is None or w in known)
                    and wordfreq.zipf_frequency(w, "en") >= MIN_ZIPF and not never(w)):
                _words.setdefault("".join(sorted(w)), []).append(w)
    return _words


# In a pair of words, short ones must be ones everybody knows: "AI ARM" and
# "FA ISO" are letters, not words.
SHORT = {"a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in",
         "is", "it", "me", "my", "no", "of", "oh", "on", "or", "so", "to", "up", "us", "we"}


def fits_a_pair(w: str) -> bool:
    return len(w) >= 4 or w in SHORT or (len(w) == 3 and _zipf(w) >= 4.0)


def anagrams(letters: str, limit: int = 3) -> list[str]:
    """One word if there is one, else two, commonest first; never the word itself."""
    words = vocabulary()
    key = "".join(sorted(letters))
    one = [w for w in words.get(key, []) if w != letters]
    if one:
        return one[:limit]
    from collections import Counter
    need, pairs = Counter(letters), []
    for k, ws in words.items():
        if len(k) > len(letters) - 1 or Counter(k) - need:
            continue
        rest = "".join(sorted((need - Counter(k)).elements()))
        if k > rest or rest not in words:          # each pair once
            continue
        a = next((w for w in ws if fits_a_pair(w)), None)
        b = next((w for w in words[rest] if fits_a_pair(w)), None)
        if a and b:
            pairs.append((a, b) if len(a) >= len(b) else (b, a))
    pairs.sort(key=lambda p: -min(_zipf(p[0]), _zipf(p[1])))
    return [f"{a} {b}" for a, b in pairs[:limit]]


def _zipf(w):
    import wordfreq
    return wordfreq.zipf_frequency(w, "en")


def anagram(writing: str, name: str = "") -> Fact | None:
    m = ANAGRAM.search(writing or "")
    if not m or not vocabulary():
        return None
    target = m.group(1).strip()
    if re.fullmatch(r"my\s+name", target, re.I):
        if not name:
            return Fact("They ask for an anagram of their name, but they have not told "
                        "you their name yet.", "")
        target = name
    letters = re.sub(r"[^a-z]", "", target.lower())
    if len(letters) < 3:
        return None
    if len(letters) > 16:                  # left to the model, it invented one
        shown = " ".join(w.capitalize() for w in target.split())
        return Fact(f"{shown} is too long for you to rearrange into real words.", "",
                    f"{shown} is too long for me to rearrange.")
    found = anagrams(letters)
    shown = " ".join(w.capitalize() for w in target.split())
    if not found:
        return Fact(f"No English word or pair of words uses exactly the letters of {shown}.", "",
                    f"I can't make a real word out of {shown}, not even two.")
    best = found[0].capitalize()
    others = f" Others: {', '.join(x.capitalize() for x in found[1:])}." if found[1:] else ""
    return Fact(f"An anagram of {shown}: {best}.{others}", best.lower(),
                f"{shown} rearranged: {best}.")
