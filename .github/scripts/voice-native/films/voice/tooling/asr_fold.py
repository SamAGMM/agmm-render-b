#!/usr/bin/env python3
"""asr_fold.py - deterministic transcript-spelling normaliser for the whole-file ASR check (runtime delta 5, proposed 26 Oct 2026; root-approved terms).

WHAT IT IS FOR. The whole-file check (post_checks_local.py: local small.en, folded script-vs-heard word diff, ratio >= 0.97) fails a take when the AUDIO IS RIGHT and only the
transcript's SPELLING differs from the script: a Whisper transcript writes "maximize" for a British "maximise", "Morrison's" for "Morrisons", "£4.75" for "four pounds seventy-five".
Folding those is not loosening the check; it removes noise that is not a defect. A real defect (a dropped "-ed" ending, a dropped "it's", a missing or wrong word, a misheard
word) must still be a diff, so every fold below is explicit, fixed, and applied to BOTH sides before the ratio:

  1 UK/US spelling pairs   a FIXED list in this file (UK_US_STEMS + UK_US_WORDS). No fuzzy matching, no suffix guessing outside the listed stems. The UK form is the canonical form.
  2 possessive / plural    ONLY for tokens listed in that short's names.txt (Morrisons == Morrison's == Morrison). A token that is not a listed name is never folded.
  3 name aliases           ONLY from the short's aliases.txt (read only here; prepare.py rewrites names.txt on every run, so declarations live in their own file):\n                           <script form> = <heard form> | <reason>
                           The reason is mandatory; a line without one is ignored (and reported). "euston = houston" with no declaration stays a FAIL: a misheard word is a defect.
  4 currency and numbers   "£4.75" == "four pounds seventy-five" == "4.75";  "£150,000" == "150,000 pounds" == "a hundred and fifty thousand pounds" (pound amounts only, pence 1-99).

fold_pair(script_text, heard_text, names_text) -> (script_tokens, heard_tokens, applied) where tokens are asr_compare._canon tokens after the folds and `applied` lists every fold
that fired as a string ("uk_us: maximize->maximise"), so the check output shows exactly what was normalised.
`python3 asr_fold.py --selftest` runs the fixtures (folds that must fire; defects that must NOT be folded away)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ------------------------------------------------------------------------------------------------------------------------------ 1 UK/US
# stems: each stem expands to exactly these suffix pairs (UK -> US). A stem is listed by hand; nothing outside the list is touched.
_ISE = ["ise", "ised", "ises", "ising", "isation", "isations", "iser", "isers"]
_IZE = ["ize", "ized", "izes", "izing", "ization", "izations", "izer", "izers"]
# Each stem expands to exactly the eight suffix pairs above (UK -> US). A stem is listed by hand; nothing outside this list and UK_US_WORDS is touched.
ISE_STEMS = """organ apolog author capital categor central character civil critic digit emphas equal familiar final formal general global harmon hospital ideal immun individual initial
item legal liberal local maxim memor minim mobil modern monet neutral normal optim patron penal personal priorit privat rational real recogn revolution scrutin special standard
stabil summar symbol synchron synthes systemat custom util visual""".split()
UK_US_WORDS = {
    # -our / -or
    "colour": "color", "colours": "colors", "coloured": "colored", "colouring": "coloring", "colourful": "colorful", "favour": "favor", "favours": "favors", "favoured": "favored",
    "favourite": "favorite", "favourites": "favorites", "behaviour": "behavior", "behaviours": "behaviors", "neighbour": "neighbor", "neighbours": "neighbors",
    "neighbourhood": "neighborhood", "labour": "labor", "honour": "honor", "honours": "honors", "honoured": "honored", "harbour": "harbor", "humour": "humor", "rumour": "rumor",
    "rumours": "rumors", "flavour": "flavor", "flavours": "flavors", "endeavour": "endeavor", "vapour": "vapor", "saviour": "savior", "armour": "armor", "odour": "odor",
    # -re / -er
    "centre": "center", "centres": "centers", "centred": "centered", "metre": "meter", "metres": "meters", "kilometre": "kilometer", "kilometres": "kilometers", "litre": "liter",
    "litres": "liters", "theatre": "theater", "theatres": "theaters", "fibre": "fiber", "fibres": "fibers", "calibre": "caliber", "sabre": "saber", "spectre": "specter",
    # -ence / -ense
    "defence": "defense", "defences": "defenses", "offence": "offense", "offences": "offenses", "pretence": "pretense",
    # one spoken word, two spellings (root condition 1): licence/license, practise/practice (UK verb/noun; US uses practice for both)
    "licence": "license", "licences": "licenses", "licenced": "licensed", "licencing": "licensing", "practise": "practice", "practised": "practiced", "practises": "practices",
    "practising": "practicing",
    # programme and friends
    "programme": "program", "programmes": "programs", "programmed": "programed", "catalogue": "catalog", "catalogues": "catalogs", "dialogue": "dialog", "dialogues": "dialogs",
    # doubled consonants
    "travelled": "traveled", "travelling": "traveling", "traveller": "traveler", "travellers": "travelers", "cancelled": "canceled", "cancelling": "canceling", "cancellation": "cancelation",
    "labelled": "labeled", "labelling": "labeling", "modelled": "modeled", "modelling": "modeling", "levelled": "leveled", "levelling": "leveling", "signalled": "signaled",
    "signalling": "signaling", "fuelled": "fueled", "fuelling": "fueling", "counsellor": "counselor", "counselling": "counseling", "jewellery": "jewelry", "marvellous": "marvelous",
    "woollen": "woolen", "skilful": "skillful", "fulfil": "fulfill", "fulfils": "fulfills", "fulfilment": "fulfillment", "instalment": "installment", "instalments": "installments",
    "enrol": "enroll", "enrols": "enrolls", "enrolment": "enrollment", "enrolled": "enrolled", "distil": "distill", "appal": "appall",
    # others
    "ageing": "aging", "judgement": "judgment", "judgements": "judgments", "acknowledgement": "acknowledgment", "acknowledgements": "acknowledgments", "grey": "gray", "greying": "graying",
    "plough": "plow", "ploughs": "plows", "mould": "mold", "moulds": "molds", "sceptic": "skeptic", "sceptics": "skeptics", "sceptical": "skeptical", "aluminium": "aluminum",
    "analogue": "analog", "cosy": "cozy", "draught": "draft", "mum": "mom", "mummy": "mommy", "pyjamas": "pajamas", "tonne": "ton", "tonnes": "tons", "storey": "story", "storeys": "stories",
    "whilst": "while", "amongst": "among", "learnt": "learned", "spelt": "spelled", "burnt": "burned", "dreamt": "dreamed", "leapt": "leaped", "smelt": "smelled", "spilt": "spilled",
}


def _build_uk_us():
    m = {}
    for stem in ISE_STEMS:
        for uk, us in zip(_ISE, _IZE):
            m[stem + uk] = stem + us
    for w in ("analyse", "analysed", "analyses", "analysing", "paralyse", "paralysed", "paralysing", "catalyse", "catalysed"):
        m[w] = w.replace("ys", "yz", 1) if w != "analyses" else "analyses"       # analyse/analyze, paralyse/paralyze; the plural 'analyses' is the same word in both
    for uk, us in UK_US_WORDS.items():
        m[uk] = us
    return m


UK_TO_US = _build_uk_us()
US_TO_UK = {us: uk for uk, us in UK_TO_US.items()}


def uk_us_key(tok):
    """canonical (UK) form of a token if it is in the fixed list, else the token unchanged"""
    return US_TO_UK.get(tok, tok)


# ------------------------------------------------------------------------------------------------------------------------------ names.txt
def parse_names(names_text):
    """-> set of name bases. names.txt lines are names only (multi-word allowed); prepare.py regenerates this file from the script, so nothing else lives in it."""
    names = set()
    for raw in (names_text or "").splitlines():
        line = raw.strip()
        if line and not line.startswith("@"):
            for t in _toks(line):
                names.add(_base(t))
    return names


def parse_aliases(aliases_text):
    """-> ({script token tuple: set of heard token tuples}, ignored lines). aliases.txt (read ONLY by this module; prepare.py never touches it) has one declaration per line:
        <script form> = <heard form> | <reason>
    The reason is mandatory; a line without one is ignored and reported. A leading '@alias' is tolerated."""
    aliases, ignored = {}, []
    for raw in (aliases_text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("@alias"):
            line = line[6:].strip()
        pair, _, reason = line.partition("|")
        if "=" not in pair or not reason.strip():
            ignored.append(line)
            continue
        a, b = pair.split("=", 1)
        sa, sb = tuple(_toks(a)), tuple(_toks(b))
        if sa and sb:
            aliases.setdefault(sa, set()).add(sb)
        else:
            ignored.append(line)
    return aliases, ignored


def _toks(s):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", s.lower().replace("’", "'"))


def _base(t):
    """possessive / plural stem of a token that is in names.txt: morrisons / morrison's / morrison -> morrison"""
    t = t.replace("’", "'")
    if t.endswith("'s"):
        t = t[:-2]
    elif t.endswith("s'"):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s"):
        t = t[:-1]
    return t


# ------------------------------------------------------------------------------------------------------------------------------ 4 money
_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def _words_to_int(words):
    """number words -> int (0 .. 999,999) or None if the phrase is not entirely a number"""
    total = cur = 0
    seen = False
    i = 0
    ws = [w for w in words if w != "and"]
    if not ws:
        return None
    while i < len(ws):
        w = ws[i]
        if w in ("a", "an") and i + 1 < len(ws) and ws[i + 1] in ("hundred", "thousand"):
            cur += 1
        elif w in _ONES:
            cur += _ONES[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred" and (seen or cur):
            cur = (cur or 1) * 100
        elif w == "thousand" and (seen or cur):
            total += (cur or 1) * 1000
            cur = 0
        else:
            return None
        seen = True
        i += 1
    return total + cur


_NUMW = "|".join(sorted(list(_ONES) + list(_TENS) + ["hundred", "thousand", "and", "a"], key=len, reverse=True))
_PHRASE = r"(?:(?:%s)[\s-]+)*(?:%s)" % (_NUMW, _NUMW)
# '<amount> pounds <pence>' where each side is digits or number words: 'four pounds seventy-five', '4 pounds 75', 'four pounds 75', '4 pounds seventy five'
_MONEY = re.compile(r"\b(\d[\d,]*|%s)\s+pounds?\s+(\d{1,2}|%s)\b" % (_PHRASE, _PHRASE))


def _amount(s):
    s = s.strip().lower()
    if re.fullmatch(r"\d[\d,]*", s):
        return int(s.replace(",", ""))
    return _words_to_int(re.split(r"[\s-]+", s))


def money_fold(text):
    """'four pounds seventy-five' / '4 pounds 75' / '£4.75' -> '4.75'; '£150,000' / '150,000 pounds' -> '150000'. Pound amounts only; pence must be 1-99; everything else
    (apostrophes and all) is untouched. A phrase that is not entirely a number is left alone."""
    t = text.replace("\u2019", "'").lower()
    t = re.sub(r"£\s*(\d[\d,]*)\.(\d{2})\b", lambda m: "%s.%s" % (m.group(1).replace(",", ""), m.group(2)), t)
    t = re.sub(r"£\s*(\d[\d,]*)", lambda m: m.group(1).replace(",", ""), t)

    def pair(m):
        pounds, pence = _amount(m.group(1)), _amount(m.group(2))
        if pounds is None or pence is None or not 1 <= pence <= 99:
            return m.group(0)
        return "%d.%02d" % (pounds, pence)
    t = _MONEY.sub(pair, t)
    return re.sub(r"(\d[\d,]*)\s+pounds?\b", lambda m: m.group(1).replace(",", ""), t)


# ------------------------------------------------------------------------------------------------------------------------------ the fold
def _replace_seq(tokens, pat, rep):
    """replace every occurrence of the token sequence `pat` by `rep` (tuples); returns (tokens, count)"""
    n, out, i, c = len(pat), [], 0, 0
    while i < len(tokens):
        if tuple(tokens[i:i + n]) == pat:
            out.extend(rep)
            i += n
            c += 1
        else:
            out.append(tokens[i])
            i += 1
    return out, c


def fold_pair(script_text, heard_text, names_text="", canon=None, aliases_text=""):
    """-> (script tokens, heard tokens, applied list). `canon` is asr_compare._canon (injected so this module has no import cycle in tests)."""
    if canon is None:
        from asr_compare import _canon as canon
    names = parse_names(names_text)
    aliases, ignored = parse_aliases(aliases_text)
    applied = ["alias line ignored (no reason or malformed): " + l for l in ignored]
    a_txt, b_txt = money_fold(script_text), money_fold(heard_text)
    a, b = canon(a_txt), canon(b_txt)

    def norm(tokens, side):
        out = []
        for t in tokens:
            if isinstance(t, tuple):
                out.append(t)
                continue
            k = uk_us_key(t)
            if k != t:
                applied.append("uk_us(%s): %s->%s" % (side, t, k))
            t2 = k
            base = _base(t2)
            if base in names and base != t2:
                applied.append("name_form(%s): %s->%s" % (side, t2, base))
                t2 = base
            elif base in names:
                t2 = base
            out.append(t2)
        return out
    a, b = norm(a, "script"), norm(b, "heard")
    for sa, heards in aliases.items():
        script_alias = tuple(norm(list(sa), "alias"))
        for sb in heards:
            hb = tuple(norm(list(sb), "alias"))
            b, c = _replace_seq(b, hb, script_alias)
            if c:
                applied.append("alias: %s -> %s x%d" % (" ".join(sb), " ".join(sa), c))
    return a, b, applied


# ------------------------------------------------------------------------------------------------------------------------------ fixtures
def _ratio(a, b):
    import difflib
    sm = difflib.SequenceMatcher(a=[str(t) for t in a], b=[str(t) for t in b], autojunk=False)
    return sm.ratio(), [(op, " ".join(map(str, a[i1:i2])), " ".join(map(str, b[j1:j2]))) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]


def selftest():
    from asr_compare import _canon
    ok = True

    def expect(name, cond, extra=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name + (("  " + extra) if extra else ""))
        ok = ok and bool(cond)

    def old(s, h):
        return _ratio(_canon(s.lower()), _canon(h.lower()))

    def new(s, h, names="", aliases=""):
        a, b, ap = fold_pair(s, h, names, _canon, aliases)
        return _ratio(a, b) + (ap,)
    # ---- folds that must fire
    s = "the plan will maximise growth and the programme will organise the whole organisation"
    h = "The plan will maximize growth and the program will organize the whole organization."
    r0, d0 = old(s, h)
    r1, d1, ap = new(s, h)
    expect("UK/US pairs: 4 diffs before, none after", len(d0) >= 4 and r1 == 1.0 and not d1, "old %.3f new %.3f" % (r0, r1))
    r0, d0 = old("Morrisons plans cuts", "Morrison's plans cuts")
    r1, d1, ap = new("Morrisons plans cuts", "Morrison's plans cuts", "Morrisons\n")
    r2, d2, ap2 = new("Morrisons plans cuts", "Morrison's plans cuts", "")
    expect("possessive fold fires only for a name listed in names.txt", d0 and not d1 and d2, "listed=%s unlisted=%s" % (d1, d2))
    r1, d1, ap = new("Greggs has built a weather formula", "Greg's has built a weather formula", "Greggs\n", "greggs = greg's | whisper writes the possessive and drops a g\n")
    r2, d2, ap2 = new("Greggs has built a weather formula", "Greg's has built a weather formula", "Greggs\n")
    expect("declared alias (aliases.txt, with reason) folds Greggs/Greg's, undeclared stays a diff", not d1 and d2, "declared=%s undeclared=%s" % (d1, d2))
    r1, d1, ap = new("the Evolv system", "the evolve system", "Evolv\n", "evolv = evolve | brand is pronounced evolve\n")
    r2, d2, ap2 = new("the Evolv system", "the evolve system", "Evolv\n", "evolv = evolve\n")
    expect("an alias without a reason is ignored (and reported)", not d1 and d2 and any("ignored" in x for x in ap2))
    r2, d2, ap2 = new("the Evolv system", "the evolve system", "Evolv\n@alias evolv = evolve | brand is pronounced evolve\n", "")
    expect("an @alias line inside names.txt is NOT honoured (aliases come only from aliases.txt)", d2)
    r1, d1, ap = new("trains stop at Euston tonight", "trains stop at Houston tonight", "Euston\n")
    r2, d2, ap2 = new("trains stop at Euston tonight", "trains stop at Houston tonight", "Euston\n", "euston = houston\n")
    expect("Euston/Houston is NOT folded without a declared reason (a misheard word stays a fail)", d1 and d2, str(d1))
    r1, d1, ap = new("trains stop at Euston tonight", "trains stop at Houston tonight", "Euston\n", "euston = houston | Whisper hears the station name as Houston and the audio says Euston\n")
    expect("...and is folded when aliases.txt declares it with a reason", not d1)
    r1, d1, ap = new("she must practise and renew the licence and the defence plan", "She must practice and renew the license and the defense plan.")
    expect("one spoken word, two spellings: licence/license, practise/practice, defence/defense", not d1, str(d1))
    r1, d1, ap = new("the offence was licenced", "The offense was licensed.")
    expect("offence/offense and licenced/licensed", not d1, str(d1))
    r1, d1, ap = new("a baguette and a sausage roll drop to four pounds seventy-five", "A baguette and a sausage roll drop to £4.75.")
    expect("money: 'four pounds seventy-five' == '£4.75'", not d1, str(d1))
    r1, d1, ap = new("it costs a hundred and fifty thousand pounds a year", "It costs £150,000 a year.")
    expect("money: 'a hundred and fifty thousand pounds' == '£150,000'", not d1, str(d1))
    r1, d1, ap = new("it costs 150,000 pounds a year", "It costs £150,000 a year.")
    expect("money: '150,000 pounds' == '£150,000'", not d1, str(d1))
    r1, d1, ap = new("a baguette drops to four pounds seventy-five", "A baguette drops to 4 pounds 75.")
    expect("money: 'four pounds seventy-five' == '4 pounds 75' (the small.en form)", not d1, str(d1))
    r1, d1, ap = new("a baguette drops to four pounds seventy-five", "A baguette drops to £4.76.")
    expect("money: a WRONG amount (4.76 for 4.75) is still a diff", d1, str(d1))
    # ---- real defects must still fail (14-word sentences: one defect drops the ratio under 0.97)
    base = "she obtained the licence and it's valid for another twelve months in total"
    cases = [("dropped -ed ending", "she obtain the licence and it's valid for another twelve months in total"),
             ("dropped it's", "she obtained the licence and valid for another twelve months in total"),
             ("missing word", "she obtained the licence and it's valid for another months in total"),
             ("misheard word", "she obtained the licence and it's valid for another twelve months in tonal")]
    for nm, heard in cases:
        r0, d0 = old(base, heard)
        r1, d1, ap = new(base, heard, "Licence\n")
        expect("defect still fails after folds: %s" % nm, d1 and r1 < 0.97 and abs(r1 - r0) < 1e-9, "old %.3f new %.3f diffs %s" % (r0, r1, d1))
    r0, d0 = old("the contract is valid for another twelve months in total from today", "the contact is valid for another twelve months in total from today")
    r1, d1, ap = new("the contract is valid for another twelve months in total from today", "the contact is valid for another twelve months in total from today")
    expect("defect still fails after folds: contract heard as contact", d1 and abs(r1 - r0) < 1e-9 and r1 < 0.97, "old %.3f new %.3f diffs %s" % (r0, r1, d1))
    # a defect INSIDE a folded word's neighbourhood is not hidden: 'maximise' spoken as 'maximum' stays a diff
    r1, d1, ap = new("we must maximise returns this year", "we must maximum returns this year")
    expect("a different word that merely looks similar is not folded", d1, str(d1))
    # US spelling of a word NOT in the fixed list is not guessed
    r1, d1, ap = new("the warrior won", "the warier won")
    expect("no fuzzy matching: unlisted spellings stay diffs", d1)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)
