#!/usr/bin/env python3
"""word_check.py: the WORD GATE for the AGMM Hail Mary shorts (27 Oct 2026, Sam 18:15: "it just goes silent in some areas and forgets to pronounce
the words and pronounce them correctly").

CALIBRATED 2 NOV 2026 (Sam/coordinator, after H23, H24, H12 and H28 good takes failed the gate):
  - silence inside a line: up to 0.9 s PASSES, 0.9 to 1.2 s is a WARN, over 1.2 s FAILS (hook line too; the old "one hook pause" count is gone). The turn line's pre-name breath limit (0.30 s) is unchanged.
  - agmm.co.uk / AGMM.co.UK / A.G.M.M. co.uk / "agmm dot co dot uk" in any spelling is expanded to a g m m dot co dot u k before alignment, on both sides.
  - numbers are collapsed to one value token on both sides: 1999 = nineteen ninety nine, $125 million = a hundred and twenty five million dollars, 4,000 = four thousand,
    1931 = nineteen thirty one. Currency words (pound/pounds/dollar/dollars) are equal. British and US spellings are equal (realised/realized, draughtsman/draftsman, millimetres/millimeters, grey/gray, colour/color ...).
  - four/for and two/to/too count as the right word only when Whisper's word probability is 0.9 or above.
  - a right word with a low probability: under 0.15 FAILS, 0.15 to 0.5 is a WARN (letters and "AI": FAIL under 0.05, WARN under 0.30); more than 2 WARN words in one line FAIL.
  - "A.I." / "A I" / "AI" / "ai" are one word on both sides.
  - a failing mix paragraph carries unmatched_heard (the Whisper words nobody matched) so a merged token like "agmm.co.uk" can be seen in words.json.
  Missing, replaced and extra words, silent words and the reveal / CTA checks are unchanged. Backup: word_check.py.bak-2026-11-02.

One file, three jobs:

  run       (GitHub Actions only: needs numpy, soundfile, faster-whisper; NO media processing on the Mac)
            python3 word_check.py run --script script.json --film H05 --pack DIR [--tag RUN:LABEL] [--pick P1=3,P5=1] --out words.json
            python3 word_check.py run --script script.json --film H05 --mix final.wav|mp4 --out words.json
            Whisper large-v3 (no prompt, so it cannot be steered towards the script) transcribes every take / the mix with word times and word
            probabilities; the heard words are aligned to the script words; the energy of the audio finds silences. words.json is written.
            It carries NUMBERS plus the heard text of FAILING words only (the repo is public; the script text stays in the private repo).
  read      python3 word_check.py read words.json [--script script.json] [--p-min 0.55 ...]   (local, stdlib only; re-applies the thresholds)
  pick      python3 word_check.py pick words.json [words2.json ...]    best PASSING take per line (hook: nearest 185-195 wpm)
  --selftest  fixtures only, no audio

FAIL conditions for a line (a take, or a paragraph of the final mix):
  1. a script word is missing, out of order or replaced (numbers are normalised both ways: "33,000" = "thirty three thousand", "a hundred" = "100")
  2. a heard word that is not in the script (stray tail, "thank you", a repeated word)
  3. a heard word with probability under 0.15 (0.15 to 0.5 only warns; letters / AI under 0.05)
  4. silence inside a spoken line longer than 1.2 s (0.9 to 1.2 s only warns; was 0.6 s, hook 0.7 s)
  5. the reveal not heard as AGM (not AGMM, AGN, AKGN ...)
  6. the CTA not heard as A G M M dot co dot U K
  7. the turn line: more than 0.3 s of silence immediately before the reveal word (one breath)
  8. a word whose own audio is silent (Whisper wrote it, the voice never said it)
The hook pace (words / first-voiced..last-voiced, "span wpm") is reported as hook_wpm; the pick wants 185-195.
Python 3.9 compatible.
"""
import argparse, difflib, glob, hashlib, json, os, re, sys

VERSION = 1
THR = {"p_min": 0.50,            # lowest accepted Whisper word probability (calibration in VOICE-V3.md: 0.50 fails ~25% of otherwise matching takes; every known bad H05 take fails on replaced words at any value)
       "p_min_letter": 0.30,     # the spelled letters (A G M, A G M M, U K) are one Whisper token each and read lower
       "p_fail": 0.15,           # 2 Nov: a right word under this FAILS; between this and p_min it is a WARN
       "p_fail_letter": 0.05,    # same for the spelled letters and AI
       "warn_words_max": 2,      # more WARN-band words than this in one line FAIL
       "p_homophone": 0.90,      # four/for, two/to/too count only at this probability or above
       "sil_max": 0.90,          # 2 Nov: silence inside a line up to this passes (was 0.60 / hook 0.70)
       "sil_warn_max": 1.20,     # 0.90 to 1.20 WARN, over this FAILS
       "breath_max": 0.30,       # silence right before the reveal word
       "word_silent_db": 42.0,   # a heard word whose loudest 20 ms is this far under the line's p95 level was not spoken (30 flagged quiet "then")
       "hook_lo": 185.0, "hook_hi": 195.0}

# ---------------------------------------------------------------- text normalisation
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
NUMW = set(ONES + [t for t in TENS if t] + ["hundred", "thousand", "million", "billion"])
SPELL = {"metres": "metres", "meters": "metres", "metre": "metre", "meter": "metre", "kilometers": "kilometres", "kilometre": "kilometres", "kilometres": "kilometres",
         "kilometer": "kilometre", "okay": "ok", "tonnes": "tonnes", "tons": "tonnes", "ton": "tonne", "centre": "center", "grey": "gray", "colour": "color",
         "chord": "cord", "rose": "rows", "plow": "plough", "kilograms": "kilos", "kilogram": "kilo", "kgs": "kilos", "kg": "kilo",
         "pound": "pounds", "dollar": "dollars", "$dollars": "dollars"}
# 2 Nov: British and US spellings are the same word (US form is canonical)
_BRIT = {}
for _us, _uk in (("realized", "realised"), ("realize", "realise"), ("realizing", "realising"), ("realization", "realisation"), ("organized", "organised"), ("organize", "organise"),
                 ("organizing", "organising"), ("organization", "organisation"), ("organizations", "organisations"), ("recognized", "recognised"), ("recognize", "recognise"),
                 ("analyzed", "analysed"), ("analyze", "analyse"), ("analyzing", "analysing"), ("apologize", "apologise"), ("specialized", "specialised"), ("prioritize", "prioritise"),
                 ("prioritized", "prioritised"), ("optimize", "optimise"), ("optimized", "optimised"), ("optimizing", "optimising"), ("minimize", "minimise"), ("maximize", "maximise"),
                 ("standardized", "standardised"), ("summarize", "summarise"), ("summarized", "summarised"), ("customize", "customise"), ("customized", "customised"),
                 ("capitalize", "capitalise"), ("emphasize", "emphasise"), ("criticize", "criticise"), ("authorized", "authorised"), ("utilize", "utilise"), ("visualize", "visualise"),
                 ("draftsman", "draughtsman"), ("draftsmen", "draughtsmen"), ("millimeters", "millimetres"), ("millimeter", "millimetre"), ("centimeters", "centimetres"),
                 ("centimeter", "centimetre"), ("color", "colour"), ("colors", "colours"), ("colored", "coloured"), ("favor", "favour"), ("favorite", "favourite"),
                 ("behavior", "behaviour"), ("honor", "honour"), ("neighbor", "neighbour"), ("labor", "labour"), ("harbor", "harbour"), ("flavor", "flavour"), ("humor", "humour"),
                 ("liter", "litre"), ("liters", "litres"), ("theater", "theatre"), ("fiber", "fibre"), ("caliber", "calibre"), ("defense", "defence"), ("offense", "offence"),
                 ("catalog", "catalogue"), ("jewelry", "jewellery"), ("traveled", "travelled"), ("traveling", "travelling"), ("canceled", "cancelled"),
                 ("modeling", "modelling"), ("modeled", "modelled"), ("labeled", "labelled"), ("aluminum", "aluminium"), ("skeptical", "sceptical"), ("mold", "mould"), ("sulfur", "sulphur")):
    _BRIT[_uk] = _us
for _uk, _us in _BRIT.items(): SPELL.setdefault(_uk, _us)
SPELL.update({"grey": "gray", "colour": "color", "centre": "center"})


def say(n):
    if n < 20: return ONES[n]
    if n < 100: return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000: return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + say(n % 100))
    if n < 1000000: return say(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + say(n % 1000))
    if n < 1000000000: return say(n // 1000000) + " million" + ("" if n % 1000000 == 0 else " " + say(n % 1000000))
    return str(n)


def _num(m):
    t = m.group(0).replace(",", "")
    if "." in t:
        a, b = t.split(".", 1)
        return " " + say(int(a or 0)) + " point " + " ".join(ONES[int(c)] for c in b) + " "
    return " " + say(int(t)) + " "


def _word_tokens(raw, mode="auto"):
    """one whitespace word -> canonical tokens. mode 'reveal': AGMM = A G M; 'cta': AGMM = A G M M."""
    t = (raw or "").lower().replace("\u2019", "'").replace("&", " and ")
    if re.sub(r"[^a-z0-9]", "", t) in ("agmmcouk", "agmmdotcodotuk", "agmmdotcouk", "agmmcodotuk"):      # 2 Nov: Whisper writes the domain as ONE token (agmm.co.uk, AGMM.co.UK, A.G.M.M.co.uk)
        return ["a", "g", "m", "m", "dot", "co", "dot", "u", "k"]
    pound = "\u00a3" in t
    dollar = "$" in t
    t = t.replace("\u00a3", " ").replace("$", " ")
    t = re.sub(r"(?<=[a-z0-9])\.(?=[a-z]{2,}\b)", " dot ", t)            # agmm.co.uk -> agmm dot co dot uk
    t = re.sub(r"^\.(?=[a-z]{2,}\b)", "dot ", t)                          # ".co" ".uk" (Whisper writes "AGMM .co .uk")
    t = re.sub(r"\d[\d,]*(?:\.\d+)?", _num, t)
    t = re.sub(r"[^a-z' ]", " ", t.replace("-", " "))
    out = []
    for w in t.split():
        w = w.strip("'")
        if not w: continue
        if w.endswith("'s") and len(w) > 3 and w not in ("it's", "that's", "he's", "she's", "there's", "what's", "let's", "here's", "who's"): w = w[:-2] + "s"      # guy's == guys (same sound)
        if w == "agmm": out += (["a", "g", "m"] if mode == "reveal" else ["a", "g", "m", "m"])
        elif w == "agm": out += ["a", "g", "m"]
        elif w == "uk": out += ["u", "k"]
        else: out.append(SPELL.get(w, w))
    if pound and out: out.append("\u00a3pounds")
    if dollar and out: out.append("$dollars")
    return out


CUR = {"\u00a3pounds": "pounds", "$dollars": "dollars"}
MULT = ("million", "thousand", "billion")
NUMSET = set(ONES + [t for t in TENS if t] + ["hundred", "thousand", "million", "billion"])


def _card(ws):
    """number words -> integer, or None if the words are not one valid cardinal ('nineteen ninety nine' is not)."""
    total, grp, lastm = 0, 0, 10 ** 12
    for w in ws:
        if w in ONES:
            v = ONES.index(w); low = grp % 100
            if v < 10 and not (low == 0 or (low >= 20 and low % 10 == 0)): return None
            if v >= 10 and low != 0: return None
            if v == 0 and len(ws) > 1: return None
            grp += v
        elif w in TENS:
            if grp % 100 != 0: return None
            grp += TENS.index(w) * 10
        elif w == "hundred":
            if not (0 < grp < 100): return None
            grp *= 100
        else:
            m = {"thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9}[w]
            if grp == 0 or m >= lastm: return None
            total += grp * m; grp = 0; lastm = m
    return total + grp


def _year(ws):
    """'nineteen ninety nine' = 1999, 'twenty twenty' = 2020, 'nineteen hundred' = 1900"""
    if len(ws) < 2: return None
    a = _card(ws[:1])
    if a is None or not (11 <= a <= 20): return None
    rest = ws[1:]
    if rest == ["hundred"]: return a * 100
    if any(w in ("hundred", "thousand", "million", "billion") for w in rest): return None
    r = _card(rest)
    return a * 100 + r if r is not None and 0 <= r < 100 else None


def _collapse(toks):
    """runs of number words -> one '#value' token (owner = first word), so 1999 = nineteen ninety nine and 125 million = a hundred and twenty five million"""
    out, i, n = [], 0, len(toks)
    while i < n:
        if toks[i][0] not in NUMSET: out.append(toks[i]); i += 1; continue
        j = i
        while j < n and toks[j][0] in NUMSET: j += 1
        run = [w for w, _ in toks[i:j]]; k0 = i
        while run:
            for k in range(len(run), 0, -1):
                v = _card(run[:k])
                if v is None: v = _year(run[:k])
                if v is not None:
                    out.append(("#%d" % v, toks[k0][1])); run = run[k:]; k0 += k; break
            else:
                out.append((run[0], toks[k0][1])); run = run[1:]; k0 += 1
        i = j
    return out


def _cross(toks):
    """toks: [(tok, owner)] -> cross-word fixes: 'a i' = ai, 'a hundred' = 'one hundred', 'hundred and fifty' = 'hundred fifty', GBP1 million = a million pounds, then number runs collapsed"""
    res = []
    for i, (w, o) in enumerate(toks):
        if w == "a" and i + 1 < len(toks) and toks[i + 1][0] == "i": res.append(("ai", o)); toks[i + 1] = ("", o); continue
        if w == "": continue
        if w == "a" and i + 1 < len(toks) and toks[i + 1][0] in ("hundred", "thousand", "million"): w = "one"
        if w == "and" and res and res[-1][0] in NUMW and i + 1 < len(toks) and toks[i + 1][0] in NUMW: continue
        if w == "co" and i + 1 < len(toks) and toks[i + 1][0] == "dot" and (not res or res[-1][0] != "dot"): res.append(("dot", o))      # "AGMM" "co.uk" -> agmm dot co dot uk
        res.append((w, o))
    fin, pend = [], None
    for i, (w, o) in enumerate(res):
        if w in CUR:
            if i + 1 < len(res) and res[i + 1][0] in MULT: pend = CUR[w]; continue      # the currency word goes after the multiplier that follows
            fin.append((CUR[w], o)); continue
        fin.append((w, o))
        if pend and w in MULT: fin.append((pend, o)); pend = None
    return _collapse(fin)


def norm_tokens(text, mode="auto", owners=False):
    toks = []
    for i, raw in enumerate((text or "").split()):
        for tk in _word_tokens(raw, mode): toks.append((tk, i))
    toks = _cross(toks)
    return toks if owners else [t for t, _ in toks]


def para_class(key, text, first_key):
    toks = norm_tokens(text, "cta")
    if "dot" in toks and "co" in toks: return "cta"
    if re.search(r"agmm|a\W+g\W+m\b", (text or "").lower()): return "reveal"
    if key == first_key: return "hook"
    return "story"


def expected_tokens(text, cls):
    return norm_tokens(text, "cta" if cls == "cta" else "reveal", owners=True)


def heard_tokens(words):
    """words: [{t,s,e,p}] -> [{tok, wi}] expanded; wi = index of the Whisper word."""
    toks = []
    for i, w in enumerate(words):
        for tk in _word_tokens(w["t"], "cta" if re.search(r"agmm", w["t"].lower()) else "auto"): toks.append((tk, i))
    return [{"tok": t, "wi": o} for t, o in _cross(toks)]


# ---------------------------------------------------------------- alignment and judging
LETTERS = set("a g m u k ai".split())


HOMOPHONES = {frozenset(x) for x in (("#4", "for"), ("#2", "to"), ("#2", "too"))}


def align(exp, heard):
    """exp: list of tokens; heard: list of {tok, wi}. Returns (exp_status, extra_heard_idx): exp_status[i] = ('ok'|'rep'|'miss', heard_idx|None)."""
    sm = difflib.SequenceMatcher(None, exp, [h["tok"] for h in heard], autojunk=False)
    st = [("miss", None)] * len(exp)
    extras = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1): st[i1 + k] = ("ok", j1 + k)
        elif tag == "replace":
            for k in range(i2 - i1): st[i1 + k] = ("rep", j1 + min(k, j2 - j1 - 1))
        elif tag == "insert":
            extras.extend(range(j1, j2))
    return st, extras


def judge(take, thr=None):
    """take: dict as written by run mode. Returns {'pass': bool, 'reasons': [...] , 'flags': {...}}"""
    T = dict(THR); T.update(thr or {})
    why = []
    warn = []
    items = take.get("exp") or []
    cls = take.get("cls", "story")
    miss = [it for it in items if it["st"] == "miss"]
    rep = [it for it in items if it["st"] == "rep"]
    if miss: why.append("missing %s" % ", ".join("w%d%s" % (m["w"], ("(%s)" % m["x"]) if m.get("x") else "") for m in miss[:6]))
    if rep: why.append("replaced %s" % ", ".join("w%d->%r" % (m["w"], m.get("heard", "?")) for m in rep[:6]))
    ex = take.get("extras") or []
    if ex: why.append("extra heard %s" % ", ".join(repr(e.get("heard")) for e in ex[:5]))
    low, shaky = [], []
    for k, it in enumerate(items):
        if it["st"] != "ok": continue
        if k == 0: continue      # Whisper's probability for the very first word of a clip is unstable (same audio: 0.84 on one runner, 0.08 on another); a garbled first word shows as replaced/missing instead
        lim = T["p_min_letter"] if it.get("letter") else T["p_min"]
        fl = T["p_fail_letter"] if it.get("letter") else T["p_fail"]
        if it.get("p") is None: continue
        if it["p"] < fl: low.append("w%d p=%.2f" % (it["w"], it["p"]))
        elif it["p"] < lim: shaky.append("w%d p=%.2f" % (it["w"], it["p"]))
    if low: why.append("low confidence %s" % ", ".join(low[:6]))
    if len(shaky) > T["warn_words_max"]: why.append("several shaky words (p under %.2f) %s" % (T["p_min"], ", ".join(shaky[:6])))
    elif shaky: warn.append("shaky word (right word, low p) %s" % ", ".join(shaky))
    sil = take.get("silences") or []
    bad = [s for s in sil if s["d"] > T["sil_warn_max"]]
    longs = [s for s in sil if T["sil_max"] < s["d"] <= T["sil_warn_max"]]
    if bad: why.append("silence inside the line %s s" % ", ".join("%.2f@%.2f" % (s["d"], s["s"]) for s in bad[:4]))
    if longs: warn.append("long pause %s s" % ", ".join("%.2f@%.2f" % (s["d"], s["s"]) for s in longs[:4]))
    ws = [it for it in items if it["st"] == "ok" and it.get("rel") is not None and it["rel"] < -T["word_silent_db"]]
    if ws: why.append("word not spoken (silent audio) %s" % ", ".join("w%d" % it["w"] for it in ws[:5]))
    br = take.get("breath")
    if cls == "reveal" and br is not None and br > T["breath_max"]: why.append("%.2f s of silence before the name (limit %.2f)" % (br, T["breath_max"]))
    if cls == "reveal" and not take.get("reveal_ok", False): why.append("reveal not heard as A G M")
    if cls == "cta" and not take.get("cta_ok", False): why.append("CTA not heard as A G M M dot co dot U K")
    flags = {"hook_wpm": take.get("hook_wpm"), "hook_pace_ok": (T["hook_lo"] <= take["hook_wpm"] <= T["hook_hi"]) if (cls == "hook" and take.get("hook_wpm")) else None}
    return {"pass": not why, "reasons": why, "warnings": warn, "flags": flags}


# ---------------------------------------------------------------- build one take record from heard words + audio facts
def build_take(key, text, cls, words, silences, first_v, last_v, levels=None, dur=None, thr=None):
    """words: Whisper words [{t,s,e,p}]; silences: [{s,e}] already inside the spoken span; levels: per Whisper word relative peak dB (or None)."""
    eo = expected_tokens(text, cls)
    exp = [t for t, _ in eo]
    owner = [o for _, o in eo]
    ht = heard_tokens(words)
    st, extra_idx = align(exp, ht)
    T_ = dict(THR); T_.update(thr or {})
    for i, (s_, hidx) in enumerate(st):      # 2 Nov: four/for, two/to/too are the right word only when Whisper is sure (p >= 0.9)
        if s_ == "rep" and hidx is not None and frozenset((exp[i], ht[hidx]["tok"])) in HOMOPHONES and float(words[ht[hidx]["wi"]].get("p", 1.0)) >= T_["p_homophone"]:
            st[i] = ("ok", hidx)
    items = []
    for i, (s_, hidx) in enumerate(st):
        it = {"w": owner[i], "st": s_, "letter": exp[i] in LETTERS}
        if hidx is not None:
            w = words[ht[hidx]["wi"]]
            it.update({"p": round(float(w.get("p", 1.0)), 3), "s": round(w["s"], 3), "e": round(w["e"], 3)})
            if levels is not None: it["rel"] = levels[ht[hidx]["wi"]]
            if s_ != "ok": it["heard"] = w["t"]
        if s_ == "miss": it["x"] = exp[i] if (exp[i] in LETTERS or exp[i] in ("dot", "co")) else ""
        items.append(it)
    extras = []
    seen = set()
    for j in extra_idx:
        wi = ht[j]["wi"]
        if wi in seen: continue
        seen.add(wi)
        # an expanded token whose word is partly matched is not extra
        if any(st[i][1] is not None and ht[st[i][1]]["wi"] == wi for i in range(len(st))): continue
        w = words[wi]
        extras.append({"heard": w["t"], "s": round(w["s"], 3), "e": round(w["e"], 3), "p": round(float(w.get("p", 1.0)), 3)})
    rec = {"cls": cls, "n_words": len((text or "").split()), "exp": items, "extras": extras,
           "silences": [{"s": round(r["s"], 3), "d": round(r["e"] - r["s"], 3)} for r in silences], "first_v": round(first_v, 3), "last_v": round(last_v, 3),
           "dur": dur, "heard_n": len(words)}
    span = max(1e-6, last_v - first_v)
    rec["hook_wpm"] = round(len((text or "").split()) / span * 60.0, 1)
    # reveal / cta evidence
    if cls == "reveal":
        k = [i for i in range(len(exp) - 2) if exp[i:i + 3] == ["a", "g", "m"]]
        last = k[-1] if k else None
        wtxt = ""
        if last is not None and st[last][1] is not None: wtxt = words[ht[st[last][1]]["wi"]]["t"].lower()
        rec["reveal_ok"] = bool(last is not None and all(items[last + d]["st"] == "ok" for d in range(3)) and not re.search(r"a\W*g\W*m\W*m", wtxt) and not any(re.search(r"\bm\b|agmm", e["heard"].lower()) for e in extras))
        if last is not None and items[last].get("s") is not None:
            ns = items[last]["s"]
            cand = [r for r in silences if r["e"] >= ns - 0.30 and r["e"] <= ns + 0.15]
            rec["breath"] = round(max([r["e"] - r["s"] for r in cand] or [0.0]), 3)
            rec["name_s"] = ns
    if cls == "cta":
        tail = ["a", "g", "m", "m", "dot", "co", "dot", "u", "k"]
        rec["cta_ok"] = all(it["st"] == "ok" for it in items[-9:]) and exp[-9:] == tail and not extras
    rec["judge"] = judge(rec, thr)
    return rec


# ---------------------------------------------------------------- Actions side: audio facts and Whisper
def audio_facts(path):
    import numpy as np, soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    x = x if x.ndim == 1 else x.mean(axis=1)
    hop = int(sr * 0.010); win = int(sr * 0.020)
    n = max(1, (len(x) - win) // hop)
    db = np.empty(n, dtype="float32")
    for i in range(n):
        seg = x[i * hop:i * hop + win]
        db[i] = 20.0 * np.log10(float(np.sqrt(np.mean(seg * seg))) + 1e-9)
    return db, 0.010, len(x) / float(sr)


def silence_runs(db, step, rel_floor=35.0, abs_floor=-62.0, min_s=0.20):
    """-> (runs[{s,e}] strictly inside the voiced span, first_voiced_s, last_voiced_s, ref_db)"""
    import numpy as np
    live = db[db > -75]
    ref = float(np.percentile(live, 95)) if len(live) else -30.0
    thr = max(ref - rel_floor, abs_floor)
    sil = (db < thr)
    # close voiced blips shorter than 40 ms inside silence
    i = 0; N = len(sil)
    while i < N:
        if not sil[i]:
            j = i
            while j < N and not sil[j]: j += 1
            if j - i < 4 and i > 0 and j < N: sil[i:j] = True
            i = j
        else: i += 1
    voiced = np.where(~sil)[0]
    if len(voiced) == 0: return [], 0.0, 0.0, ref
    fv, lv = int(voiced[0]), int(voiced[-1])
    runs = []; i = fv
    while i <= lv:
        if sil[i]:
            j = i
            while j <= lv and sil[j]: j += 1
            if (j - i) * step >= min_s: runs.append({"s": i * step, "e": j * step})
            i = j
        else: i += 1
    return runs, fv * step, (lv + 1) * step, ref


def word_levels(db, step, ref, words):
    out = []
    for w in words:
        a, b = w["s"], w["e"]
        c = (b - a) * 0.2
        i0 = max(0, int((a + c) / step)); i1 = max(i0 + 1, int((b - c) / step))
        seg = db[i0:i1] if i1 > i0 else db[i0:i0 + 1]
        out.append(round(float(seg.max()) - ref, 1) if len(seg) else None)
    return out


def merge_digit_fragments(words):
    """Whisper splits "638,000" into "638" + ",000" (or "33" + "000"): one word, the lower probability."""
    merged = []
    for w in words:
        if merged and (re.match(r"^[.,]\d", w["t"]) or re.match(r"^\d{3}$", w["t"])) and re.search(r"\d$", merged[-1]["t"]):
            if re.match(r"^\d{3}$", w["t"]): w = dict(w, t="," + w["t"])
            merged[-1] = dict(merged[-1], t=merged[-1]["t"] + w["t"], e=w["e"], p=min(merged[-1]["p"], w["p"]))
        else: merged.append(w)
    return merged


_MODEL = {}


def whisper_words(path, model_name="large-v3"):
    from faster_whisper import WhisperModel
    if model_name not in _MODEL:
        _MODEL[model_name] = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    m = _MODEL[model_name]
    segs, info = m.transcribe(path, language="en", beam_size=5, word_timestamps=True, vad_filter=False, condition_on_previous_text=False, temperature=0.0)
    words = []
    for s in segs:
        for w in (s.words or []):
            t = w.word.strip()
            if t: words.append({"t": t, "s": float(w.start), "e": float(w.end), "p": float(w.probability)})
    return merge_digit_fragments(words)


def load_script(path):
    d = json.load(open(path))
    paras = [(p.get("key") or p.get("id"), p["text"]) for p in d["paragraphs"]]
    return d, paras


def run_pack(a):
    d, paras = load_script(a.script)
    first = paras[0][0]
    ptext = dict(paras)
    picks = {}
    for kv in (a.pick or "").split(","):
        if "=" in kv: k, v = kv.split("="); picks[k.strip()] = int(v)
    takes = {}
    tag = a.tag or "pack"
    for wav in sorted(glob.glob(os.path.join(a.pack, "*.wav"))):
        stem = os.path.basename(wav)[:-4]
        m = re.match(r"^(P\d+)\.t(\d+)$", stem)
        if not m or m.group(1) not in ptext: continue
        key, tk = m.group(1), int(m.group(2))
        if a.keys and key not in a.keys.split(","): continue
        if a.takes and tk > a.takes: continue
        text = ptext[key]; cls = para_class(key, text, first)
        words = whisper_words(wav, a.model)
        db, step, dur = audio_facts(wav)
        runs, fv, lv, ref = silence_runs(db, step)
        lv_ = word_levels(db, step, ref, words)
        rec = build_take(key, text, cls, words, runs, fv, lv, lv_, round(dur, 3))
        rec.update({"key": key, "take": tk, "tag": tag, "stem": stem, "ref_db": round(ref, 1)})
        takes["%s/%s" % (tag, stem)] = rec
        print("%s %s %s %s%s" % (tag, stem, cls, "PASS" if rec["judge"]["pass"] else "FAIL", (" wpm=%s" % rec["hook_wpm"]) if cls == "hook" else ""), flush=True)
    return takes, picks


def run_mix(a):
    d, paras = load_script(a.script)
    first = paras[0][0]
    path = a.mix
    if not path.lower().endswith(".wav"):
        import subprocess
        w = path + ".16k.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path, "-ar", "16000", "-ac", "1", w], check=True)
        path = w
    words = whisper_words(path, a.model)
    stem_runs = None
    if a.stem:                    # the voice stem (no music): energy silences are measurable there
        db, step, _dur = audio_facts(a.stem)
        stem_runs, _fv, _lv, _ref = silence_runs(db, step)
    exp_all, own, classes = [], [], {}
    for key, text in paras:
        cls = para_class(key, text, first); classes[key] = cls
        for tk, _o in expected_tokens(text, cls):
            exp_all.append(tk); own.append(key)
    words_all = words
    ht = heard_tokens(words)
    st, extra_idx = align(exp_all, ht)
    takes = {}
    for key, text in paras:
        cls = classes[key]
        idx = [i for i, o in enumerate(own) if o == key]
        ms = [ht[st[i][1]]["wi"] for i in idx if st[i][1] is not None]
        pw = words[min(ms):max(ms) + 1] if ms else []
        ps, pe = (pw[0]["s"], pw[-1]["e"]) if pw else (0.0, 0.0)
        if stem_runs is not None: gaps = [r for r in stem_runs if r["s"] >= ps - 0.05 and r["e"] <= pe + 0.05]
        else: gaps = [{"s": pw[i]["e"], "e": pw[i + 1]["s"]} for i in range(len(pw) - 1) if pw[i + 1]["s"] - pw[i]["e"] > 0.2]
        rec = build_take(key, text, cls, pw, gaps, ps, pe, None, None)
        if not rec["judge"]["pass"]:       # the Whisper words nobody matched near this paragraph (so a merged token like agmm.co.uk can be seen); failing lines only
            matched = {ht[st[i][1]]["wi"] for i in range(len(st)) if st[i][1] is not None}
            if cls == "cta": rec["last_words"] = [{"t": w["t"], "s": round(w["s"], 2), "e": round(w["e"], 2), "p": round(float(w["p"]), 2)} for w in words_all[-6:]]      # what Whisper wrote at the very end of the mix (CTA diagnosis)
            rec["unmatched_heard"] = [{"t": w["t"], "s": round(w["s"], 2), "p": round(float(w["p"]), 2)} for wi, w in enumerate(words) if wi not in matched and (not pw or ps - 1.0 <= w["s"] <= pe + 4.0)][:12]
        rec.update({"key": key, "take": 0, "tag": "mix", "stem": key, "source": "mix", "para_start": ps, "para_end": pe, "silence_source": "voice stem energy" if stem_runs is not None else "word gaps"})
        takes["mix/%s" % key] = rec
        print("mix %s %s %s" % (key, cls, "PASS" if rec["judge"]["pass"] else "FAIL"), flush=True)
    return takes, {}


def cmd_run(a):
    d, paras = load_script(a.script)
    takes, picks = run_mix(a) if a.mix else run_pack(a)
    sha = hashlib.sha256(open(a.script, "rb").read()).hexdigest()
    checked = sorted(t for t, r in takes.items() if (r.get("source") == "mix") or (picks.get(r["key"]) == r["take"]))
    out = {"version": VERSION, "model": a.model, "film": a.film, "script_sha256": sha, "source": "mix" if a.mix else "pack", "thresholds": THR, "checked": checked,
           "n_paragraphs": len(paras), "takes": takes}
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out, len(takes), "takes")


# ---------------------------------------------------------------- local side
def judge_file(path, thr=None, only=None):
    """-> (ok, results{take: judge}, d). only = list of take ids; default: d['checked'] or all."""
    d = json.load(open(path))
    ids = only or d.get("checked") or sorted(d["takes"])
    res = {}
    for t in ids:
        rec = d["takes"].get(t)
        res[t] = judge(rec, thr) if rec else {"pass": False, "reasons": ["take not in words.json"], "flags": {}}
    need = d.get("n_paragraphs")
    if d.get("source") == "mix" and need and len(res) < need: res["_coverage"] = {"pass": False, "reasons": ["only %d of %d paragraphs checked" % (len(res), need)], "flags": {}}
    return all(v["pass"] for v in res.values()), res, d


def cmd_read(a):
    thr = {}
    for k in ("p_min", "p_min_letter", "sil_max", "breath_max"):
        if getattr(a, k, None) is not None: thr[k] = getattr(a, k)
    ok, res, d = judge_file(a.words, thr, a.only.split(",") if a.only else None)
    for t, r in sorted(res.items()):
        rec = d["takes"].get(t, {})
        wn = r.get("warnings") or []
        print("%-4s %-22s %s%s%s" % ("FAIL" if not r["pass"] else ("WARN" if wn else "PASS"), t, "; ".join(r["reasons"] + ["WARN: " + x for x in wn]) or "all words heard", (" hook_wpm=%s" % rec.get("hook_wpm")) if rec.get("cls") == "hook" else "", (" unmatched_heard=%s" % [u["t"] for u in rec.get("unmatched_heard", [])]) if (not r["pass"] and rec.get("unmatched_heard")) else ""))
    nw = sum(1 for r in res.values() if r.get("warnings"))
    print("WORDS %s%s" % ("PASS" if ok else "FAIL", (" (%d line(s) with WARN)" % nw) if nw else ""))
    return 0 if ok else 1


def rank_key(rec, prefer=()):
    j = judge(rec)
    ps = [it["p"] for it in rec["exp"] if it.get("p") is not None]
    mean_p = sum(ps) / len(ps) if ps else 0.0
    minp = min(ps) if ps else 0.0
    pen = 0.0
    if rec["cls"] == "hook":
        w = rec.get("hook_wpm") or 0.0
        import math
        pen = 0.0 if THR["hook_lo"] <= w <= THR["hook_hi"] else abs(math.log(190.0 / max(w, 1.0)))     # distance (log) to the 190 wpm target: the cheapest retempo wins
        if not (0.8 <= 190.0 / max(w, 1.0) <= 1.3): pen += 5.0                                         # needs more than a 0.8-1.3 tempo change: a fresh take is better
    pref = 0 if (not prefer or any(x in rec.get("tag", "") for x in prefer)) else 1
    return (not j["pass"], round(pen, 2), pref, -round(minp, 2), -round(mean_p, 3))


def cmd_pick(a):
    allt = {}
    for p in a.files:
        d = json.load(open(p))
        for t, r in d["takes"].items():
            if r.get("source") == "mix": continue
            if any(x and x in t for x in a.exclude.split(",")): continue
            allt[t] = r
    by = {}
    for t, r in allt.items(): by.setdefault(r["key"], []).append((rank_key(r, tuple(x for x in a.prefer.split(",") if x)), t, r))
    for key in sorted(by):
        rows = sorted(by[key], key=lambda x: x[0])
        n_pass = sum(1 for x in rows if not x[0][0])
        best = rows[0]
        print("%s: %d takes, %d pass; best %s %s%s" % (key, len(rows), n_pass, best[1], "PASS" if not best[0][0] else "FAIL(" + "; ".join(best[2]["judge"]["reasons"])[:80] + ")",
                                                       (" wpm=%s" % best[2].get("hook_wpm")) if best[2]["cls"] == "hook" else ""))
    if a.spec:
        spec, missing = [], []
        for key in sorted(by):
            best = sorted(by[key], key=lambda x: x[0])[0]
            if best[0][0]: missing.append(key); continue
            run, label = best[2]["tag"].split(":", 1)
            row = {"run": int(run), "label": label, "key": key, "take": best[2]["take"], "as": 1}
            w = best[2].get("hook_wpm")
            if best[2]["cls"] == "hook" and w and not (THR["hook_lo"] <= w <= THR["hook_hi"]):
                if not (0.8 <= 190.0 / w <= 1.3): missing.append(key + "(hook %.0f wpm, tempo change too big)" % w); continue
                row["post"] = {"tempo": round(190.0 / w, 4), "max_gap": 9}      # pitch-preserving atempo to 190 wpm; gaps kept (the hook pause stays)
            spec.append(row)
        json.dump({"spec": spec, "no_passing_take": missing}, open(a.spec, "w"), indent=1)
        print("spec", len(spec), "lines; no passing take for", missing)
    return 0


# ---------------------------------------------------------------- selftest (fixtures only)
def _w(text, s0=0.0, p=0.95, dt=0.28, gaps=None):
    out = []; t = s0
    for i, w in enumerate(text.split()):
        g = (gaps or {}).get(i, 0.0)
        t += g
        pp = p[i] if isinstance(p, list) else p
        out.append({"t": w, "s": t, "e": t + dt, "p": pp}); t += dt + 0.02
    return out


def selftest():
    bad = []
    def chk(name, cond, detail=""):
        if not cond: bad.append(name + " " + detail)
        print("%-5s %s %s" % ("ok" if cond else "FAIL", name, detail))
    HOOK = "What if thirty-three thousand hours of meetings... all happened at once?"
    REV = "So the building gives way, and your last penny drops straight into... AGMM."
    ADS = "An AI consultancy that finds where your business is leaking time and money, then builds the AI to fix it. Faster operations. Bigger margins. Custom systems that run themselves."
    CTA = "Book the audit at agmm dot co dot uk."
    def take(text, cls, heard, sil=(), fv=0.0, lv=None, levels=None):
        w = _w(heard) if isinstance(heard, str) else heard
        lv = lv if lv is not None else w[-1]["e"]
        return build_take("P", text, cls, w, [{"s": a, "e": b} for a, b in sil], fv, lv, levels)
    # normalisation both ways
    chk("num both ways", norm_tokens("33,000 hours") == norm_tokens("thirty-three thousand hours") == ["#33000", "hours"])
    chk("a hundred = 100", norm_tokens("a hundred times") == norm_tokens("100 times") == ["#100", "times"])
    chk("hundred and fifty", norm_tokens("three hundred and fifty six") == norm_tokens("356"))
    chk("pounds sign", norm_tokens("a million pounds") == norm_tokens("£1 million"))
    chk("agmm.co.uk", norm_tokens("AGMM.co.uk.", "cta") == norm_tokens("A. G. M. M. dot co dot U K.", "cta") == "a g m m dot co dot u k".split())
    # GOOD takes pass
    r = take(HOOK, "hook", "What if 33,000 hours of meetings all happened at once?", sil=[(1.4, 1.95)])
    chk("good hook passes (numbers normalised, 0.55 s pause)", r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(REV, "reveal", "So the building gives way and your last penny drops straight into AGM")
    chk("good reveal passes", r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(CTA, "cta", "Book the audit at AGMM.co.uk")
    chk("good CTA passes (AGMM.co.uk heard)", r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(ADS, "story", "An AI consultancy that finds where your business is leaking time and money then builds the A.I. to fix it Faster operations Bigger margins Custom systems that run themselves")
    chk("good ad passes", r["judge"]["pass"], str(r["judge"]["reasons"]))
    # KNOWN FAILURES (H05, 27 Oct)
    r = take(REV, "reveal", "So the building gives way and your last penny drops straight into AGN")
    chk("H05 'AGN' fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(REV, "reveal", "So the building gives way and your last penny drops straight into AKGN")
    chk("H05 'AKGN' fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(REV, "reveal", "So the building gives way and your last penny drops straight into AGMM")
    chk("reveal heard AGMM fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(ADS, "story", "An AI consultancy that finds where your business is leaking time and money then builds the AI to fix it After operations Bigger margins Custom systems that run themselves")
    chk("H05 'After operations' fails", not r["judge"]["pass"] and any("replaced" in x or "missing" in x for x in r["judge"]["reasons"]), str(r["judge"]["reasons"]))
    r = take(ADS, "story", "An AI consultancy that finds where your business is leaking time and money then builds the AI to fix it Faster operations Bigger margins Custom systems have run themselves")
    chk("H05 'have run themselves' fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(CTA, "cta", "Book the audit at AGM.com")
    chk("H05 CTA 'AGM.com' fails", not r["judge"]["pass"] and not r["cta_ok"], str(r["judge"]["reasons"]))
    r = take(CTA, "cta", "Book the audit at AGM dot co dot UK")
    chk("CTA with 3 letters fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    # other fail conditions
    r = take(HOOK, "hook", "What if 33,000 hours of meetings all happened at once Thank you")
    chk("stray tail fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(HOOK, "hook", "What if 33,000 hours meetings all happened at once")
    chk("missing word fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(HOOK, "hook", "What if all happened at once 33,000 hours of meetings")
    chk("out of order fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    ps = [0.95] * 11; ps[4] = 0.31
    r = take(HOOK, "hook", _w("What if 33,000 hours of meetings all happened at once", p=ps))
    chk("p 0.31 right word is a WARN, not a fail (2 Nov)", r["judge"]["pass"] and r["judge"]["warnings"], str(r["judge"]))
    ps[4] = 0.10
    r = take(HOOK, "hook", _w("What if 33,000 hours of meetings all happened at once", p=ps))
    chk("low confidence p 0.10 fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    ps[4] = 0.31; ps[5] = 0.40; ps[6] = 0.45
    r = take(HOOK, "hook", _w("What if 33,000 hours of meetings all happened at once", p=ps))
    chk("three shaky words fail", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(HOOK, "hook", "What if 33,000 hours of meetings all happened at once", sil=[(1.0, 1.65)])
    chk("hook pause 0.65 s passes", r["judge"]["pass"] and not r["judge"]["warnings"], str(r["judge"]["reasons"]))
    r = take(HOOK, "hook", "What if 33,000 hours of meetings all happened at once", sil=[(1.0, 1.85)])
    chk("hook pause 0.85 s passes (2 Nov)", r["judge"]["pass"] and not r["judge"]["warnings"], str(r["judge"]))
    r = take(HOOK, "hook", "What if 33,000 hours of meetings all happened at once", sil=[(1.0, 2.05)])
    chk("hook pause 1.05 s is a WARN", r["judge"]["pass"] and r["judge"]["warnings"], str(r["judge"]))
    r = take(HOOK, "hook", "What if 33,000 hours of meetings all happened at once", sil=[(1.0, 2.35)])
    chk("hook pause 1.35 s fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(ADS, "story", "An AI consultancy that finds where your business is leaking time and money then builds the AI to fix it Faster operations Bigger margins Custom systems that run themselves", sil=[(2.0, 2.85)])
    chk("0.85 s silence mid story passes", r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(ADS, "story", "An AI consultancy that finds where your business is leaking time and money then builds the AI to fix it Faster operations Bigger margins Custom systems that run themselves", sil=[(2.0, 3.3)])
    chk("1.3 s silence mid story fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(REV, "reveal", "So the building gives way and your last penny drops straight into AGM", sil=[(3.0, 3.45)])
    nm = r.get("name_s", 0)
    r = take(REV, "reveal", _w("So the building gives way and your last penny drops straight into", 0.0) + [{"t": "AGM", "s": 4.6, "e": 5.0, "p": 0.9}], sil=[(4.1, 4.6)])
    chk("0.5 s before the name fails (one breath)", not r["judge"]["pass"] and r["breath"] > 0.3, "breath=%s %s" % (r.get("breath"), r["judge"]["reasons"]))
    lv = [0.0] * 12; lv[6] = -50.0
    r = take(HOOK, "hook", "What if 33,000 hours of meetings all happened at once", levels=lv[:11])
    chk("word with silent audio fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    # pace
    r = take(HOOK, "hook", "What if 33,000 hours of meetings all happened at once", fv=0.0, lv=3.5)
    chk("hook wpm = words/span", abs(r["hook_wpm"] - 11 / 3.5 * 60) < 0.2, str(r["hook_wpm"]))
    mw = merge_digit_fragments([{"t": "33", "s": 0, "e": 1, "p": .9}, {"t": "000", "s": 1, "e": 2, "p": .8}, {"t": "hours", "s": 2, "e": 3, "p": .9}])
    chk("whisper '33' '000' = 33,000", len(mw) == 2 and mw[0]["t"] == "33,000", str(mw))
    chk("guy's == guys", norm_tokens("One guy's still") == norm_tokens("One guys still"))
    # ---- 2 Nov calibration
    CTA2 = "Book the audit at A.G.M.M. dot co dot U K."
    for h in ("Book the audit at agmm.co.uk", "Book the audit at AGMM.co.uk.", "Book the audit at AGMM.co.UK", "Book the audit at AGMM co.uk", "Book the audit at A.G.M.M.co.uk"):
        r = take(CTA2, "cta", h)
        chk("CTA merged token passes: %s" % h, r["judge"]["pass"] and r["cta_ok"], str(r["judge"]["reasons"]))
    r = take(CTA2, "cta", "Book the audit at AGM.co.uk")
    chk("CTA with 3 letters merged still fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    chk("1999 = nineteen ninety nine", norm_tokens("In 1999") == norm_tokens("In nineteen ninety nine"))
    chk("1931 = nineteen thirty one", norm_tokens("1931") == norm_tokens("nineteen thirty one"))
    chk("$125 million", norm_tokens("$125 million") == norm_tokens("a hundred and twenty five million dollars") == norm_tokens("a hundred and twenty five million dollar"))
    chk("4,000 / 10", norm_tokens("4,000 and 10") == norm_tokens("four thousand and ten"))
    chk("US = UK spellings", norm_tokens("realised draughtsman millimetres grey colour") == norm_tokens("realized draftsman millimeters gray color"))
    P1 = "In 1999 he paid $125 million for the grey draughtsman."
    r = take(P1, "story", "In nineteen ninety nine he paid 125 million dollars for the gray draftsman")
    chk("mixed numbers and spellings pass", r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(P1, "story", "In 1999 he paid $125 million for the grey draughtsman")
    chk("digits both sides pass", r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(P1, "story", "In 1998 he paid $125 million for the grey draughtsman")
    chk("wrong year still fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take(P1, "story", "In 1999 he paid $152 million for the grey draughtsman")
    chk("wrong amount still fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take("Take two of them for four", "story", _w("Take to of them for four", p=0.95))
    chk("homophone to/two at p 0.95 passes", r["judge"]["pass"], str(r["judge"]["reasons"]))
    r = take("Take two of them for four", "story", _w("Take to of them for four", p=0.80))
    chk("homophone to/two at p 0.80 fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    for ai in ("A.I.", "AI", "ai", "a.i.", "A I"):
        r = take("builds the AI to fix it", "story", "builds the %s to fix it" % ai)
        chk("AI spelling %r passes" % ai, r["judge"]["pass"], str(r["judge"]["reasons"]))
    ps = [0.99] * 6; ps[2] = 0.063
    r = take("builds the AI to fix it", "story", _w("builds the AI to fix it", p=ps))
    chk("AI at p 0.063 is a WARN (letter-like), not a fail", r["judge"]["pass"] and r["judge"]["warnings"], str(r["judge"]))
    r = take("builds the AI to fix it", "story", "builds the to fix it")
    chk("missing AI still fails", not r["judge"]["pass"], str(r["judge"]["reasons"]))
    print("SELFTEST %s" % ("PASS" if not bad else "FAIL: %s" % bad))
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run"); r.add_argument("--script", required=True); r.add_argument("--film", required=True); r.add_argument("--pack"); r.add_argument("--mix"); r.add_argument("--stem")
    r.add_argument("--tag"); r.add_argument("--pick"); r.add_argument("--keys"); r.add_argument("--takes", type=int); r.add_argument("--model", default="large-v3"); r.add_argument("--out", required=True)
    q = sub.add_parser("read"); q.add_argument("words"); q.add_argument("--only"); q.add_argument("--p-min", dest="p_min", type=float); q.add_argument("--p-min-letter", dest="p_min_letter", type=float)
    q.add_argument("--sil-max", dest="sil_max", type=float); q.add_argument("--breath-max", dest="breath_max", type=float)
    k = sub.add_parser("pick"); k.add_argument("files", nargs="+"); k.add_argument("--spec", help="write the assemble spec [{run,label,key,take,as}] of the best PASSING take per line here"); k.add_argument("--prefer", default="", help="comma list: tags containing one of these rank first among passing takes"); k.add_argument("--exclude", default="", help="comma list of tags (run:label substrings) to ignore")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    if a.cmd == "run": return cmd_run(a)
    if a.cmd == "read": sys.exit(cmd_read(a))
    if a.cmd == "pick": sys.exit(cmd_pick(a))
    ap.print_help()


if __name__ == "__main__":
    main()
