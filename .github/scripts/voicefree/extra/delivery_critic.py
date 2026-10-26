#!/usr/bin/env python3
"""delivery_critic.py -- "listens" to one voiced take and says whether the story, as DELIVERED, could be delivered better.
AUDIO-CRITIC card, root 19 Oct 2026. Sam: "genuine audio listening that the free agents can confirm whether the story audibly told
could be audibly delivered better".

  ~/alfred/yt-engine/.venv/bin/python delivery_critic.py <ID | dir | dir::stem> [--layers signal,text,audio] [--runs 2] [--out F] [--no-cache]
  (heavy: local whisper + Praat. Run through kit/tools/heavy.py in a batch:  kit/tools/heavy.py -- <python> delivery_critic.py S92)

ID resolves in shorts/voice/<ID>, films/voice/<ID>, pilot/voice/<ID> (P1, P2-v2 -> pilot/voice/P2 stem P2-v2), shorts/voice/_backup/<name>.
Needs <dir>/<stem>.wav and <dir>/<stem>.words.json (the assembled take and its word timings, punctuation kept).

THREE LAYERS, each independent, each can fail:
 1. SIGNAL (local, deterministic, no network). Per sentence: wpm, articulation wpm, pause before/after (measured on the audio, not the
    token gaps), pitch range / sd (Praat), terminal fall, key-word stress (numbers, names, turn words: energy, pitch and duration against
    the sentence's other words), stray voiced runs inside mid-sentence gaps and repeated words whisper heard (land-and-restart),
    whisper word probability and words whisper did not hear (mumble), breaths, clicks. Per take: sentence density (tts-sentence-density),
    fragment runs, monotony, hook (first 3 s) against the body, CTA delivery (pace, terminal pitch lift = the S02 signature, fade).
    Output: flags (warn/fail) and a signal verdict PASS / IMPROVE / FAIL. Thresholds are in TH with where each came from.
 2. TEXT JUDGE. The transcript, sentence by sentence, with the numbers above (never the audio), to a healthy free text lane through
    `fleet run reader`, with a rubric. JSON: verdict, clarity by ear, sentences a listener would lose, emphasis/pause placement, hook
    urgency, exactly three re-voiceable fixes. Run --runs times; the layer is "stable" only if every run returns the same verdict.
 3. AUDIO LISTENER (primary when available). The actual voice (whole take <= 75 s, else per paragraph) to Mistral's own API,
    model voxtral-small-2507, chat completions with input_audio, same rubric. Key: MISTRAL_KEY from ~/alfred/.env (read as gen_paragraphs.py
    env_key() does, never printed). Per-call usage (audio seconds, tokens) and an EUR estimate at Mistral's list price are appended to
    tooling/delivery_calibration/audio_calls.jsonl. OpenRouter (voxtral-small-24b-2507) is an optional fallback, OFF unless
    DELIVERY_AUDIO_OR=1. Unavailable (no key, cap reached, error) -> reported, falls back to layers 1+2.
Final verdict: audio if stable, else text if stable and the signal agrees it is not worse, else signal. See combine().

Reuses voice_pace.py (number spelling, syllables, frame energy, gap_profile). Backups of what this card touched: *.bak-2026-10-19-audio.
"""
import argparse, base64, difflib, glob, hashlib, json, os, re, subprocess, sys, time, urllib.error, urllib.request

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the hub update check hung on a dead socket once (voice_lead_check note, 6 Oct)
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voice_pace as vp  # noqa: E402

V2 = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
HOME = os.path.expanduser("~")
PITCH_PY = os.path.join(HOME, "alfred/venvs/pitch/bin/python")
CACHE = os.path.join(HERE, "_dc_cache")
CALIB = os.path.join(HERE, "delivery_calibration")
FLEET = os.path.join(HOME, ".fleet/bin/fleet")
HOP = 0.010

# ------------------------------------------------------------------------------------------------ thresholds
# Every number says where it came from. "pilot" = the approved pilots P1, P2-v2, P3-v2; "calib" = the 19 Oct calibration run
# (delivery_calibration/results.json), set from the approved side's spread, not tuned to the known-bad side.
TH = {
    "dens_lo": 8.0, "dens_hi": 16.0,        # sentences per 100 words, body (tts-sentence-density memory, measured 18 Aug)
    "dens_beat_max": 20.0,                  # no single paragraph above this (same memory)
    "frag_words": 4, "frag_run_max": 2,     # a fragment is <= 4 words; never more than 2 in a row (same memory)
    "wpm_sent_hi": 215.0,                   # one sentence faster than this is rushed (voice_pace: C13 P03 at 210-216 was the reference rush)
    "wpm_sent_lo": 110.0,
    "pause_end_min_ms": 150.0,              # a sentence ending with less than this has no landing
    "kw_stress_e_db": 1.5, "kw_stress_f0_st": 2.0, "kw_stress_dur": 1.30,   # a key word "landed" when ANY of the three clears
    "low_prob": 0.50,                       # whisper word probability under this = a mumble candidate
    "cta_lift_st": 4.0,                     # calib: S02 old CTA +7.0 (Sam-confirmed bad, memory: +6.8); approved max +2.2
    "cta_nofade_db": 0.5,                   # calib: approved max -0.1 (0/18); 5 of 8 "pre-cta" re-voices at >= +1.4
    "fast_syl_rate": 7.9,                   # calib: approved max 7.69 (P1 sentence 12); C13 old take 8.3
}

NUMW = {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion", "trillion"}
TURNW = {"but", "except", "only", "nothing", "never", "none", "without", "instead", "however", "yet"}
NONAME = {"I", "I'm", "I've", "I'll", "I'd", "AI", "UK", "US", "OK", "A", "The", "And", "But", "So", "Then", "It", "It's", "That", "That's",
          "They", "We", "You", "If", "In", "On", "At", "He", "She", "This", "There", "Now", "Here", "What", "When", "Why", "How", "Every",
          "Follow", "Nothing", "None", "Not", "No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"}
ABBR = re.compile(r"^(?:[A-Za-z]\.){2,}$|^(?:Mr|Mrs|Ms|Dr|St|vs|Inc|Ltd|Co|approx|No)\.$")


# ------------------------------------------------------------------------------------------------ resolving
def resolve(spec):
    if "::" in spec:
        d, stem = spec.split("::", 1)
        return os.path.abspath(os.path.expanduser(d)), stem
    cands = []
    if os.path.isdir(os.path.expanduser(spec)):
        cands.append(os.path.abspath(os.path.expanduser(spec)))
    for base in ("shorts/voice", "films/voice", "pilot/voice", "shorts/voice/_backup"):
        cands.append(os.path.join(V2, base, spec))
        pre = re.split(r"[-]", spec)[0]
        cands.append(os.path.join(V2, base, pre))
    for d in cands:
        if not os.path.isdir(d):
            continue
        for stem in (os.path.basename(d), os.path.basename(os.path.dirname(d)), re.split(r"-", os.path.basename(d))[0], spec, re.split(r"-", spec)[0]):
            if os.path.exists(os.path.join(d, stem + ".wav")) and os.path.exists(os.path.join(d, stem + ".words.json")):
                return d, stem
    raise SystemExit("cannot resolve %r to a dir with <stem>.wav + <stem>.words.json" % spec)


def sha_file(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def read_wav(p):
    try:
        return vp.read_wav(p)
    except Exception:
        import soundfile as sf
        a, sr = sf.read(p, always_2d=True)
        return a.mean(axis=1), sr


# ------------------------------------------------------------------------------------------------ cached heavy steps
def cached(key, fn, use_cache=True):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, key + ".json")
    if use_cache and os.path.exists(p):
        return json.load(open(p))
    v = fn()
    tmp = p + ".tmp%d" % os.getpid()
    json.dump(v, open(tmp, "w"))
    os.replace(tmp, p)
    return v


def run_whisper(wav):
    from faster_whisper import WhisperModel
    m = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=2)
    segs, _ = m.transcribe(wav, language="en", word_timestamps=True, beam_size=3, vad_filter=False, condition_on_previous_text=False)
    out = []
    for s in segs:
        for w in s.words or []:
            out.append({"w": w.word.strip(), "s": round(float(w.start), 3), "e": round(float(w.end), 3), "p": round(float(w.probability), 3)})
    return out


def run_praat(wav):
    r = subprocess.run(["nice", "-n", "10", PITCH_PY, os.path.join(HERE, "_dc_praat.py"), wav], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("praat helper failed: " + r.stderr[-300:])
    return json.loads(r.stdout)


# ------------------------------------------------------------------------------------------------ helpers
def nan_arr(lst):
    return np.array([np.nan if x is None else x for x in lst], float)


def sl(arr, t_off, a, b):
    i0 = max(0, int(round((a - t_off) / HOP)))
    i1 = min(len(arr), int(round((b - t_off) / HOP)) + 1)
    return arr[i0:i1] if i1 > i0 else arr[0:0]


def tok_words(t):
    return len(vp.spoken(t)) if re.search(r"\d", t) else 1


def tok_syl(t):
    return max(1, sum(vp.syllables(x) for x in vp.spoken(t)))


def q(x, nd=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def is_end(w):
    return bool(re.search(r"[.?!][\"'”’)]*$", w)) and not ABBR.match(w.strip("\"'“”()"))


def keyword_kind(word, first_in_sentence, emph):
    bare = re.sub(r"^[^\w$£€]+|[^\w%]+$", "", word)
    if re.search(r"\d", bare) or bare.lower() in NUMW:
        return "number"
    if bare.lower() in TURNW:
        return "turn"
    core = re.sub(r"'s$", "", bare)
    if not first_in_sentence and core[:1].isupper() and core not in NONAME and not (len(core) <= 3 and core.isupper()):
        return "name"
    if emph and bare.lower() not in ("one",):
        return "name"
    return None



def script_text_for(d, stem):
    """the script the listener was supposed to hear: paragraphs.txt, else the P??.receipt.json texts, else the assembled receipt's full text"""
    pt = os.path.join(d, "paragraphs.txt")
    if os.path.exists(pt):
        return open(pt).read()
    rs = sorted(glob.glob(os.path.join(d, "P??.receipt.json")))
    if rs:
        try:
            return "\n\n".join((json.load(open(r)).get("text") or "") for r in rs)
        except Exception:
            pass
    rp = os.path.join(d, stem + ".receipt.json")
    if os.path.exists(rp):
        try:
            return json.load(open(rp)).get("full_script_text") or ""
        except Exception:
            pass
    return ""


def split_sentences(text):
    out = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = " ".join(para.split())
        if not para:
            continue
        parts = re.split(r"(?<=[.?!])[\"'\u201d\u2019)]*\s+(?=[\"'\u201c(]?[A-Z0-9$\u00a3])", para)
        out += [x for x in parts if x.strip()]
    return out


def assign_sentences(words, sent_texts):
    """sentence index per token from the SCRIPT's sentence boundaries (the take's own ASR punctuation is not reliable: Mistral writes
    'right, they sound right.' where the script has two sentences). Monotonic difflib alignment of the spoken-word expansions."""
    sw, sid = [], []
    for si, t in enumerate(sent_texts):
        for tok in t.split():
            for x in vp._expanded([tok]):
                sw.append(x); sid.append(si)
    tw, tk = [], []
    for k, w in enumerate(words):
        for x in vp._expanded([w["word"]]):
            tw.append(x); tk.append(k)
    sm = difflib.SequenceMatcher(None, tw, sw, autojunk=False)
    got = {}
    for a_, b_, size in sm.get_matching_blocks():
        for i in range(size):
            k = tk[a_ + i]
            got.setdefault(k, sid[b_ + i])
    out, last = [], 0
    for k in range(len(words)):
        last = max(last, got.get(k, last))
        out.append(last)
    return out

# ------------------------------------------------------------------------------------------------ the signal layer
def extract(d, stem, use_cache=True):
    wav = os.path.join(d, stem + ".wav")
    words = [w for w in json.load(open(os.path.join(d, stem + ".words.json")))["words"] if not re.fullmatch(r"[—–-]+", w["word"])]
    if not words:
        raise SystemExit("no words in %s" % stem)
    a, sr = read_wav(wav)
    db, _ = vp.frame_db(a, sr, HOP)
    sha = sha_file(wav)
    wh = cached("whisper-" + sha, lambda: run_whisper(wav), use_cache)
    pr = cached("praat-" + sha, lambda: run_praat(wav), use_cache)
    f0 = nan_arr(pr["f0_st"])
    idb = nan_arr(pr["int_db"])
    f0_t0, i_t0 = pr["t0"], pr["i_t0"]
    dur = len(a) / sr

    inside = np.zeros(len(db), bool)
    for w in words:
        inside[int(w["start"] / HOP): min(len(db), int(np.ceil(w["end"] / HOP)))] = True
    level = float(np.percentile(db[inside], 90))      # speech p90 (dB mean-square), same convention as voice_pace
    floor = float(np.percentile(db, 3))

    # ---- whisper <-> token alignment (word probability, words not heard, extra words heard)
    tw, tix = [], []
    for k, w in enumerate(words):
        for x in vp._expanded([w["word"]]):
            tw.append(x); tix.append(k)
    ww, wix = [], []
    for j, x in enumerate(wh):
        for y in vp._expanded([x["w"]]):
            ww.append(y); wix.append(j)
    sm = difflib.SequenceMatcher(None, tw, ww, autojunk=False)
    tok_prob = [None] * len(words)
    tok_heard = [False] * len(words)
    tok_heard_as = {}
    matched_t = set()
    extra = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                k = tix[i]
                matched_t.add(i)
                p = wh[wix[j]]["p"]
                tok_prob[k] = p if tok_prob[k] is None else min(tok_prob[k], p)
        elif tag == "replace":
            ks = sorted({tix[i] for i in range(i1, i2)})
            hs = " ".join(wh[wix[j]]["w"] for j in sorted(set(wix[j] for j in range(j1, j2))))
            for k in ks:
                tok_heard_as[k] = hs
            if (i2 - i1) != (j2 - j1) or True:
                pass
        elif tag == "insert":
            for j in sorted(set(wix[j] for j in range(j1, j2))):
                extra.append({"w": wh[j]["w"], "t": wh[j]["s"], "p": wh[j]["p"], "j": j})
    cnt_t = {}
    for i, k in enumerate(tix):
        cnt_t.setdefault(k, [0, 0])
        cnt_t[k][0] += 1
        cnt_t[k][1] += (i in matched_t)
    for k, (n, m) in cnt_t.items():
        tok_heard[k] = (m == n)
    # an inserted whisper word that repeats a neighbouring whisper word = a stutter / restart heard
    for e in extra:
        j = e["j"]
        nb = [vp._norm(wh[x]["w"]) for x in range(max(0, j - 2), min(len(wh), j + 3)) if x != j]
        e["repeat"] = vp._norm(e["w"]) in nb and len(vp._norm(e["w"])) > 0
        if e["repeat"]:     # whisper sometimes splits ONE word into two timestamps ('every' 'Every' with no silence between): not a stutter
            prev_e = wh[j - 1]["e"] if j > 0 else wh[j]["s"]
            lo_, hi_ = int(prev_e / HOP), int(wh[j]["s"] / HOP) + 1
            e["gap_s"] = round(max(0.0, wh[j]["s"] - prev_e), 3)
            seg_ = db[lo_:hi_]
            e["silence_between"] = bool(len(seg_) and float(seg_.min()) < level - vp.PAUSE_DB and e["gap_s"] >= 0.06)
            e["repeat"] = e["silence_between"]

    # ---- boundaries: real silence between consecutive tokens
    n = len(words)
    thr = level - vp.PAUSE_DB

    def pause_at(e_prev, s_next):
        lo, hi = max(0, int((e_prev - 0.15) / HOP)), min(len(db), int((s_next + 0.15) / HOP))
        rr = vp.silent_runs(db, HOP, lo, hi, thr, 0.04)
        return max([b - a_ for a_, b in rr], default=0.0)

    bnd = []
    for k in range(n - 1):
        g = words[k + 1]["start"] - words[k]["end"]
        bnd.append({"gap": g, "sil": pause_at(words[k]["end"], words[k + 1]["start"]) if g > -0.05 else 0.0})

    # ---- sentences
    sents, cur = [], []
    st = split_sentences(script_text_for(d, stem))
    seg_from = "asr-punctuation"
    if st:
        sid = assign_sentences(words, st)
        seg_from = "script"
        for k in range(n):
            if cur and sid[k] != sid[cur[-1]]:
                sents.append(cur); cur = []
            cur.append(k)
        sents.append(cur)
    else:
        for k, w in enumerate(words):
            cur.append(k)
            if is_end(w["word"]) or k == n - 1:
                sents.append(cur); cur = []
    S = []
    all_pk_e, all_pk_f = [], []
    for si, ks in enumerate(sents):
        t0, t1 = words[ks[0]]["start"], words[ks[-1]]["end"]
        span = max(t1 - t0, 0.2)
        nw = sum(tok_words(words[k]["word"]) for k in ks)
        syl = sum(tok_syl(words[k]["word"]) for k in ks)
        inner = [bnd[k]["sil"] for k in ks[:-1] if bnd[k]["sil"] >= vp.PAUSE_MIN_S]
        pts = sum(inner)
        sd = {"i": si, "para": words[ks[0]].get("paragraph"), "t0": q(t0, 2), "t1": q(t1, 2), "k0": ks[0], "k1": ks[-1],
              "text": " ".join(words[k]["word"] for k in ks), "words": nw, "syl": syl,
              "wpm": q(nw / span * 60, 1), "artic_wpm": q(nw / max(span - pts, 0.4) * 60, 1), "inner_pauses": len(inner), "inner_pause_s": q(pts)}
        # pitch of the sentence
        v = sl(f0, f0_t0, t0, t1)
        v = v[np.isfinite(v)]
        if len(v) >= 12:
            sd.update(f0_med_st=q(np.median(v)), f0_range_st=q(np.percentile(v, 90) - np.percentile(v, 10)), f0_sd_st=q(np.std(v)),
                      voiced_frac=q(len(v) / max(1, (t1 - t0) / HOP), 2))
            tail = sl(f0, f0_t0, t1 - 0.35, t1 + 0.05)
            tail = tail[np.isfinite(tail)]
            head = sl(f0, f0_t0, t0, t0 + 0.30)
            head = head[np.isfinite(head)]
            if len(tail) >= 5:
                sd["term_delta_st"] = q(np.median(tail) - np.median(v))
            if len(head) >= 5:
                sd["onset_delta_st"] = q(np.median(head) - np.median(v))
        # word-level stress
        wl = []
        for k in ks:
            w = words[k]
            ee = sl(idb, i_t0, w["start"] - 0.02, w["end"] + 0.02)
            ee = ee[np.isfinite(ee)]
            ff = sl(f0, f0_t0, w["start"], w["end"])
            ff = ff[np.isfinite(ff)]
            wl.append({"k": k, "e": float(ee.max()) if len(ee) else None, "f": float(ff.max()) if len(ff) else None,
                       "dsyl": (w["end"] - w["start"]) / tok_syl(w["word"])})
        kws = []
        for pos, (k, m_) in enumerate(zip(ks, wl)):
            kind = keyword_kind(words[k]["word"], pos == 0, words[k].get("emphasis"))
            if not kind or m_["e"] is None:
                continue
            oth = [x for x in wl if x["k"] != k]
            e_oth = [x["e"] for x in oth if x["e"] is not None]
            f_oth = [x["f"] for x in oth if x["f"] is not None]
            d_oth = [x["dsyl"] for x in oth]
            if len(e_oth) < 3:
                continue
            e_d = m_["e"] - float(np.median(e_oth))
            f_d = (m_["f"] - float(np.median(f_oth))) if (m_["f"] is not None and len(f_oth) >= 3) else None
            d_r = m_["dsyl"] / max(float(np.median(d_oth)), 1e-3)
            landed = bool(e_d >= TH["kw_stress_e_db"] or (f_d is not None and f_d >= TH["kw_stress_f0_st"]) or d_r >= TH["kw_stress_dur"])
            kws.append({"word": re.sub(r"^[^\w$£€]+|[^\w%]+$", "", words[k]["word"]), "kind": kind, "e_db": q(e_d, 1), "f0_st": q(f_d, 1), "dur_ratio": q(d_r),
                        "landed": landed})
        sd["keywords"] = kws
        # asr
        probs = [(k, tok_prob[k]) for k in ks if tok_prob[k] is not None]
        sd["asr_min_prob"] = q(min([p for _, p in probs], default=None), 2)
        sd["low_prob_words"] = [words[k]["word"] for k, p in probs if p < TH["low_prob"]]
        sd["unheard_words"] = [{"word": words[k]["word"], "heard_as": tok_heard_as.get(k)} for k in ks if not tok_heard[k]]
        sd["extra_heard"] = [e for e in extra if t0 - 0.1 <= e["t"] <= t1 + 0.1]
        S.append(sd)
    # pauses around boundaries (energy-measured) and sawtooth
    for si, sd in enumerate(S):
        sd["pause_after_ms"] = q(1000 * bnd[sd["k1"]]["sil"], 0) if si < len(S) - 1 else None
        sd["pause_before_ms"] = q(1000 * bnd[sd["k0"] - 1]["sil"], 0) if si > 0 else None
        sd["tok_gap_after_ms"] = q(1000 * bnd[sd["k1"]]["gap"], 0) if si < len(S) - 1 else None

    # ---- land-and-restart: stray voiced runs inside mid-sentence gaps (voice_pace.gap_profile) and repeated words whisper heard
    strays = []
    for sd in S:
        for k in range(sd["k0"], sd["k1"]):
            if bnd[k]["gap"] >= 0.20:
                gp = vp.gap_profile(a, sr, db, HOP, words[k], words[k + 1], level)
                if gp["kind"] == "stray":
                    strays.append({"sent": sd["i"], "after": words[k]["word"], "before": words[k + 1]["word"], "gap_s": gp["gap_s"],
                                   "run_len": max(r["len"] for r in gp["runs"] if r["cls"] == "stray")})
    # sawtooth: boundaries where the voice falls >= 2.5 st at the end and starts the next sentence >= 2.5 st above it
    saw = 0
    for si in range(len(S) - 1):
        a_, b_ = S[si], S[si + 1]
        if a_.get("term_delta_st") is not None and b_.get("onset_delta_st") is not None and a_["term_delta_st"] <= -2.5 and b_["onset_delta_st"] >= 2.5:
            saw += 1

    # ---- breaths and clicks in the gaps
    breaths, clicks = 0, 0
    loud_breaths = 0
    for k in range(n - 1):
        g0, g1 = words[k]["end"], words[k + 1]["start"]
        if g1 - g0 < 0.14:
            continue
        lo, hi = int((g0 + 0.04) / HOP), int((g1 - 0.04) / HOP)
        if hi - lo < 8:
            continue
        seg = db[lo:hi]
        mid = (seg > floor + 9) & (seg < level - 16)
        # longest run of mid-level frames
        best, run = 0, 0
        for m_ in mid:
            run = run + 1 if m_ else 0
            best = max(best, run)
        if best * HOP >= 0.12:
            x = a[int((g0 + 0.04) * sr): int((g1 - 0.04) * sr)]
            if vp._voicing(x, sr) < 0.45:
                breaths += 1
                if float(seg.max()) > level - 24:
                    loud_breaths += 1
        # click: a 2 ms burst far above the gap's own steady level, strong in absolute terms
        x = a[int((g0 + 0.03) * sr): int((g1 - 0.03) * sr)]
        w2 = max(8, int(0.002 * sr))
        if len(x) > 6 * w2:
            nfr = (len(x) - 1) // w2
            fr = np.abs(np.diff(x[: nfr * w2 + 1]))[: nfr * w2].reshape(nfr, w2).max(axis=1)
            if fr.max() > 12 * (np.median(fr) + 1e-6) and fr.max() > 10 ** ((level - 22) / 20):
                clicks += 1

    # ---- hook, body, CTA
    t_first = words[0]["start"]
    hook_ks = [k for k, w in enumerate(words) if w["start"] < t_first + 3.0]
    hk_end = words[hook_ks[-1]]["end"]
    hook_words = sum(tok_words(words[k]["word"]) for k in hook_ks)
    last_para = words[-1].get("paragraph")
    cta_ks = [k for k, w in enumerate(words) if w.get("paragraph") == last_para]
    if len(cta_ks) > 40 or not cta_ks:      # not a stand-alone CTA paragraph: take the last sentence
        cta_ks = sents[-1]
    cta_t0, cta_t1 = words[cta_ks[0]]["start"], words[cta_ks[-1]]["end"]
    body_ks = [k for k in range(n) if k not in set(hook_ks) and k not in set(cta_ks)]

    def spk_db(t0, t1):
        x = sl(idb, i_t0, t0, t1)
        x = x[np.isfinite(x)]
        return x[x > np.percentile(x, 90) - 24] if len(x) else x

    def f0range(t0, t1):
        v = sl(f0, f0_t0, t0, t1)
        v = v[np.isfinite(v)]
        return (float(np.percentile(v, 90) - np.percentile(v, 10)), float(np.median(v))) if len(v) >= 12 else (None, None)

    def span_wpm(ks_):
        if not ks_:
            return None
        t0_, t1_ = words[ks_[0]]["start"], words[ks_[-1]]["end"]
        return sum(tok_words(words[k]["word"]) for k in ks_) / max(t1_ - t0_, 0.3) * 60

    body_t0 = words[body_ks[0]]["start"] if body_ks else t_first
    body_t1 = words[body_ks[-1]]["end"] if body_ks else words[-1]["end"]
    body_wpm_sent = [s["wpm"] for s in S if s["k0"] in set(body_ks) and s["words"] >= 5]
    hook = {"t_first_word": q(t_first, 2), "wpm": q(hook_words / max(hk_end - t_first, 0.5) * 60, 1), "window_s": q(hk_end - t_first, 2)}
    hdb, bdb = spk_db(t_first, hk_end), spk_db(body_t0, body_t1)
    hook["energy_db"] = q(float(np.mean(hdb)), 1) if len(hdb) else None
    hook["body_energy_db"] = q(float(np.mean(bdb)), 1) if len(bdb) else None
    hook["f0_range_st"], hook["f0_med_st"] = [q(x) for x in f0range(t_first, hk_end)]
    br, bm = f0range(body_t0, body_t1)
    hook["body_f0_range_st"], hook["body_f0_med_st"] = q(br), q(bm)
    hook["body_wpm"] = q(float(np.median(body_wpm_sent)), 1) if body_wpm_sent else None
    hook["first_sentence_wpm"] = S[0]["wpm"]
    cta = {"text": " ".join(words[k]["word"] for k in cta_ks), "words": sum(tok_words(words[k]["word"]) for k in cta_ks),
           "wpm": q(span_wpm(cta_ks), 1), "paragraph": last_para,
           "pause_before_ms": q(1000 * bnd[cta_ks[0] - 1]["sil"], 0) if cta_ks[0] > 0 else None}
    cdb = spk_db(cta_t0, cta_t1)
    lw0, lw1 = words[cta_ks[-1]]["start"], words[cta_ks[-1]]["end"]
    ldb = spk_db(lw0, lw1)
    cta["energy_db"] = q(float(np.mean(cdb)), 1) if len(cdb) else None
    cta["last_word_db_minus_cta"] = q(float(np.mean(ldb)) - float(np.mean(cdb)), 1) if len(ldb) and len(cdb) else None
    v = sl(f0, f0_t0, cta_t0, cta_t1 + 0.1)
    vv = v[np.isfinite(v)]
    if len(vv) >= 20:
        idx = np.where(np.isfinite(v))[0]
        last = idx[-1]
        seg = np.arange(max(0, last - 20), last + 1)
        sv = v[seg]; m_ = np.isfinite(sv)
        cta["last_minus_med_st"] = q(v[last] - np.median(vv))
        cta["final_slope_st_s"] = q(np.polyfit(seg[m_] * HOP, sv[m_], 1)[0], 1) if m_.sum() >= 6 else None
        cta["f0_range_st"] = q(np.percentile(vv, 90) - np.percentile(vv, 10))
    cta["body_wpm"] = hook["body_wpm"]

    # ---- take summary
    nwords = sum(s["words"] for s in S)
    all_w = [s for s in S]
    bodyS = [s for s in S if s["k0"] in set(body_ks)] or S
    frag = [s["words"] <= TH["frag_words"] for s in S]
    runmax, run = 0, 0
    for fgm in frag:
        run = run + 1 if fgm else 0
        runmax = max(runmax, run)
    paras = {}
    for s in S:
        paras.setdefault(s["para"], []).append(s)
    pdens = {p: q(100 * len(ss) / max(1, sum(x["words"] for x in ss)), 1) for p, ss in paras.items()}
    kw_all = [k for s in S for k in s["keywords"]]
    pa = [s["pause_after_ms"] for s in S if s["pause_after_ms"] is not None]
    spans = words[-1]["end"] - t_first
    sp_dur = sum(words[k]["end"] - words[k]["start"] for k in range(n))
    take = {
        "duration_s": q(dur, 1), "words": nwords, "sentences": len(S),
        "span_wpm": q(nwords / max(spans, 1) * 60, 1),
        "sent_per_100w": q(100 * len(S) / max(nwords, 1), 1),
        "sent_per_100w_body": q(100 * len(bodyS) / max(sum(s["words"] for s in bodyS), 1), 1),
        "para_density": pdens, "fragment_run_max": runmax, "fragments": int(sum(frag)),
        "wpm_sent_sd": q(float(np.std([s["wpm"] for s in S if s["words"] >= 5])), 1) if len([s for s in S if s["words"] >= 5]) > 2 else None,
        "pause_end_mean_ms": q(float(np.mean(pa)), 0) if pa else None, "pause_end_min_ms": q(float(min(pa)), 0) if pa else None,
        "pause_end_under_150": int(sum(1 for x in pa if x < TH["pause_end_min_ms"])),
        "f0_range_st_med": q(float(np.median([s["f0_range_st"] for s in S if s.get("f0_range_st") is not None]))) if any(s.get("f0_range_st") is not None for s in S) else None,
        "f0_sd_st_med": q(float(np.median([s["f0_sd_st"] for s in S if s.get("f0_sd_st") is not None]))) if any(s.get("f0_sd_st") is not None for s in S) else None,
        "keywords": len(kw_all), "keywords_landed": int(sum(1 for k in kw_all if k["landed"])),
        "keyword_landing": q(sum(1 for k in kw_all if k["landed"]) / len(kw_all), 2) if kw_all else None,
        "terminal_fall_rate_per_10s": q(10 * sum(1 for s in S if (s.get("term_delta_st") or 0) <= -2.5) / max(spans, 1), 2),
        "sawtooth_boundaries": saw, "sawtooth_rate_per_10s": q(10 * saw / max(spans, 1), 2),
        "strays": strays, "repeat_words_heard": [e for e in extra if e["repeat"]],
        "extra_heard_n": len(extra), "unheard_n": int(sum(len(s["unheard_words"]) for s in S)), "low_prob_n": int(sum(len(s["low_prob_words"]) for s in S)),
        "unheard_per_100w": q(100 * sum(len(s["unheard_words"]) for s in S) / max(nwords, 1), 1),
        "breaths": breaths, "loud_breaths": loud_breaths, "clicks": clicks,
        "breaths_per_min": q(breaths / max(spans, 1) * 60, 1), "clicks_per_min": q(clicks / max(spans, 1) * 60, 1),
        "speech_frac": q(sp_dur / max(spans, 1), 3),
    }
    return {"id": stem, "dir": d, "wav_sha": sha, "take": take, "hook": hook, "cta": cta, "sentences": S, "segmented_by": seg_from}


# ------------------------------------------------------------------------------------------------ signal flags and verdict
# Calibrated 19 Oct 2026 against 18 approved takes (3 pilots + 15 best approved shorts) and 66 pre-revoice snapshots (delivery_calibration/).
# Honest finding: take-level aggregates (density, pauses, pitch range, key-word landing, hook ratios, mumble counts, pace spread) do NOT separate
# the snapshots from the approved takes (AUC 0.37-0.62), because every take went through the same TTS and the same 165-172 wpm assembly band.
# What did separate: the CTA's terminal lift (S02, the one Sam-confirmed bad take, +7.0 st; approved max +2.2), a CTA that does not fade
# (last word >= 0.5 dB above the CTA's own level: 0 of 18 approved, 5 of 8 takes re-voiced "pre-cta"), and a sentence at >= 7.9 syllables/s of
# speech (approved max 7.69; the voice_pace rush signature). Those are the scored rules. Everything else is reported as info and given to the
# judges as facts, never scored. The restart detectors (stray voiced run, repeated word) never fired on any of the 84 real takes; they are
# proven by the mutation test in delivery_calibration/mutation_test.py.
RULE_W = {"restart_stray": 100, "restart_repeat": 100, "cta_lift": 2, "cta_nofade": 2, "fast_syl": 2, "density_beat": 1, "mumble_sentence": 1}


def signal_flags(F, cal=None):
    """flags = [{"id","sev","w","sent","msg"}]; sev fail|warn|info. info = reported, not scored."""
    th = dict(TH)
    th.update((cal or {}).get("TH", {}))
    T, H, C, S = F["take"], F["hook"], F["cta"], F["sentences"]
    fl = []

    def add(i, sev, sent, msg):
        fl.append({"id": i, "sev": sev, "w": RULE_W.get(i, 0) if sev != "info" else 0, "sent": sent, "msg": msg})

    for s in T["strays"]:
        add("restart_stray", "fail", s["sent"], "voiced run of %.2fs inside the %.2fs gap between '%s' and '%s' with no word timed: a false start or stutter" %
            (s["run_len"], s["gap_s"], s["after"], s["before"]))
    for e in T["repeat_words_heard"]:
        add("restart_repeat", "fail", None, "whisper heard '%s' repeated near %.1fs with silence between the two and no repeat in the script (stutter / restart)" % (e["w"], e["t"]))
    # scored: the CTA and a sentence at rush speed
    if C.get("last_minus_med_st") is not None and C["last_minus_med_st"] >= th["cta_lift_st"]:
        add("cta_lift", "warn", T["sentences"] - 1, "CTA ends %.1f st above its own median pitch (slope %s st/s): a lifted last word, the S02 signature (approved takes top out at +2.2)" %
            (C["last_minus_med_st"], C.get("final_slope_st_s")))
    if C.get("last_word_db_minus_cta") is not None and C["last_word_db_minus_cta"] >= th["cta_nofade_db"]:
        add("cta_nofade", "warn", T["sentences"] - 1, "CTA's last word is %.1f dB ABOVE the CTA's level (approved takes land 2-16 dB below): the close is pushed, not landed" % C["last_word_db_minus_cta"])
    for s in S:
        if s["words"] >= 6:
            sr_ = s["syl"] / max(1e-3, (s["t1"] - s["t0"]) - (s.get("inner_pause_s") or 0))
            if sr_ >= th["fast_syl_rate"]:
                add("fast_syl", "warn", s["i"], "sentence %d at %.1f syllables/s of speech (approved takes top out at 7.7): rushed" % (s["i"], sr_))
    hi_beats = {p: d for p, d in T["para_density"].items() if d is not None and d > th["dens_beat_max"]}
    if hi_beats:
        add("density_beat", "warn", None, "paragraph(s) %s above %g sentences/100 words (fragment runs make the voice land and restart)" % (", ".join("%s=%s" % x for x in hi_beats.items()), th["dens_beat_max"]))
    for s in S:
        if len(s["low_prob_words"]) >= 3:
            add("mumble_sentence", "warn", s["i"], "sentence %d has low-confidence words: %s" % (s["i"], ", ".join(s["low_prob_words"][:5])))
    # info only (not scored): given to the judges as facts
    if T["fragment_run_max"] > th["frag_run_max"]:
        add("fragment_run", "info", None, "%d fragments in a row" % T["fragment_run_max"])
    if T["keywords"] >= 3 and T["keyword_landing"] is not None and T["keyword_landing"] < 0.55:
        flat = [(s["i"], k["word"]) for s in S for k in s["keywords"] if not k["landed"]]
        add("keywords_flat", "info", None, "%d of %d key words carry no extra energy, pitch or length: %s" % (T["keywords"] - T["keywords_landed"], T["keywords"], ", ".join("s%d '%s'" % x for x in flat[:6])))
    if H["wpm"] and H.get("body_wpm") and H["wpm"] < H["body_wpm"] * 0.7:
        add("hook_slow", "info", 0, "hook %.0f wpm against a body of %.0f" % (H["wpm"], H["body_wpm"]))
    if T["unheard_per_100w"] is not None and T["unheard_per_100w"] >= 9:
        add("mumble", "info", None, "%d words the second transcriber did not hear (%.1f per 100)" % (T["unheard_n"], T["unheard_per_100w"]))
    return fl


def signal_verdict(fl, cal=None):
    th = (cal or {}).get("verdict", {"improve_at": 2, "fail_at": 100})
    score = sum(f.get("w", 0) for f in fl)
    if any(f["sev"] == "fail" for f in fl) or score >= th["fail_at"]:
        v = "FAIL"
    elif score >= th["improve_at"]:
        v = "IMPROVE"
    else:
        v = "PASS"
    return v, score


def load_cal():
    p = os.path.join(CALIB, "thresholds.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ------------------------------------------------------------------------------------------------ judgement prompts
RUBRIC = """You judge how well a short spoken story is DELIVERED by ear, for a 55-65 second business video narrated by one voice (a cloned UK male voice).
A listener hears it once, at speed, on a phone, with no way to re-read. Score what the ear gets, not what the page says.
Judge only these five things:
 1. Is the story clear by ear alone (can a listener follow who did what, what it cost, and the point, without seeing the text)?
 2. Which sentence would a listener lose or mishear (too dense, a number or name that blurs, a clause that runs on, a fragment run)?
 3. Where should emphasis or a pause go (the number, the name, the turn "but/except/only", the landing before the point)?
 4. Is the hook (the first sentence, first ~3 seconds) delivered with urgency?
 5. Three concrete, re-voiceable fixes. A fix is either a TEXT rewrite for sentence density (join fragments with commas and dashes, keep it spoken, 8-16
    sentences per 100 words, never more than two fragments in a row) or a PAUSE/EMPHASIS mark on a specific word. Each fix must name its sentence number.
Verdict: PASS = a listener would not lose anything and nothing needs re-voicing. IMPROVE = clear overall but at least one named sentence/beat should be
re-voiced or rewritten for a concrete reason a listener would hear. FAIL = a listener would lose the thread, a number/name is garbled, or the voice
restarts/stumbles/rushes through something that matters.
Calibration: the approved reference takes (the house standard, signed off by the owner) PASS. This voice is a clone that by design says each sentence quickly
and then leaves roughly 0.4-0.7 s of silence, so sentence-level speeds of 150-300 wpm and short sentences are NORMAL in the references; judge a sentence only
if it is far outside that style or the TEXT itself is dense, long or full of numbers. Default to PASS and leave a sentence out unless you can say what a
listener would mishear. Even on a PASS, still give three optional polish fixes (they are advice, not a reason to downgrade).
Return ONLY one JSON object, no markdown, no commentary:
{"verdict":"PASS|IMPROVE|FAIL",
 "clear_by_ear":{"score":1-5,"why":"<=25 words"},
 "listener_loses":[{"sentence":<int>,"why":"<=20 words"}],
 "emphasis_pause":[{"sentence":<int>,"word":"<word>","action":"stress|pause_before|pause_after|slow_down","why":"<=15 words"}],
 "hook":{"urgent":true|false,"why":"<=20 words"},
 "fixes":[{"sentence":<int>,"type":"rewrite|pause|emphasis","before":"<current text or word>","after":"<new text or mark>","why":"<=20 words"},{...},{...}]}
Exactly three fixes. Sentence numbers are the [n] markers below."""

REFERENCE = ("Reference for the numbers (approved pilot takes): speaking pace 156-173 wpm; body 8-16 sentences per 100 words; silence after a sentence "
             "usually 250-600 ms; a key word usually carries +1.5 dB or +2 st or +30% length over its neighbours.")


def numbered_script(F):
    return "\n".join("[%d] %s" % (s["i"], s["text"]) for s in F["sentences"])


def sentence_table(F):
    L = []
    for s in F["sentences"]:
        kw = ",".join("%s%s" % (k["word"], "" if k["landed"] else "(flat)") for k in s["keywords"]) or "-"
        L.append("[%d] %s | %swpm | after-pause %sms | pitch-range %sst | end-fall %sst | keywords %s | heard-badly %s" % (
            s["i"], s["text"], int(s["wpm"] or 0), int(s["pause_after_ms"]) if s["pause_after_ms"] is not None else "-", s.get("f0_range_st", "-"),
            s.get("term_delta_st", "-"), kw, ",".join(u["word"] + ("->" + u["heard_as"] if u.get("heard_as") else "") for u in s["unheard_words"]) or "-"))
    return "\n".join(L)


def text_prompt(F, flags):
    T, H, C = F["take"], F["hook"], F["cta"]
    summ = ("Take: %s words, %s sentences, %.0f wpm overall, %s sentences per 100 words (body %s), hook %s wpm against a body of %s wpm, "
            "hook energy %s dB against %s, CTA %s wpm, end-lift %s st. Key words landed: %s of %s." %
            (T["words"], T["sentences"], T["span_wpm"], T["sent_per_100w"], T["sent_per_100w_body"], H["wpm"], H["body_wpm"], H["energy_db"], H["body_energy_db"],
             C["wpm"], C.get("last_minus_med_st"), T["keywords_landed"], T["keywords"]))
    fl = "\n".join("- " + f["msg"] for f in flags) or "- none"
    return "%s\n\n%s\n\nYou are NOT given the audio. You are given the transcript and measurements taken from the audio.\n\nMEASUREMENTS\n%s\n\nSIGNAL FLAGS\n%s\n\nSENTENCES (text | measurements)\n%s\n" % (
        RUBRIC, REFERENCE, summ, fl, sentence_table(F))


def audio_prompt(F, label=""):
    return ("%s\n\nYou are given the AUDIO of the take. Listen to it. Judge from what you HEAR: pace, where the voice stresses, where it pauses, whether "
            "sentences are lost, whether it restarts or stumbles, how the first sentence and the closing 'follow' line land. The script, with sentence "
            "numbers, is below so you can name sentences; the audio is the evidence.\n\nSCRIPT\n%s\n" % (RUBRIC, numbered_script(F)))


def parse_json_obj(txt):
    if not txt:
        return None
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            return json.loads(s2)
        except Exception:
            return None


def valid_judgement(j):
    return isinstance(j, dict) and j.get("verdict") in ("PASS", "IMPROVE", "FAIL") and isinstance(j.get("fixes"), list)


def fleet_run(prompt, timeout=240):
    r = subprocess.run([FLEET, "run", "reader", prompt], capture_output=True, text=True, timeout=timeout)
    return r.stdout, r.stderr[-300:], r.returncode


def judge_text(F, flags, runs=2):
    prompt = text_prompt(F, flags)
    outs = []
    for r in range(runs):
        j, err = None, ""
        for attempt in range(2):
            try:
                txt, err, rc = fleet_run(prompt)
            except Exception as e:
                txt, err = "", repr(e)[:200]
            j = parse_json_obj(txt)
            if valid_judgement(j):
                break
            j = None
        outs.append({"run": r + 1, "ok": j is not None, "judgement": j, "error": None if j else err or "unparseable"})
    return stability("text", outs)


def stability(layer, outs):
    good = [o["judgement"] for o in outs if o["ok"]]
    verdicts = [g["verdict"] for g in good]
    stable = len(good) == len(outs) and len(set(verdicts)) == 1 and len(good) >= 2
    sets = [set(x.get("sentence") for x in g.get("listener_loses", []) if isinstance(x, dict)) for g in good]
    jac = None
    if len(sets) >= 2 and (sets[0] | sets[1]):
        jac = round(len(sets[0] & sets[1]) / len(sets[0] | sets[1]), 2)
    return {"layer": layer, "runs": outs, "verdicts": verdicts, "stable": stable, "loses_jaccard": jac,
            "verdict": verdicts[0] if stable else None}


# ------------------------------------------------------------------------------------------------ audio listener (Mistral direct, Voxtral Small)
MISTRAL_MODEL = "voxtral-small-2507"
# Mistral list price (assumption, from their pricing page; the response carries tokens/seconds, not money): audio $0.004 per minute, text
# $0.10 per M input tokens and $0.30 per M output tokens; USD->EUR 0.92. Marked "est" everywhere it is logged.
M_AUDIO_USD_MIN, M_IN_USD_M, M_OUT_USD_M, USD_EUR = 0.004, 0.10, 0.30, 0.92
AUDIO_CALLS = os.path.join(CALIB, "audio_calls.jsonl")
AUDIO_DAY_CAP_EUR = float(os.environ.get("DELIVERY_AUDIO_DAY_CAP_EUR", "3.0"))    # a guard against a runaway batch, not a budget


def mistral_key():
    for line in open(os.path.join(HOME, "alfred", ".env"), errors="ignore").read().splitlines():      # same read as gen_paragraphs.py env_key()
        if line.startswith("MISTRAL_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def spent_today_eur():
    tot, today = 0.0, time.strftime("%Y-%m-%d")
    if os.path.exists(AUDIO_CALLS):
        for ln in open(AUDIO_CALLS):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("ts", "").startswith(today):
                tot += float(r.get("cost_eur_est") or 0)
    return tot


def log_call(rec):
    os.makedirs(CALIB, exist_ok=True)
    with open(AUDIO_CALLS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def mistral_audio_call(key, prompt, b64, tag, retries=4):
    body = {"model": MISTRAL_MODEL, "temperature": 0.2, "max_tokens": 1500,
            "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": b64}, {"type": "text", "text": prompt}]}]}
    t = time.time()
    r = None
    for attempt in range(retries):
        req = urllib.request.Request("https://api.mistral.ai/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
            break
        except urllib.error.HTTPError as e:
            msg = e.read()[:200].decode("utf8", "replace")
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(8 * (attempt + 1)); continue
            return None, {"error": "HTTP %d %s" % (e.code, msg)}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5); continue
            return None, {"error": repr(e)[:200]}
    u = r.get("usage") or {}
    secs = float(u.get("prompt_audio_seconds") or 0)
    usd = secs / 60 * M_AUDIO_USD_MIN + (u.get("prompt_tokens") or 0) / 1e6 * M_IN_USD_M + (u.get("completion_tokens") or 0) / 1e6 * M_OUT_USD_M
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "route": "mistral", "model": MISTRAL_MODEL, "audio_seconds": secs,
           "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"), "total_tokens": u.get("total_tokens"),
           "audio_tokens": (u.get("prompt_tokens_details") or {}).get("audio_tokens"), "cost_eur_est": round(usd * USD_EUR, 6), "seconds": round(time.time() - t, 1)}
    log_call(rec)
    try:
        return r["choices"][0]["message"]["content"], rec
    except Exception:
        return None, {"error": "no content in response", **rec}


# ---- OpenRouter fallback (OFF unless DELIVERY_AUDIO_OR=1). The account needs >= $0.50 of credit for audio and the fleet's daily stop applies.
OR_MODEL = "mistralai/voxtral-small-24b-2507"
OR_DAILY_STOP = 1.50          # the fleet's stop (router.json openrouter_daily_usd capped by FLEET_OR_DAILY_MAX, default 1.5)
OR_MIN_BALANCE = 0.55         # the endpoint refuses audio below $0.50 of account credit


def or_key():
    k = os.environ.get("OPENROUTER_KEY")
    if k:
        return k
    for line in open(os.path.join(HOME, "alfred", ".env"), errors="ignore"):      # the only place real keys live (CRLF-safe), same as ~/.fleet/bin/fleet
        line = line.strip().lstrip("export ").strip()
        if line.startswith("OPENROUTER_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def or_get(path, key):
    req = urllib.request.Request("https://openrouter.ai/api/v1" + path, headers={"Authorization": "Bearer " + key})
    return json.load(urllib.request.urlopen(req, timeout=15)).get("data") or {}


def or_budget(key):
    """(ok, reason, info). Never prints the key."""
    try:
        auth = or_get("/auth/key", key)
        cred = or_get("/credits", key)
    except Exception as e:
        return False, "openrouter counters unreachable: %r" % (e,), {}
    daily = float(auth.get("usage_daily") or 0)
    bal = float(cred.get("total_credits") or 0) - float(cred.get("total_usage") or 0)
    info = {"usage_daily": round(daily, 4), "balance": round(bal, 4)}
    if daily >= OR_DAILY_STOP:
        return False, "fleet OpenRouter daily stop reached ($%.2f of $%.2f)" % (daily, OR_DAILY_STOP), info
    if bal < OR_MIN_BALANCE:
        return False, "OpenRouter balance $%.2f is under the $0.50 the audio endpoint needs" % bal, info
    return True, "ok", info


def to_b64_audio(path, t0=None, t1=None):
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if t0 is not None:
        cmd += ["-ss", "%.3f" % t0, "-to", "%.3f" % t1]
    # mp3 48 kbps mono 16 kHz: a 60 s wav (1.9 MB) gets HTTP 429 "rate_limited 1300" from Mistral (request size, measured 19 Oct: 30 s wav ok,
    # 40 s wav 429, 60 s mp3 ok), the same minute as 360 KB of mp3
    cmd += ["-i", path, "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "48k", "-f", "mp3", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    return base64.b64encode(r.stdout).decode()


def or_audio_call(key, prompt, b64, tag):
    body = {"model": OR_MODEL, "temperature": 0.2, "max_tokens": 1500, "usage": {"include": True},
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "input_audio", "input_audio": {"data": b64, "format": "mp3"}}]}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=180))
    except urllib.error.HTTPError as e:
        return None, {"error": "HTTP %d %s" % (e.code, e.read()[:200].decode("utf8", "replace"))}
    except Exception as e:
        return None, {"error": repr(e)[:200]}
    u = r.get("usage") or {}
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "route": "openrouter", "model": OR_MODEL, "cost_usd": u.get("cost"),
           "cost_eur_est": round(float(u.get("cost") or 0) * USD_EUR, 6), "prompt_tokens": u.get("prompt_tokens"),
           "completion_tokens": u.get("completion_tokens"), "seconds": round(time.time() - t, 1)}
    log_call(rec)
    try:
        return r["choices"][0]["message"]["content"], rec
    except Exception:
        return None, {"error": "no content in response", **rec}


def judge_audio(F, d, stem, runs=2):
    use_or = os.environ.get("DELIVERY_AUDIO_OR") == "1"
    key = or_key() if use_or else mistral_key()
    route = "openrouter" if use_or else "mistral"
    if not key:
        return {"layer": "audio", "route": route, "available": False, "reason": "no %s in ~/alfred/.env" % ("OPENROUTER_KEY" if use_or else "MISTRAL_KEY"),
                "stable": False, "verdict": None, "runs": []}
    if use_or:
        ok, why, info = or_budget(key)
        if not ok:
            return {"layer": "audio", "route": route, "available": False, "reason": why, "budget": info, "stable": False, "verdict": None, "runs": []}
    elif spent_today_eur() >= AUDIO_DAY_CAP_EUR:
        return {"layer": "audio", "route": route, "available": False, "reason": "audio guard: EUR %.2f spent today (cap %.2f, DELIVERY_AUDIO_DAY_CAP_EUR)" % (
            spent_today_eur(), AUDIO_DAY_CAP_EUR), "stable": False, "verdict": None, "runs": []}
    wav = os.path.join(d, stem + ".wav")
    dur = F["take"]["duration_s"]
    chunks = [(None, None)]
    if dur > 75:      # per paragraph, cut on the paragraph's token times
        paras = {}
        for s in F["sentences"]:
            paras.setdefault(s["para"], []).append(s)
        chunks = [(max(0.0, ss[0]["t0"] - 0.2), ss[-1]["t1"] + 0.3) for ss in paras.values()]
    prompt = audio_prompt(F)
    outs, cost = [], 0.0
    for r in range(runs):
        j, errs, calls = None, [], []
        for (c0, c1) in chunks:
            b64 = to_b64_audio(wav, c0, c1)
            txt, meta = (or_audio_call if use_or else mistral_audio_call)(key, prompt, b64, "%s run%d" % (stem, r + 1))
            calls.append(meta)
            cost += float(meta.get("cost_eur_est") or 0)
            jj = parse_json_obj(txt)
            if valid_judgement(jj):
                j = jj if j is None else (jj if rank(jj["verdict"]) > rank(j["verdict"]) else j)
            else:
                errs.append(meta.get("error") or "unparseable")
        outs.append({"run": r + 1, "ok": j is not None, "judgement": j, "error": None if j else "; ".join(errs)[:300], "calls": calls})
    res = stability("audio", outs)
    res.update(available=True, route=route, cost_eur_est=round(cost, 6), chunks=len(chunks))
    return res


def rank(v):
    return {"PASS": 0, "IMPROVE": 1, "FAIL": 2}.get(v, -1)


# ------------------------------------------------------------------------------------------------ combine
def combine(sig_v, text, audio):
    """audio (stable) is the primary listener: its verdict, raised (never lowered) by a signal FAIL (a restart/stutter is a measured fact).
    Else the stable text verdict, raised by the signal, never lowered. Else the signal alone."""
    src = "signal"
    v = sig_v
    if audio and audio.get("stable"):
        src, v = "audio", audio["verdict"]
        if sig_v == "FAIL" and rank(v) < 2:
            v, src = "FAIL", "audio+signal(fail)"
    elif text and text.get("stable"):
        src, v = "text", text["verdict"]
        if rank(sig_v) > rank(v):
            v, src = sig_v, "text+signal(raised)"
    return v, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--layers", default="signal,text,audio")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    layers = set(a.layers.split(","))
    d, stem = resolve(a.id)
    cal = load_cal()
    F = extract(d, stem, use_cache=not a.no_cache)
    flags = signal_flags(F, cal)
    sv, score = signal_verdict(flags, cal)
    res = {"id": stem, "dir": d, "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "signal": {"verdict": sv, "score": score, "flags": flags}, "features": F}
    text = audio = None
    if "text" in layers:
        text = judge_text(F, flags, a.runs)
        res["text"] = text
    if "audio" in layers:
        audio = judge_audio(F, d, stem, a.runs)
        res["audio"] = audio
    fv, src = combine(sv, text, audio)
    res["verdict"], res["verdict_from"] = fv, src
    fixes = []
    for L in (audio, text):
        if L and L.get("stable"):
            for o in L["runs"]:
                if o["ok"]:
                    fixes = o["judgement"].get("fixes", [])
                    break
        if fixes:
            break
    res["fixes"] = fixes
    out = a.out or os.path.join(d, stem + ".delivery.json")
    json.dump(res, open(out, "w"), indent=1, default=str)
    if a.json:
        print(json.dumps(res, default=str))
    else:
        print("%s DELIVERY-%s (from %s) | signal %s score %s | text %s | audio %s | flags: %s" % (
            stem, fv, src, sv, score, (text or {}).get("verdicts", "-") if text else "-", ((audio or {}).get("verdicts") if audio and audio.get("available") else
            ("unavailable: " + audio["reason"]) if audio else "-"), "; ".join(f["id"] for f in flags) or "none"))
    sys.exit(0 if fv == "PASS" else 1 if fv == "IMPROVE" else 2)


if __name__ == "__main__":
    main()
