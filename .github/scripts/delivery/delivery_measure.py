#!/usr/bin/env python3
"""delivery_measure.py: how a narrator DELIVERS a line (pace, pauses, emphasis, pitch, energy). Runs on Actions only.

  ref mode : delivery_measure.py ref IN.mp4 OUT.json --name r4 --pivot 20.38      (demucs vocals stem first, then Whisper words + pitch + energy)
  wav mode : delivery_measure.py wav IN.wav OUT.json --name P1.t3 [--words WORDS.json]  (a clean TTS take, no demucs)

Output JSON: words [{w,s,e,f0,f0max,f0min,db,dbpk,st}], sentences [...], pauses [...], sections {hook,story,turn,ad} and global speaker stats.
Units: wpm_span = words / (first word start .. last word end); wpm_net = words / (speaking time with gaps over 0.25 s removed).
Pitch in semitones relative to the file's own median f0 ("st"); energy as dB relative to the file's own median word level ("rel_db").
"""
import argparse, json, math, os, re, subprocess, sys
import numpy as np

NUM_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion trillion".split())


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("FAILED %s\n%s" % (" ".join(cmd[:5]), (r.stderr or r.stdout)[-800:]))
    return r.stdout


def to_wav(src, dst, sr=16000, mono=True):
    sh(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-vn", "-ac", "1" if mono else "2", "-ar", str(sr), dst])


def demucs_vocals(src, work):
    full = os.path.join(work, "full.wav")
    sh(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-vn", "-ac", "2", "-ar", "44100", full])
    sh([sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-d", "cpu", "-o", os.path.join(work, "dm"), full])
    v = os.path.join(work, "dm", "htdemucs", "full", "vocals.wav")
    out = os.path.join(work, "vocals16.wav")
    to_wav(v, out, 16000)
    return out


def whisper_words(path, model="small.en", prompt=None):
    from faster_whisper import WhisperModel
    m = WhisperModel(model, device="cpu", compute_type="int8")
    import soundfile as sf
    audio, _sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    segs, _ = m.transcribe(audio, word_timestamps=True, language="en", beam_size=5, initial_prompt=prompt, condition_on_previous_text=False)
    out = []
    for s in segs:
        for w in (s.words or []):
            out.append({"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)})
    return out


def pitch_track(path):
    import parselmouth
    snd = parselmouth.Sound(path)
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=70, pitch_ceiling=450, very_accurate=False, voicing_threshold=0.5, silence_threshold=0.05)
    f0 = p.selected_array["frequency"]
    t = p.xs()
    f0 = np.where(f0 > 0, f0, np.nan)
    return t, f0


def energy_track(x, sr, hop=0.01, win=0.03):
    n = int(win * sr); h = int(hop * sr)
    if len(x) < n:
        return np.array([0.0]), np.array([-120.0])
    k = (len(x) - n) // h + 1
    idx = np.arange(k)[:, None] * h + np.arange(n)[None, :]
    fr = x[idx]
    rms = np.sqrt((fr ** 2).mean(axis=1)) + 1e-9
    return (np.arange(k) * h + n / 2) / sr, 20 * np.log10(rms)


def is_num(w):
    t = re.sub(r"[^a-z0-9\-]", "", w.lower())
    return bool(re.search(r"\d", t)) or any(p in NUM_WORDS for p in t.split("-"))


def metrics(words, t_f, f0, t_e, db, speaker_f0, speaker_db):
    """add per-word pitch/energy"""
    for w in words:
        m = (t_f >= w["s"]) & (t_f <= w["e"])
        v = f0[m]; v = v[~np.isnan(v)]
        if len(v) >= 2:
            w["f0"] = round(float(np.median(v)), 1); w["f0max"] = round(float(np.percentile(v, 90)), 1); w["f0min"] = round(float(np.percentile(v, 10)), 1)
            w["st"] = round(12 * math.log2(w["f0"] / speaker_f0), 2); w["stmax"] = round(12 * math.log2(w["f0max"] / speaker_f0), 2)
        else:
            w["f0"] = w["f0max"] = w["f0min"] = w["st"] = w["stmax"] = None
        me = (t_e >= w["s"]) & (t_e <= w["e"])
        e = db[me]
        if len(e):
            w["db"] = round(float(np.percentile(e, 75)), 1); w["dbpk"] = round(float(e.max()), 1); w["rel_db"] = round(w["db"] - speaker_db, 1)
        else:
            w["db"] = w["dbpk"] = w["rel_db"] = None
    return words


def sentences_of(words):
    out, cur = [], []
    for w in words:
        cur.append(w)
        if re.search(r"[.?!]$", w["w"]) or re.search(r"\.\.\.$", w["w"]):
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


def span_stats(ws, speaker_f0=None):
    ws = [w for w in ws if w.get("s") is not None]
    if not ws:
        return None
    s, e = ws[0]["s"], ws[-1]["e"]
    gaps = [max(0.0, b["s"] - a["e"]) for a, b in zip(ws, ws[1:])]
    long = sum(g for g in gaps if g > 0.25)
    n = len(ws)
    sp = max(e - s, 1e-3); net = max(sp - long, 1e-3)
    st = [w["st"] for w in ws if w.get("st") is not None]
    stm = [w["stmax"] for w in ws if w.get("stmax") is not None]
    rel = [w["rel_db"] for w in ws if w.get("rel_db") is not None]
    pk = [w["dbpk"] for w in ws if w.get("dbpk") is not None]
    d = {"start": round(s, 2), "end": round(e, 2), "n_words": n, "dur": round(sp, 2), "wpm_span": round(n / sp * 60, 1), "wpm_net": round(n / net * 60, 1),
         "pause_total": round(sum(gaps), 2), "pause_max": round(max(gaps), 2) if gaps else 0.0, "n_pauses_ge_0p25": sum(1 for g in gaps if g >= 0.25),
         "text": " ".join(w["w"] for w in ws)}
    if st:
        d["st_med"] = round(float(np.median(st)), 2); d["st_p10"] = round(float(np.percentile(st, 10)), 2); d["st_p90"] = round(float(np.percentile(st, 90)), 2)
        d["st_range_p10_p90"] = round(d["st_p90"] - d["st_p10"], 2); d["st_max"] = round(float(max(stm)), 2) if stm else None
        d["st_sd"] = round(float(np.std(st)), 2)
    if rel:
        d["rel_db_med"] = round(float(np.median(rel)), 2); d["rel_db_p90"] = round(float(np.percentile(rel, 90)), 2); d["rel_db_sd"] = round(float(np.std(rel)), 2)
        d["peak_db_max"] = round(float(max(pk)), 1) if pk else None
    return d


def emphasis(ws):
    """top words by emphasis score = z(pitch peak) + z(loudness) + z(duration) inside their own sentence"""
    rows = []
    sts = np.array([w["stmax"] for w in ws if w.get("stmax") is not None] or [0.0]); rl = np.array([w["rel_db"] for w in ws if w.get("rel_db") is not None] or [0.0])
    du = np.array([w["e"] - w["s"] for w in ws] or [0.0])
    def z(v, arr):
        sd = arr.std() or 1.0
        return (v - arr.mean()) / sd
    for w in ws:
        if w.get("stmax") is None or w.get("rel_db") is None:
            continue
        sc = z(w["stmax"], sts) + z(w["rel_db"], rl) + 0.5 * z(w["e"] - w["s"], du)
        rows.append((round(float(sc), 2), w.get("i"), w["w"]))
    rows.sort(reverse=True)
    return rows[:3]


def pauses_around_numbers(words):
    out = []
    for i, w in enumerate(words):
        if is_num(w["w"]):
            before = round(w["s"] - words[i - 1]["e"], 2) if i > 0 else None
            after = round(words[i + 1]["s"] - w["e"], 2) if i + 1 < len(words) else None
            out.append({"num": w["w"], "t": w["s"], "pause_before": before, "pause_after": after})
    return out


def measure(wav16, words=None, pivot=None, name="", whisper_prompt=None):
    import soundfile as sf
    x, sr = sf.read(wav16, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    dur = len(x) / sr
    if words is None:
        words = whisper_words(wav16, prompt=whisper_prompt)
    t_f, f0 = pitch_track(wav16)
    t_e, db = energy_track(x, sr)
    # speaker baselines from the voiced speaking frames
    voiced = f0[~np.isnan(f0)]
    spk_f0 = float(np.median(voiced)) if len(voiced) else 150.0
    speak = np.zeros(len(t_e), bool)
    for w in words:
        speak |= (t_e >= w["s"]) & (t_e <= w["e"])
    spk_db = float(np.median(db[speak])) if speak.any() else float(np.median(db))
    words = metrics(words, t_f, f0, t_e, db, spk_f0, spk_db)
    for i, w in enumerate(words):
        w["i"] = i
    sents = sentences_of(words)
    srows = []
    prev_end = None
    for sl in sents:
        r = span_stats(sl)
        r["pause_before"] = round(sl[0]["s"] - prev_end, 2) if prev_end is not None else None
        r["emphasis"] = emphasis(sl)
        srows.append(r); prev_end = sl[-1]["e"]
    gaps = [{"after": a["w"], "before": b["w"], "t": a["e"], "gap": round(b["s"] - a["e"], 2)} for a, b in zip(words, words[1:]) if b["s"] - a["e"] >= 0.15]
    out = {"name": name, "duration": round(dur, 2), "speaker_f0_hz": round(spk_f0, 1), "speaker_db": round(spk_db, 1), "n_words": len(words), "words": words,
           "sentences": srows, "gaps_ge_0p15": gaps, "numbers": pauses_around_numbers(words), "overall": span_stats(words)}
    if pivot is not None and sents:
        # turn sentence = the sentence that contains the pivot (or ends within 0.6 s after it); story = sentences before it (after the hook); ad = after it
        ti = None
        for i, sl in enumerate(sents):
            if sl[0]["s"] - 0.3 <= pivot <= sl[-1]["e"] + 0.6:
                ti = i; break
        if ti is None:
            ti = min(range(len(sents)), key=lambda i: abs(sents[i][-1]["e"] - pivot))
        hook = sents[0]
        story = [w for sl in sents[1:ti] for w in sl]
        turn = sents[ti]
        ad = [w for sl in sents[ti + 1:] for w in sl]
        # the last second before the reveal word, the reveal word itself
        out["sections"] = {"hook": span_stats(hook), "story": span_stats(story), "turn": span_stats(turn), "ad": span_stats(ad), "turn_index": ti,
                           "hook_emphasis": emphasis(hook), "turn_emphasis": emphasis(turn), "hook_numbers": pauses_around_numbers(hook), "turn_pause_before": srows[ti]["pause_before"]}
        st, tu = out["sections"]["story"], out["sections"]["turn"]
        if st and tu and "st_med" in st and "st_med" in tu:
            out["turn_jump"] = {"pitch_med_st": round(tu["st_med"] - st["st_med"], 2), "pitch_p90_st": round(tu["st_p90"] - st["st_p90"], 2), "range_st": round(tu["st_range_p10_p90"] - st["st_range_p10_p90"], 2),
                                "energy_med_db": round(tu["rel_db_med"] - st["rel_db_med"], 2), "energy_p90_db": round(tu["rel_db_p90"] - st["rel_db_p90"], 2),
                                "wpm_net": round(tu["wpm_net"] - st["wpm_net"], 1)}
            # second yardstick: the last 2 s of the turn sentence vs the story
        tail = [w for w in turn if w["s"] >= turn[-1]["e"] - 2.0]
        tl = span_stats(tail)
        if tl and st and "st_med" in tl and "st_med" in st:
            out["turn_tail"] = {**tl, "pitch_med_vs_story_st": round(tl["st_med"] - st["st_med"], 2), "pitch_max_vs_story_p90_st": round((tl.get("st_max") or 0) - st["st_p90"], 2),
                                "energy_med_vs_story_db": round(tl["rel_db_med"] - st["rel_db_med"], 2), "energy_p90_vs_story_db": round(tl["rel_db_p90"] - st["rel_db_p90"], 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "wav"]); ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--name", default=""); ap.add_argument("--pivot", type=float, default=None); ap.add_argument("--work", default="work")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if a.mode == "ref":
        w16 = demucs_vocals(a.inp, a.work)
    else:
        w16 = os.path.join(a.work, "in16.wav"); to_wav(a.inp, w16, 16000)
    res = measure(w16, pivot=a.pivot, name=a.name)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res.get(k) for k in ("name", "duration", "n_words", "overall", "turn_jump", "turn_tail")}, indent=1)[:2500])


if __name__ == "__main__":
    main()
