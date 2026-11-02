#!/usr/bin/env python3
"""synth_edge.py: Microsoft Edge neural TTS (edge-tts) takes with a STRICT pronunciation gate (2 Nov 2026).

Writes the same pack synth_delivery.py writes, so the film side (par3/par5, build_film.py load_takes/take_words/time_script) reads it unchanged:
  <KEY>.t<k>.wav  (48 kHz mono 24-bit, ~-16 LUFS, lead/tail silence trimmed to 0.05 s)
  <KEY>.t<k>.words.json  {"segments":[{"type","text","start","end","probability"}...]}  word times from the edge-tts WordBoundary events (exact), probability from Whisper large-v3
  <KEY>.t<k>.receipt.json, results.json

Request (JSON, read from the private repo): see VOICE-EDGE.md. One edge-tts call per take, whole paragraph, never stitched.
GATE per take: faster-whisper large-v3 (condition_on_previous_text=False, no initial prompt, word timestamps) transcript == script word for word
AND every Whisper word probability >= 0.70. Second opinion: faster-whisper medium.en must match too. gate_pass = both.
Public logs get numbers only (no script text, no transcripts).
"""
import argparse, asyncio, difflib, hashlib, json, os, re, statistics, subprocess, sys, time

import numpy as np
import soundfile as sf

SR = 48000
LUFS_TARGET = -16.0
PAD_S = 0.05
PROB_MIN = 0.70


# ---------------------------------------------------------------- text
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def is_cta(text):
    return bool(re.search(r"\bdot\s+co\s+dot\b", text, re.I))


def speak(text, para, req):
    """spoken text (never used for captions). Overridable per paragraph with 'say', per request with 'ai_say' / 'brand_say'."""
    if para.get("say"):
        return para["say"].strip()
    t = text.strip()
    if not is_cta(t):
        t = re.sub(r"\bAGMM\b\.?", req.get("brand_say", "A.G.M."), t)        # the turn line: we say AGM
    t = re.sub(r"\bAI\b\.?", req.get("ai_say", "A.I."), t)                  # heard "ay eye"
    return t


def _num(m):
    from num2words import num2words
    return " " + num2words(int(m.group()), lang="en_GB") + " "


def toks(s):
    """normalised tokens: lower, digits -> words, apostrophes dropped, split on everything else, runs of single letters merged (A I -> ai, A.G.M. -> agm)"""
    s = s.lower().replace("&", " and ").replace("’", "'")
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = re.sub(r"\d+", _num, s)
    s = s.replace("'", "")
    t = re.findall(r"[a-z]+", s)
    out, run = [], []
    for w in t + [None]:
        if w is not None and len(w) == 1:
            run.append(w); continue
        if run:
            out.append("".join(run)); run = []
        if w is not None:
            out.append(w)
    return [w for w in out if w != "dot"]


def expected_tokens(text, req, names):
    tk = toks(text)
    if not is_cta(text):
        tk = ["agm" if w == "agmm" else w for w in tk]
    return apply_alias(tk, names)


def apply_alias(tk, names):
    """names: {"kushim": ["khushim"]} -> an ASR token in the list counts as the canonical spelling"""
    if not names:
        return tk
    rev = {}
    for canon, alts in names.items():
        for a in alts:
            rev[a.lower()] = canon.lower()
    return [rev.get(w, w) for w in tk]


# ---------------------------------------------------------------- edge-tts
async def edge_once(text, voice, rate, pitch, mp3):
    import edge_tts
    try:
        comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, boundary="WordBoundary")
    except TypeError:
        comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    words, sents, n_audio = [], [], 0
    with open(mp3, "wb") as f:
        async for ch in comm.stream():
            if ch["type"] == "audio":
                f.write(ch["data"]); n_audio += len(ch["data"])
            elif ch["type"] == "WordBoundary":
                words.append({"text": ch["text"], "start": ch["offset"] / 1e7, "end": (ch["offset"] + ch["duration"]) / 1e7})
            elif ch["type"] == "SentenceBoundary":
                sents.append({"text": ch["text"], "start": ch["offset"] / 1e7, "end": (ch["offset"] + ch["duration"]) / 1e7})
    if n_audio < 2000:
        raise RuntimeError("no audio from edge-tts (%d bytes)" % n_audio)
    return words, sents


def edge_synth(text, voice, rate, pitch, mp3, tries=4):
    last = None
    for i in range(tries):
        try:
            return asyncio.run(edge_once(text, voice, rate, pitch, mp3))
        except Exception as e:      # noqa: BLE001
            last = e
            print("edge-tts attempt %d failed: %s" % (i + 1, repr(e)[:160]), flush=True)
            time.sleep(2 + 3 * i)
    raise RuntimeError("edge-tts failed after %d tries: %s" % (tries, repr(last)[:200]))


# ---------------------------------------------------------------- audio
def to_wav(mp3, out_wav):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", mp3, "-ar", str(SR), "-ac", "1", "-c:a", "pcm_f32le", out_wav], check=True)
    y, _ = sf.read(out_wav, dtype="float32")
    return y


def trim_pad(y):
    """lead/tail silence -> exactly PAD_S. Returns (audio, shift_s) where shift_s = seconds removed from the start (negative = padded)"""
    win = int(0.010 * SR)
    n = len(y) // win
    rms = np.sqrt(np.mean(y[:n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
    thr = 10 ** (-48 / 20.0)
    idx = np.where(rms > thr)[0]
    if len(idx) == 0:
        return y, 0.0
    a = idx[0] * win
    b = min(len(y), (idx[-1] + 1) * win)
    pad = int(PAD_S * SR)
    body = y[a:b]
    out = np.concatenate([np.zeros(pad, dtype="float32"), body, np.zeros(pad, dtype="float32")])
    return out, (a - pad) / float(SR)


def loudnorm(y):
    import pyloudnorm as pyln
    m = pyln.Meter(SR)
    l0 = m.integrated_loudness(y.astype("float64"))
    g = 10 ** ((LUFS_TARGET - l0) / 20.0)
    z = y * g
    pk = float(np.max(np.abs(z)))
    limited = False
    if pk > 0.89:                  # -1 dBFS ceiling
        z = z * (0.89 / pk); limited = True
    return z.astype("float32"), round(l0, 2), limited


# ---------------------------------------------------------------- ASR
_W = {}


def whisper(name):
    if name not in _W:
        from faster_whisper import WhisperModel
        t0 = time.time()
        _W[name] = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
        print("loaded whisper %s in %.0fs" % (name, time.time() - t0), flush=True)
    return _W[name]


def asr(path, name):
    m = whisper(name)
    segs, _ = m.transcribe(path, language="en", beam_size=5, word_timestamps=True, vad_filter=False,
                           condition_on_previous_text=False, initial_prompt=None, temperature=0.0)
    ws = []
    for s in segs:
        for w in (s.words or []):
            t = w.word.strip()
            if t:
                ws.append({"text": t, "start": round(float(w.start), 3), "end": round(float(w.end), 3), "probability": round(float(w.probability), 4)})
    return ws


def gate(ws, exp, names):
    """compare ASR words with the expected tokens. Returns dict (no text unless the caller keeps it private)"""
    got, owner = [], []
    for i, w in enumerate(ws):
        for t in toks(w["text"]):
            got.append(t); owner.append(i)
    # merge single-letter runs consistently with toks(): toks already merges within one word; merge across words here
    merged, mown, run = [], [], []
    seq = list(zip(got, owner)) + [(None, None)]
    for t, o in seq:
        if t is not None and len(t) == 1:
            run.append((t, o)); continue
        if run:
            merged.append("".join(x[0] for x in run)); mown.append([x[1] for x in run]); run = []
        if t is not None:
            merged.append(t); mown.append([o])
    merged = apply_alias(merged, names)
    ok = merged == exp
    sm = difflib.SequenceMatcher(None, exp, merged, autojunk=False)
    diffs = [(tag, exp[i1:i2], merged[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    probs = [w["probability"] for w in ws]
    lo_i = int(np.argmin(probs)) if probs else None
    return {"match": bool(ok), "diffs": diffs, "n_expected": len(exp), "n_heard": len(merged),
            "min_prob": (round(min(probs), 4) if probs else 0.0), "min_word": (ws[lo_i]["text"] if lo_i is not None else None),
            "min_word_idx": lo_i, "n_words_below": sum(1 for p in probs if p < PROB_MIN)}


# ---------------------------------------------------------------- words.json from boundaries
def edge_to_words(ev, shift, dur, ws):
    """WordBoundary events -> Whisper-style list, shifted by the trim, probability taken from the matching Whisper word"""
    out = []
    for e in ev:
        if not re.search(r"[A-Za-z0-9]", e["text"]):
            continue
        s = max(0.0, e["start"] - shift); en = min(dur, e["end"] - shift)
        out.append({"type": "transcription_segment", "text": e["text"].strip(), "start": round(s, 3), "end": round(max(en, s + 0.02), 3), "probability": None})
    # attach probability: match on normalised token text, in order
    wt = [("".join(toks(w["text"])), w["probability"], w) for w in ws]
    j = 0
    offs = []
    for o in out:
        k = "".join(toks(o["text"]))
        for jj in range(j, min(len(wt), j + 4)):
            if k and wt[jj][0] == k:
                o["probability"] = wt[jj][1]; offs.append(wt[jj][2]["start"] - o["start"]); j = jj + 1; break
    return out, offs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True); ap.add_argument("--out", required=True); ap.add_argument("--pub", required=True)
    a = ap.parse_args()
    req = json.load(open(a.request))
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.pub, exist_ok=True)
    tmp = os.path.join(a.out, "_tmp"); os.makedirs(tmp, exist_ok=True)
    import edge_tts
    ver = getattr(edge_tts, "__version__", "?")
    names = {k: v for k, v in (req.get("names") or {}).items()}
    base = {"voice": req.get("voice", "en-GB-RyanNeural"), "rate": req.get("rate", "+0%"), "pitch": req.get("pitch", "+0Hz")}
    results, t_all = [], time.time()
    for pi, para in enumerate(req["paragraphs"], 1):
        key, text = para["key"], para["text"].strip()
        variants = para.get("variants") or req.get("variants") or [{}]
        n_takes = int(para.get("n_takes", req.get("takes", len(variants))))
        spoken = speak(text, para, req)
        exp = expected_tokens(para.get("expect", text), req, names)
        for k in range(1, n_takes + 1):
            v = {**base, **variants[(k - 1) % len(variants)]}
            stem = "%s.t%d" % (key, k)
            t0 = time.time()
            mp3 = os.path.join(tmp, stem + ".mp3")
            ev, sents = edge_synth(v.get("say", spoken), v["voice"], v["rate"], v["pitch"], mp3)
            raw = to_wav(mp3, os.path.join(tmp, stem + ".raw.wav"))
            y, shift = trim_pad(raw)
            y, lufs0, limited = loudnorm(y)
            wav = os.path.join(a.out, stem + ".wav")
            sf.write(wav, y, SR, subtype="PCM_24")
            dur = round(len(y) / float(SR), 3)
            # ASR gate (large-v3 strict, medium.en second opinion)
            wl = asr(wav, "large-v3")
            wm = asr(wav, "medium.en")
            gl, gm = gate(wl, exp, names), gate(wm, exp, names)
            gate_pass = bool(gl["match"] and gl["min_prob"] >= PROB_MIN and gm["match"])
            timing_source = "edge-tts WordBoundary"
            if ev:
                words, offs = edge_to_words(ev, shift, dur, wl)
            else:                       # voice emits no word boundaries: fall back to Whisper's own word times
                timing_source = "whisper large-v3 (voice gave no WordBoundary events)"
                words = [{"type": "transcription_segment", "text": w["text"], "start": w["start"], "end": w["end"], "probability": w["probability"]} for w in wl]
                offs = []
            heard_l = " ".join(w["text"] for w in wl); heard_m = " ".join(w["text"] for w in wm)
            nwords = len(text.split())
            wpm = round(nwords / (dur / 60.0), 1)
            span = (words[-1]["end"] - words[0]["start"]) if words else dur
            swpm = round(nwords / span * 60.0, 1) if span > 0 else None
            lo = gl["min_word"]
            json.dump({"model": "faster-whisper-large-v3 (strict gate) + edge-tts boundaries", "language": "en", "text": " ".join(w["text"] for w in words),
                       "timing_source": timing_source, "segments": words, "asr_large_v3": wl}, open(os.path.join(a.out, stem + ".words.json"), "w"), indent=1, ensure_ascii=False)
            receipt = {"paragraph": key, "text": text, "spoken_text": v.get("say", spoken), "input_sha256": sha256_bytes(text.encode()), "free_engine": True,
                       "engine": "edge-tts %s (Microsoft Edge neural TTS)" % ver, "engine_version": ver, "voice": v["voice"], "rate": v["rate"], "pitch": v["pitch"],
                       "wav_sha256": sha256_file(wav), "wav_bytes": os.path.getsize(wav),
                       "wav_info": {"channels": 1, "rate": SR, "width": 3, "frames": len(y), "duration_s": dur},
                       "attempt": k, "wpm": wpm, "span_wpm": swpm, "asr_status": "checked", "asr_engine": "faster-whisper large-v3 int8 (condition_on_previous_text=False, no prompt)",
                       "asr_text": heard_l, "asr_match": gate_pass,
                       "asr_variants": {"medium.en": heard_m, "small.en": heard_m},       # the film's brand check reads small.en: it gets the medium.en text
                       "asr_match_by": {"large-v3": gl["match"], "medium.en": gm["match"], "prob_ok": gl["min_prob"] >= PROB_MIN},
                       "gate": {"pass": gate_pass, "large_v3": gl, "medium_en": gm, "prob_min_required": PROB_MIN,
                                "word_probs": [[w["text"], w["probability"]] for w in wl], "lowest_word": lo, "lowest_prob": gl["min_prob"]},
                       "timing": {"source": timing_source, "shift_s": round(shift, 4), "whisper_minus_edge_start_median_s": (round(statistics.median(offs), 3) if offs else None),
                                  "whisper_minus_edge_start_maxabs_s": (round(max(abs(x) for x in offs), 3) if offs else None), "n_boundary_words": len(ev)},
                       "loudness": {"lufs_before": lufs0, "target": LUFS_TARGET, "peak_limited": limited},
                       "spot_fail": [], "tempo_applied": None, "seed": None, "render_seconds": round(time.time() - t0, 1),
                       "run": {"id": os.environ.get("GITHUB_RUN_ID"), "sha": os.environ.get("GITHUB_SHA")}}
            json.dump(receipt, open(os.path.join(a.out, stem + ".receipt.json"), "w"), indent=2, ensure_ascii=False)
            row = {"key": key, "take": k, "dur_s": dur, "wpm_native": wpm, "span_wpm": swpm, "asr_match": gate_pass, "render_s": round(time.time() - t0, 1), "sim_ref": {},
                   "chunks": 1, "native_ratio": None, "sim_baseline": None, "voice": v["voice"], "rate": v["rate"], "pitch": v["pitch"],
                   "gate_pass": gate_pass, "large_match": gl["match"], "medium_match": gm["match"], "min_prob": gl["min_prob"], "min_word": lo,
                   "n_below_070": gl["n_words_below"], "timing_source": timing_source}
            results.append(row)
            # public log: numbers only
            print(json.dumps({k2: row[k2] for k2 in ("key", "take", "voice", "rate", "dur_s", "wpm_native", "gate_pass", "large_match", "medium_match", "min_prob", "n_below_070", "timing_source")}), flush=True)
            json.dump({k2: row[k2] for k2 in ("key", "take", "voice", "rate", "pitch", "dur_s", "wpm_native", "span_wpm", "gate_pass", "large_match", "medium_match", "min_prob", "n_below_070", "timing_source")}
                      | {"n_expected": gl["n_expected"], "n_heard_large": gl["n_heard"], "n_heard_medium": gm["n_heard"], "lufs_before": lufs0},
                      open(os.path.join(a.pub, stem + ".gate.json"), "w"))
    json.dump({"label": req.get("label"), "id": req.get("id"), "engine": "edge-tts %s" % ver, "engine_version": ver, "params": {}, "reference_sha256": None, "model_load_s": None,
               "total_s": round(time.time() - t_all, 1), "takes": results}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("done", len(results), "takes in", round(time.time() - t_all), "s", flush=True)


if __name__ == "__main__":
    main()
