#!/usr/bin/env python3
"""synth.py - free-engine TTS on Actions: paragraphs in (private request), takes + receipts + words + speaker-similarity numbers out (voice-free 20 Oct 2026).

  synth.py --engine chatterbox --request REQUEST.json --private PRIVATE_CHECKOUT --out OUT_DIR

REQUEST.json (lives in the PRIVATE repo, never in this public one: script text must not reach public logs):
  {"label": str, "id": "S95", "ref": "voice-free/ref/....wav", "ref_sims": [paths], "takes": 2, "params": {...},
   "paragraphs": [{"key": "P01", "text": "..."}], "baseline": {"P01": "path of the Mistral PASS paragraph wav"}}
Per paragraph and take: sentence-chunked synthesis (a chunk is 1-2 sentences, the pause between chunks is explicit), 24 kHz mono PCM16, faster-whisper
word timings in the Mistral response shape, the native asr_compare verdict, native and span wpm, render seconds, resemblyzer cosine to Sam's reference
set and to the baseline take. Output files: <key>.t<k>.wav / .words.json / .receipt.json and results.json (numbers only).
"""
import argparse, datetime as dt, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf_common as C


class Chatterbox:
    name = "chatterbox"
    version = "chatterbox-tts 0.1.7 (MIT) + ResembleAI/chatterbox weights"

    def __init__(self, ref, params):
        import torch
        from chatterbox.tts import ChatterboxTTS
        torch.set_num_threads(os.cpu_count() or 4)
        self.torch = torch
        self.p = params
        t = time.time()
        self.m = ChatterboxTTS.from_pretrained(device="cpu")
        self.load_s = round(time.time() - t, 1)
        self.m.prepare_conditionals(ref, exaggeration=params.get("exaggeration", 0.5))
        self.sr = self.m.sr

    def synth(self, text, seed):
        self.torch.manual_seed(seed)
        p = self.p
        wav = self.m.generate(text, repetition_penalty=p.get("repetition_penalty", 1.2), min_p=p.get("min_p", 0.05), top_p=p.get("top_p", 1.0),
                              exaggeration=p.get("exaggeration", 0.5), cfg_weight=p.get("cfg_weight", 0.5), temperature=p.get("temperature", 0.8))
        return wav.squeeze(0).cpu().numpy().astype("float32")


class FishSpeech:
    """Not wired. fish_gate.py refuses the only Fish weights that fit a CPU runner (openaudio-s1-mini is CC-BY-NC-SA-4.0 and gated; S2-Pro is Research
    Licence; Drama 3 is 35B). Fill this in only after Sam supplies licensed weights and an HF token (secret HF_TOKEN_FISH)."""
    name = "fish"
    version = "unwired"

    def __init__(self, ref, params):
        raise SystemExit("fish engine not wired: no commercially licensed Fish weights are runnable on a CPU runner (see fish_gate.py)")


ENGINES = {"chatterbox": Chatterbox, "fish": FishSpeech}


def chunks(text, max_words):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    out, cur = [], []
    for s in sents:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            out.append(" ".join(cur)); cur = [s]
        else:
            cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def trim(a, sr, rel_db=-50.0, keep_head=0.04, keep_tail=0.14):
    """Trim edge silence RELATIVE to the chunk's own level (a fixed -42 dBFS floor cut the quiet tail of final words such as 'documents', 20 Oct: the native
    spot-check then found the last word's window silent). Keeps a 0.14 s tail."""
    if len(a) == 0:
        return a
    w = int(0.01 * sr)
    n = len(a) // w
    if n < 3:
        return a
    rms = 20 * np.log10(np.sqrt((a[:n * w].reshape(n, w) ** 2).mean(axis=1)) + 1e-9)
    floor = max(float(rms.max()) + rel_db, -75.0)
    idx = np.where(rms > floor)[0]
    if len(idx) == 0:
        return a
    s = max(0, idx[0] * w - int(keep_head * sr)); e = min(len(a), (idx[-1] + 1) * w + int(keep_tail * sr))
    return a[s:e]


def normalise_lufs(a, sr, target=-18.0):
    """Chatterbox renders about 15 dB quieter than the Mistral takes the native chain was built around (median word -37 dBFS). The assembler's silence detector uses an
    ABSOLUTE -45 dB floor, so quiet trailing words were cut as 'silence' (S73, 'documents.'). Bring each take to the -18 LUFS the final mix targets before anything else."""
    import pyloudnorm as pyln
    try:
        loud = pyln.Meter(sr).integrated_loudness(a.astype("float64"))
    except Exception:      # noqa: BLE001
        return a
    if not np.isfinite(loud):
        return a
    y = a * (10 ** ((target - loud) / 20.0))
    pk = float(np.max(np.abs(y)))
    if pk > 0.97:
        y = np.tanh(y / 0.97) * 0.97          # soft limit: only the rare peak above full scale is touched
    return y.astype("float32")


NATIVE_FILMS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voice-native", "films", "voice", "tooling"))


def fold_equal(text, heard, names=None):
    """Delta 7: True when the script and the transcript are the SAME word sequence after the whole-file check's own folds (asr_fold.fold_pair: UK/US spelling, name possessives and
    phonetic name aliases, pound amounts, ordinals 16th == sixteenth, spoken years). Zero diffs only, so this is never looser than the whole-file rule; numbers must be equal, not within 2%.
    asr_compare.compare is strict about token COUNT, so "16th" (read as 16 + th) failed take after take (S256 P02, 26 Oct) although the audio was right."""
    if NATIVE_FILMS not in sys.path:
        sys.path.insert(0, NATIVE_FILMS)
    try:
        from asr_fold import fold_pair
        from post_checks_local import fold, _canon
    except ImportError:
        return False
    a, b, _ = fold_pair(fold(text), fold(heard), "\n".join(names or []), _canon)
    return bool(a) and [str(t) for t in a] == [str(t) for t in b]


def native_ratio(text, heard, names=None):
    """The hosted chain's own whole-file ASR rule (films/voice/tooling/post_checks_local.py): local small.en, no prompt, folded script-vs-heard word diff, ratio >= 0.97 or the
    short fails (S255, S268 20-26 Oct: names such as Dunelm, Royal Mail). Applied here per paragraph so a take that would fail that rule is re-rolled before it is sealed.
    Returns (ratio, first diffs)."""
    import difflib
    if NATIVE_FILMS not in sys.path:
        sys.path.insert(0, NATIVE_FILMS)
    from post_checks_local import fold, _canon
    try:        # delta 7: the same folds the hosted check applies (asr_fold), so a per-take ratio predicts the whole-file ratio
        from asr_fold import fold_pair
        ta, tb, _ = fold_pair(fold(text), fold(heard), "\n".join(names or []), _canon)
        a, b = [str(t) for t in ta], [str(t) for t in tb]
    except ImportError:
        a = [str(t) for t in _canon(fold(text))]
        b = [str(t) for t in _canon(fold(heard))]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    diffs = [{"script": " ".join(a[i1:i2]), "heard": " ".join(b[j1:j2])} for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
    return round(sm.ratio(), 4), diffs[:8]


def word_levels(x, sr, words):
    rows = []
    for w in words:
        seg = x[int(w["start"] * sr):int(w["end"] * sr)]
        rows.append(None if len(seg) == 0 else 20 * np.log10(np.sqrt(float((seg ** 2).mean())) + 1e-9))
    vals = [d for d in rows if d is not None and d > -90]
    return rows, (float(np.median(vals)) if vals else None)


def gain_ride(wav_path, words, lift_below=18.0, max_lift=10.0):
    """Production gain-riding: a word more than `lift_below` dB under the take's median word level (Chatterbox lets the last word of a sentence trail off to -45..-50
    dB, which the native spot-check calls silence) is lifted to median - lift_below, by at most `max_lift` dB, with 25 ms raised-cosine edges. Words that would need
    more are left alone and fail spot_fail. Writes the wav in place; returns the list of lifts."""
    import soundfile as sf
    x, sr = sf.read(wav_path, dtype="float32")
    rows, med = word_levels(x, sr, words)
    if med is None:
        return []
    lifts, g = [], np.ones(len(x), dtype="float32")
    ramp = int(0.025 * sr)
    for w, db in zip(words, rows):
        if db is None or db <= -90 or db >= med - lift_below:
            continue
        gain_db = med - lift_below - db
        if gain_db > max_lift:
            continue
        lin = 10 ** (gain_db / 20.0)
        a, b = max(0, int(w["start"] * sr) - ramp), min(len(x), int(w["end"] * sr) + ramp)
        env = np.ones(b - a, dtype="float32") * lin
        n = min(ramp, (b - a) // 2)
        if n > 0:
            r = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n)).astype("float32")
            env[:n] = 1 + (lin - 1) * r
            env[-n:] = 1 + (lin - 1) * r[::-1]
        g[a:b] = np.maximum(g[a:b], env)
        lifts.append({"word": w["text"], "start": w["start"], "lift_db": round(gain_db, 1)})
    if lifts:
        y = np.clip(x * g, -0.99, 0.99)
        sf.write(wav_path, y, sr, subtype="PCM_16")
    return lifts


def spot_fail(wav_path, words, floor_db=-44.0, final_median_db=-20.0):
    """The native spotcheck_words rule (RMS inside a word's window must be above -50 dB) applied to EVERY word with a 4 dB margin, on the level the native chain will see:
    the assembler loudness-normalises to -18 LUFS, which puts the median word at about -20 dB, so each word's RMS is shifted by (final_median_db - this take's median word
    RMS) first. Returns the failing words (empty window or digital silence included)."""
    import soundfile as sf
    x, sr = sf.read(wav_path, dtype="float32")
    rows, med = word_levels(x, sr, words)
    gain = final_median_db - med if med is not None else 0.0
    bad = []
    for w, db in zip(words, rows):
        if db is None:
            bad.append({"word": w["text"], "start": w["start"], "why": "empty window"})
        elif db + gain <= floor_db:
            bad.append({"word": w["text"], "start": w["start"], "end": w["end"], "rms_db_final": round(db + gain, 1)})
    return bad


DIGITS = {'zero': '0', 'oh': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'}


def soft_match(text, asr_text):
    """Same two comparator false-negatives api_queue.soft_match accepts (spacing only; a spelled digit run written as digits)."""
    def alnum(t): return re.sub(r'[^a-z0-9]', '', t.lower())
    if asr_text and alnum(text) == alnum(asr_text):
        return 'spacing-only difference'
    ws = re.findall(r"[a-z0-9']+", text.lower())
    out, run = [], []
    for w in ws + ['']:
        if w in DIGITS:
            run.append(DIGITS[w]); continue
        if run:
            out.append(''.join(run)); run = []
        if w:
            out.append(w)
    if asr_text and alnum(' '.join(out)) == alnum(asr_text) and any(len(x) >= 2 and x.isdigit() for x in out):
        return 'spelled digit run equals transcribed digits'
    return ''


def stretch(path, factor):
    """Pitch-preserving slow-down with ffmpeg atempo (factor < 1 slows). In place, 24 kHz mono PCM16."""
    import subprocess
    tmp = path + ".st.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path, "-af", "atempo=%.4f" % factor, "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", tmp], check=True)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True); ap.add_argument("--request", required=True); ap.add_argument("--private", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    req = json.load(open(a.request))
    os.makedirs(a.out, exist_ok=True)
    P = req.get("params") or {}
    ref = os.path.join(a.private, req["ref"])
    ref_sha = C.sha256_file(ref)
    t_all = time.time()
    if req.get("sim_only"):       # measure the baseline takes against Sam's reference set; no synthesis
        ref_embs = {os.path.basename(p): C.embed(os.path.join(a.private, p)) for p in (req.get("ref_sims") or [req["ref"]])}
        out = {}
        for k, p in (req.get("baseline") or {}).items():
            e = C.embed(os.path.join(a.private, p))
            out[k] = {n: round(C.cosine(e, r), 4) for n, r in ref_embs.items()}
        json.dump({"label": req.get("label"), "id": req.get("id"), "baseline_sims": out}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
        print("sim_only done", len(out), flush=True)
        return
    if req.get("align_only"):       # re-time existing takes: {"align": [{"key","wav","words"}]} (private paths); writes <key>.words.json
        res = {}
        for it in req["align"]:
            w = json.load(open(os.path.join(a.private, it["words"])))
            segs, info = C.align_words(os.path.join(a.private, it["wav"]), w["segments"])
            w["segments"] = segs
            w["timing"] = info
            json.dump(w, open(os.path.join(a.out, it["key"] + ".words.json"), "w"), indent=2, ensure_ascii=False)
            res[it["key"]] = info
            print(it["key"], json.dumps(info), flush=True)
        json.dump({"label": req.get("label"), "id": req.get("id"), "align": res}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
        return
    eng = ENGINES[a.engine](ref, P)
    print("engine", eng.name, "loaded in", eng.load_s, "s; ref", os.path.basename(ref), flush=True)
    compare = C.load_compare()
    ref_embs = {os.path.basename(p): C.embed(os.path.join(a.private, p)) for p in (req.get("ref_sims") or [req["ref"]])}
    base_emb = {}
    for k, p in (req.get("baseline") or {}).items():
        base_emb[k] = C.embed(os.path.join(a.private, p))
    names = req.get("names") or []
    name_prompt = ("Names and terms in this script: " + ", ".join(names) + ".") if names else None
    max_words = int(P.get("chunk_words", 32))
    pause_sent, pause_mid = float(P.get("pause_sentence_s", 0.22)), float(P.get("pause_clause_s", 0.10))
    budget_s = float(req.get("budget_minutes", 45)) * 60
    results = []
    for takei in range(1, int(req.get("takes", 1)) + 1):
        for pi, para in enumerate(req["paragraphs"], 1):
            if time.time() - t_all > budget_s:
                print("time budget reached, stopping", flush=True); break
            key, text = para["key"], para["text"].strip()
            seed = int(P.get("seed", 1234)) + 1000 * takei + pi
            t0 = time.time()
            pieces = []
            variants = para.get("tts_texts") or [text]          # pronunciation hints go to the TTS input only; receipts and ASR comparison use the script text
            tts_idx = (takei - 1) % len(variants)
            cs = chunks(variants[tts_idx], max_words)
            for ci, ch in enumerate(cs):
                audio = trim(eng.synth(ch, seed + 97 * ci), eng.sr)
                pieces.append(audio)
                if ci < len(cs) - 1:
                    pieces.append(np.zeros(int(pause_sent * eng.sr), dtype="float32"))
            full = normalise_lufs(np.concatenate(pieces), eng.sr)
            render_s = round(time.time() - t0, 1)
            stem = "%s.t%d" % (key, takei)
            wav_path = os.path.join(a.out, stem + ".wav")
            C.write_wav24k(wav_path, full, eng.sr)
            info = C.wav_info(wav_path)
            tempo = None
            short_line = len(text.split()) <= 12
            exact = P.get("exact_wpm")           # film voice-over (F20): land the take on one pace, whatever its length
            if exact:
                raw0 = len(text.split()) / (info["duration_s"] / 60.0)
                if abs(raw0 - float(exact)) / float(exact) > 0.012:
                    tempo = min(max(float(exact) / raw0, float(P.get("exact_min_tempo", 0.8))), float(P.get("exact_max_tempo", 1.25)))
                    stretch(wav_path, tempo)
                    info = C.wav_info(wav_path)
            target = None if exact else ((P.get("target_wpm_short") if short_line else None) or P.get("target_wpm"))
            min_tempo = float(P.get("min_tempo_short", 0.78)) if short_line else float(P.get("min_tempo", 0.85))
            if target:
                raw = len(text.split()) / (info["duration_s"] / 60.0)
                if raw > float(target) * 1.03:
                    # hosted voice_lead_check FAILS a CTA above 195 wpm and a paragraph at 190 wpm with 6.6 syl/s (S149, S201 20 Oct): slow fast takes, CTAs further
                    tempo = max(float(target) / raw, min_tempo)
                    stretch(wav_path, tempo)
                    info = C.wav_info(wav_path)
                elif P.get("min_wpm") and len(text.split()) > 12 and raw < float(P["min_wpm"]) * 0.97:
                    # a slow paragraph (S119 20 Oct: raw speaking 146 wpm, which the assembler's 1.10 x ceiling cannot lift into the 164-173 band) is sped up a little here
                    tempo = min(float(P["min_wpm"]) / raw, float(P.get("max_speedup", 1.06)))
                    stretch(wav_path, tempo)
                    info = C.wav_info(wav_path)
            assert (info["channels"], info["rate"], info["width"]) == (1, 24000, 2), info
            # two transcribers, as voice_lead_check does: a paragraph passes if either matches the script (small.en mishears names such as Cora as Quora)
            resp = C.asr_words(wav_path, "medium.en", prompt=name_prompt)
            resp_s = C.asr_words(wav_path, "small.en")
            for _r in (resp, resp_s):          # whisper writes "10 .8 million" for "ten point eight million"
                _r["text"] = re.sub(r"(\d)\s+,(\d)", r"\1,\2", re.sub(r"(\d)\s+\.(\d)", r"\1.\2", _r["text"]))   # "10 .8", "638 ,000"
            resp["segments"], align_info = C.align_words(wav_path, resp["segments"])
            es = P.get("exact_span_wpm")          # film VO: the pace window is on the word span (first word start to last word end), so correct the take once more on that measure
            if es:
                sp0 = C.span_wpm(resp, text)
                if sp0 and abs(sp0 - float(es)) / float(es) > 0.008:
                    t2 = float(es) / sp0
                    stretch(wav_path, t2)
                    info = C.wav_info(wav_path)
                    for _w in resp["segments"]:
                        _w["start"], _w["end"] = round(_w["start"] / t2, 3), round(_w["end"] / t2, 3)
                    tempo = (tempo or 1.0) * t2
            ok_m, detail = compare(text, resp["text"])
            ok_s, detail_s = compare(text, resp_s["text"])
            sm = "" if (ok_m or ok_s) else (soft_match(text, resp["text"]) or soft_match(text, resp_s["text"])
                                            or ("fold-equal: identical after the whole-file folds" if (fold_equal(text, resp["text"], names) or fold_equal(text, resp_s["text"], names)) else ""))
            rides = gain_ride(wav_path, resp["segments"])
            if rides:
                info = C.wav_info(wav_path)
            nat_ratio, nat_diffs = native_ratio(text, resp_s["text"], names)
            sp_bad = spot_fail(wav_path, resp["segments"])
            ok = bool(ok_m or ok_s or sm) and not sp_bad          # a silent word window fails the take exactly as the native chain would
            if not ok_m and ok_s:
                detail = {"matched_by": "small.en", "medium_en": detail}
            elif sm:
                detail = {"matched_by": "soft rule: " + sm}
            wpm_native = round(len(text.split()) / (info["duration_s"] / 60.0), 1)
            swpm = C.span_wpm(resp, text)
            emb = C.embed(wav_path)
            sims = {n: round(C.cosine(emb, e), 4) for n, e in ref_embs.items()}
            sim_base = round(C.cosine(emb, base_emb[key]), 4) if key in base_emb else None
            receipt = {"paragraph": key, "text": text, "input_sha256": C.sha256_bytes(text.encode()),
                       "free_engine": True, "engine": eng.name, "engine_version": eng.version, "free_route": "github-actions-cpu (public repo, encrypted artifact)",
                       "reference_file": os.path.basename(ref), "reference_sha256": ref_sha,
                       "wav_sha256": C.sha256_file(wav_path), "wav_bytes": os.path.getsize(wav_path), "wav_info": {k: info[k] for k in ("channels", "rate", "width", "frames", "duration_s")},
                       "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "attempt": takei, "wpm": wpm_native, "span_wpm": swpm,
                       "asr_status": "checked", "asr_engine": resp["model"], "asr_text": resp["text"], "asr_match": bool(ok), "asr_compare_detail": detail,
                       "asr_variants": {"small.en": resp_s["text"]}, "asr_match_by": {"medium.en": bool(ok_m), "small.en": bool(ok_s), "soft": sm or None}, "tempo_applied": tempo, "timing": align_info, "spot_fail": sp_bad, "gain_ride": rides, "native_ratio": nat_ratio, "native_diffs": nat_diffs,
                       "tts_http_status": None, "params": P, "seed": seed, "chunks": len(cs), "tts_variant": tts_idx, "tts_input_differs": variants[tts_idx] != text, "render_seconds": render_s,
                       "speaker_sim": sims, "speaker_sim_to_baseline": sim_base,
                       "run": {"id": os.environ.get("GITHUB_RUN_ID"), "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "sha": os.environ.get("GITHUB_SHA"),
                               "url": "https://github.com/%s/actions/runs/%s" % (os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GITHUB_RUN_ID", ""))}}
            json.dump(resp, open(os.path.join(a.out, stem + ".words.json"), "w"), indent=2, ensure_ascii=False)
            json.dump(receipt, open(os.path.join(a.out, stem + ".receipt.json"), "w"), indent=2, ensure_ascii=False)
            row = {"key": key, "take": takei, "dur_s": info["duration_s"], "wpm_native": wpm_native, "span_wpm": swpm, "asr_match": bool(ok), "render_s": render_s,
                   "rtf": round(render_s / info["duration_s"], 2), "sim_ref": sims, "sim_baseline": sim_base, "chunks": len(cs), "native_ratio": nat_ratio}
            results.append(row)
            print(json.dumps(row), flush=True)
    json.dump({"label": req.get("label"), "id": req.get("id"), "engine": eng.name, "engine_version": eng.version, "params": P, "reference_sha256": ref_sha,
               "model_load_s": eng.load_s, "total_s": round(time.time() - t_all, 1), "takes": results}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print("done", len(results), "takes in", round(time.time() - t_all), "s", flush=True)


if __name__ == "__main__":
    main()
