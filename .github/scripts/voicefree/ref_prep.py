#!/usr/bin/env python3
"""ref_prep.py - cut clean single-speaker 8-15 s reference segments of Sam's voice from his TikTok audio (p01, voice-free 20 Oct 2026).

Runs on GitHub Actions (CPU). Inputs: a directory of audio-only mp3s of Sam's own @samai4233 videos (his consent, 20 Oct) and the hash-locked
reference. Output: segments (24 kHz mono PCM16 WAV), manifest.json (a LIST: file, source, start, end, transcript, sha256, + measured quality) and
summary.json (what was rejected and why). Nothing printed to the log contains a transcript.

Rejection rules (each can fail, counted in summary.json):
  music    - (all seven TikToks carry a bed, so the gate is relative and the kept segment is the Demucs VOCALS stem, original kept beside it)
             Demucs (htdemucs, MIT) splits each file into vocals and no_vocals; over the word frames of a window, music_gap_db = median vocals dB
             minus median no_vocals dB. Music (or any bed) under speech shrinks the gap. Reject below GAP_MIN, which is set against the clean
             hash-locked reference run through the same separation (reported in summary.json).
  speaker  - resemblyzer cosine of the window to the locked reference below SIM_MIN, or any 3 s slice of the window below SLICE_MIN of the window mean
             (a second voice, a duet, a stitched clip).
  clip     - peak above 0.98.
  length   - outside 8-15 s.
"""
import argparse, glob, json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vf_common as C

GAP_MIN, SIM_MIN, SLICE_MIN = 6.0, 0.70, 0.75   # 20 Oct run 1: Sam's TikToks all carry a music bed (best windows 15 dB under the voice, most 0-3 dB); the clean locked reference measures 36.7 dB
MIN_S, MAX_S = 8.0, 15.0
GAP_BREAK = 0.30


def decode(src, out, sr, ac=1):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-ac", str(ac), "-ar", str(sr), "-c:a", "pcm_s16le", out], check=True)


def separate(wav, outdir):
    """Demucs two-stem split on CPU. Returns (vocals float array, no_vocals float array, sr), both mono at the stem rate."""
    import soundfile as sf
    subprocess.run([sys.executable, "-m", "demucs.separate", "-n", "htdemucs", "--two-stems", "vocals", "-d", "cpu", "--float32", "-o", outdir, wav], check=True,
                   stdout=subprocess.DEVNULL)
    stem = os.path.join(outdir, "htdemucs", os.path.splitext(os.path.basename(wav))[0])
    v, sr = sf.read(os.path.join(stem, "vocals.wav")); n, sr2 = sf.read(os.path.join(stem, "no_vocals.wav"))
    mono = lambda a: (a if a.ndim == 1 else a.mean(axis=1)).astype("float32")
    return mono(v), mono(n), sr


def frames_db(x, sr, win=0.025, hop=0.010):
    n, h = int(win * sr), int(hop * sr)
    cnt = 1 + max(0, (len(x) - n) // h)
    out = np.empty(cnt, dtype="float32")
    for i in range(cnt):
        seg = x[i * h:i * h + n]
        out[i] = 20 * np.log10(np.sqrt(np.mean(seg * seg)) + 1e-9)
    return out


def metrics(x, voc, nov, sr, words, t0, t1):
    """x: whole-file float array (24 kHz); voc/nov: Demucs stems of the same file at the stem rate (sr_s). words: [(start,end)] inside [t0,t1].
    -> music_gap_db (vocals minus no_vocals, median over word frames), snr_db (informational), peak."""
    seg = x[int(t0 * 24000):int(t1 * 24000)]
    sv, sn = voc[int(t0 * sr):int(t1 * sr)], nov[int(t0 * sr):int(t1 * sr)]
    dbv, dbn = frames_db(sv, sr), frames_db(sn, sr)
    db = frames_db(seg, 24000)
    m = min(len(dbv), len(dbn))
    mask = np.zeros(m, dtype=bool)
    for (a, b) in words:
        i0 = max(0, int((a - t0) / 0.010)); i1 = min(m, int((b - t0) / 0.010) + 1)
        mask[i0:i1] = True
    if mask.sum() < 20:
        return None
    mk = mask[:len(db)]
    snr = float(np.median(db[:len(mk)][mk]) - np.percentile(db[:len(mk)][~mk], 10)) if (~mk).sum() >= 10 and mk.sum() >= 10 else None
    return {"music_gap_db": round(float(np.median(dbv[:m][mask]) - np.median(dbn[:m][mask])), 1), "vocal_db": round(float(np.median(dbv[:m][mask])), 1),
            "bed_db": round(float(np.median(dbn[:m][mask])), 1), "snr_db": None if snr is None else round(snr, 1), "peak": round(float(np.max(np.abs(seg))), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--locked", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--gap-min", type=float, default=GAP_MIN)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, "_work"); os.makedirs(work, exist_ok=True)
    import soundfile as sf
    t_start = time.time()
    locked_emb = C.embed(a.locked)
    # calibration: the clean hash-locked reference through the same separation and metric (whole file as one window of word frames = all frames above -50 dB)
    l24 = os.path.join(work, "locked.24k.wav"); decode(a.locked, l24, 24000)
    lx, _ = sf.read(l24); lx = lx.astype("float32")
    lv, ln, lsr = separate(l24, os.path.join(work, "dm"))
    ldb = frames_db(lx, 24000)
    lwords = [(i * 0.010, i * 0.010 + 0.010) for i in range(len(ldb)) if ldb[i] > -45]
    lm = metrics(lx, lv, ln, lsr, lwords, 0.0, len(lx) / 24000)
    locked_stats = {"file": os.path.basename(a.locked), "sha256": C.sha256_file(a.locked), "duration_s": round(len(lx) / 24000, 2), "calibration_metrics": lm}
    print("locked reference calibration", lm, flush=True)
    voc24_by_src = {}
    cands, rejects, per_src, gap_by_src = [], {"music": 0, "speaker": 0, "clip": 0, "no_words": 0}, {}, {}
    for src in sorted(glob.glob(os.path.join(a.src, "*.mp3"))):
        base = os.path.splitext(os.path.basename(src))[0]
        w16, w24 = os.path.join(work, base + ".16k.wav"), os.path.join(work, base + ".24k.wav")
        decode(src, w16, 16000); decode(src, w24, 24000)
        resp = C.asr_words(w16)
        words = [(w["start"], w["end"], w["text"]) for w in resp["segments"]]
        x, sr24 = sf.read(w24); x = x.astype("float32")
        dur = len(x) / sr24
        per_src[base] = {"duration_s": round(dur, 1), "words": len(words)}
        if not words:
            rejects["no_words"] += 1; continue
        voc, nov, ssr = separate(w24, os.path.join(work, "dm"))
        import librosa
        voc24_by_src[base] = librosa.resample(voc, orig_sr=ssr, target_sr=24000).astype("float32")
        units, cur = [], [words[0]]
        for w in words[1:]:
            if w[0] - cur[-1][1] >= GAP_BREAK or cur[-1][2].rstrip('"\')').endswith(('.', '?', '!')):
                units.append(cur); cur = [w]
            else:
                cur.append(w)
        units.append(cur)
        for i in range(len(units)):
            for j in range(i, len(units)):
                t0, t1 = units[i][0][0], units[j][-1][1]
                t0p, t1p = max(0.0, t0 - 0.15), min(dur, t1 + 0.25)
                d = t1p - t0p
                if d > MAX_S:
                    break
                if d < MIN_S:
                    continue
                ws = [(w[0], w[1]) for u in units[i:j + 1] for w in u]
                m = metrics(x, voc, nov, ssr, ws, t0p, t1p)
                if m is None:
                    rejects["no_words"] += 1; continue
                gap_by_src.setdefault(base, []).append(m["music_gap_db"])
                if m["peak"] > 0.98:
                    rejects["clip"] += 1; continue
                if m["music_gap_db"] < a.gap_min:
                    rejects["music"] += 1; continue
                cands.append({"source": base, "t0": round(t0p, 2), "t1": round(t1p, 2), "metrics": m, "text": " ".join(w[2] for u in units[i:j + 1] for w in u), "w24": w24})
    for b, v in gap_by_src.items():
        per_src[b]["music_gap_p10_p50_p90"] = [round(float(np.percentile(v, q)), 1) for q in (10, 50, 90)]; per_src[b]["windows"] = len(v)
        print("source", b[-6:], "windows", len(v), "music_gap_db p10/p50/p90", per_src[b]["music_gap_p10_p50_p90"], flush=True)
    print("candidates passing music/clip:", len(cands), "rejects so far:", rejects, flush=True)
    cands.sort(key=lambda c: -c["metrics"]["music_gap_db"])
    kept = []
    for c in cands[:160]:
        tmp = os.path.join(work, "cand.wav")
        sr = 24000
        v = voc24_by_src[c["source"]]
        seg = v[int(c["t0"] * sr):int(c["t1"] * sr)]
        sf.write(tmp, seg, sr, subtype="PCM_16")
        e = C.embed(tmp)
        sim = C.cosine(e, locked_emb)
        slice_min = 1.0
        step = int(3 * sr)
        for k in range(0, max(1, len(seg) - step + 1), step):
            sl = os.path.join(work, "sl.wav"); sf.write(sl, seg[k:k + step], sr, subtype="PCM_16")
            slice_min = min(slice_min, C.cosine(C.embed(sl), e))
        c["sim_locked"], c["slice_min"] = round(sim, 3), round(slice_min, 3)
        if sim < SIM_MIN or slice_min < SLICE_MIN:
            rejects["speaker"] += 1; continue
        c["score"] = min(c["metrics"]["music_gap_db"], 40) / 10.0 + 4 * sim + 2 * slice_min
        kept.append(c)
    kept.sort(key=lambda c: -c["score"])
    chosen, per = [], {}
    for c in kept:
        if per.get(c["source"], 0) >= 2:
            continue
        if any(c["source"] == o["source"] and not (c["t1"] <= o["t0"] or c["t0"] >= o["t1"]) for o in chosen):
            continue
        chosen.append(c); per[c["source"]] = per.get(c["source"], 0) + 1
        if len(chosen) >= a.n:
            break
    manifest = []
    segdir = os.path.join(a.out, "segments"); os.makedirs(segdir, exist_ok=True)
    for k, c in enumerate(chosen, 1):
        name = "sam-seg%02d-%s-%05d.wav" % (k, c["source"][-6:], int(c["t0"] * 100))
        path = os.path.join(segdir, name)
        v = voc24_by_src[c["source"]]; sr = 24000
        vs = v[int(c["t0"] * sr):int(c["t1"] * sr)]
        sf.write(path, vs / max(1.0, float(np.max(np.abs(vs))) / 0.95), sr, subtype="PCM_16")
        opath = os.path.join(segdir, name.replace(".wav", ".orig.wav"))
        x, _ = sf.read(c["w24"]); sf.write(opath, x[int(c["t0"] * sr):int(c["t1"] * sr)], sr, subtype="PCM_16")
        m = c["metrics"]
        manifest.append({"file": "segments/" + name, "file_original": "segments/" + name.replace(".wav", ".orig.wav"), "sha256_original": C.sha256_file(opath), "kind": "demucs vocals stem", "source": c["source"] + ".mp3", "start": c["t0"], "end": c["t1"], "duration_s": round(c["t1"] - c["t0"], 2),
                         "transcript": c["text"], "sha256": C.sha256_file(path), "music_gap_db": m["music_gap_db"], "vocal_db": m["vocal_db"], "bed_db": m["bed_db"],
                         "snr_db": m["snr_db"], "peak": m["peak"], "sim_to_locked_ref": c["sim_locked"], "slice_min_sim": c["slice_min"]})
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    json.dump({"locked_reference": locked_stats, "sources": per_src, "rejects": rejects, "candidates_after_music": len(cands), "speaker_screened": min(len(cands), 160),
               "kept_after_speaker": len(kept), "chosen": len(manifest), "thresholds": {"music_gap_min_db": a.gap_min, "sim_min": SIM_MIN, "slice_min": SLICE_MIN},
               "elapsed_s": round(time.time() - t_start, 1)}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    import shutil; shutil.rmtree(work, ignore_errors=True)
    print("chosen", len(manifest), "rejects", rejects, "elapsed", round(time.time() - t_start), "s", flush=True)
    for m in manifest:
        print(m["file"], "dur", m["duration_s"], "gap", m["music_gap_db"], "sim", m["sim_to_locked_ref"], "slice_min", m["slice_min_sim"], flush=True)
    if len(manifest) < 3:
        sys.exit("fewer than 3 clean segments: thresholds too strict or sources are music-heavy (see summary distribution above)")


if __name__ == "__main__":
    main()
