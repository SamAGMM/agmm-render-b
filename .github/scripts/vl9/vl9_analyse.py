#!/usr/bin/env python3
"""VL9 voice lab: prosody measurements per take (numbers only, no script text).
Reads synth_edge.py output (<KEY>.t<k>.wav + results.json), writes analysis.json.
F0 via Praat autocorrelation (parselmouth), male range 70-350 Hz."""
import argparse, glob, json, math, os
import numpy as np, soundfile as sf, parselmouth
from parselmouth.praat import call


def measure(wav, n_words):
    y, sr = sf.read(wav)
    if y.ndim > 1:
        y = y.mean(axis=1)
    dur = len(y) / float(sr)
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=70, pitch_ceiling=350)
    f0 = pitch.selected_array["frequency"]
    v = f0[f0 > 0]
    out = {"dur_s": round(dur, 3)}
    if len(v) > 10:
        med = float(np.median(v))
        st = 12 * np.log2(v / med)
        out.update({"f0_mean_hz": round(float(v.mean()), 1), "f0_median_hz": round(med, 1),
                    "f0_range_st_p5_p95": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2),
                    "f0_range_st_p2_p98": round(float(np.percentile(st, 98) - np.percentile(st, 2)), 2),
                    "f0_sd_st": round(float(st.std()), 2),
                    "voiced_frac": round(len(v) / float(len(f0)), 3)})
        # contour: consecutive voiced frames, jumps > 4 st (octave errors / warble proxy), and droop first quarter vs last quarter
        cont = f0.copy(); d = []
        for i in range(1, len(cont)):
            if cont[i] > 0 and cont[i - 1] > 0:
                d.append(abs(12 * math.log2(cont[i] / cont[i - 1])))
        d = np.array(d) if d else np.array([0.0])
        out["f0_jump_frac_gt4st"] = round(float((d > 4).mean()), 4)
        out["f0_mean_abs_delta_st_per_10ms"] = round(float(d.mean()), 3)
        q = max(1, len(st) // 4)
        out["f0_droop_st_last_q_minus_first_q"] = round(float(st[-q:].mean() - st[:q].mean()), 2)
    # jitter / shimmer / HNR (artefact and voice quality)
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 70, 350)
        out["jitter_local"] = round(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3), 4)
        out["shimmer_local"] = round(call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6), 4)
        hnr = snd.to_harmonicity_cc(0.01, 70, 0.1, 1.0)
        out["hnr_db"] = round(call(hnr, "Get mean", 0, 0), 2)
    except Exception as e:  # noqa: BLE001
        out["voice_quality_error"] = repr(e)[:100]
    # energy per 20 ms frame, 10 ms hop
    fl, hop = int(0.02 * sr), int(0.01 * sr)
    rms = np.array([np.sqrt(np.mean(y[i:i + fl] ** 2) + 1e-12) for i in range(0, max(1, len(y) - fl), hop)])
    db = 20 * np.log10(rms + 1e-9)
    thr = db.max() - 35
    sp = db > thr
    out["energy_sd_db"] = round(float(db[sp].std()), 2) if sp.any() else None
    out["energy_range_db_p10_p90"] = round(float(np.percentile(db[sp], 90) - np.percentile(db[sp], 10)), 2) if sp.any() else None
    # interior pauses >= 120 ms
    idx = np.where(sp)[0]
    pauses = []
    if len(idx):
        run = 0
        for f in range(idx[0], idx[-1] + 1):
            if not sp[f]:
                run += 1
            else:
                if run * 0.01 >= 0.12:
                    pauses.append(run * 0.01)
                run = 0
    out["n_pauses"] = len(pauses)
    out["pause_mean_s"] = round(float(np.mean(pauses)), 3) if pauses else 0.0
    out["pause_max_s"] = round(float(max(pauses)), 3) if pauses else 0.0
    out["pause_total_s"] = round(float(sum(pauses)), 3)
    out["wpm"] = round(n_words / (dur / 60.0), 1)
    speak_s = dur - sum(pauses)
    out["artic_wpm_excl_pauses"] = round(n_words / (speak_s / 60.0), 1) if speak_s > 0 else None
    out["clip_frac_ge_0995"] = round(float((np.abs(y) >= 0.995).mean()), 6)
    out["peak_dbfs"] = round(float(20 * np.log10(np.abs(y).max() + 1e-12)), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = json.load(open(os.path.join(a.synth, "results.json")))
    rows = []
    for r in res.get("takes", []):
        stem = "%s.t%d" % (r["key"], r["take"])
        wav = os.path.join(a.synth, stem + ".wav")
        n_words = round(r["wpm_native"] * r["dur_s"] / 60.0) if r.get("wpm_native") else 0
        m = measure(wav, n_words)
        m.update({"key": r["key"], "take": r["take"], "voice": r["voice"], "rate": r["rate"], "pitch": r["pitch"], "n_words": n_words,
                  "gate_pass": r.get("gate_pass"), "large_match": r.get("large_match"), "medium_match": r.get("medium_match"),
                  "min_prob": r.get("min_prob"), "n_below_070": r.get("n_below_070")})
        rows.append(m); print(json.dumps(m), flush=True)
    failed = [{k: f.get(k) for k in ("key", "take", "voice", "rate", "pitch", "error")} for f in res.get("failed", [])]
    json.dump({"rows": rows, "failed": failed}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
