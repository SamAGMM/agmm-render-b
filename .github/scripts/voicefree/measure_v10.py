# measure_v10.py <synth-out dir> <pub dir> : numbers only. UTMOS22 strong (SpeechMOS), Praat F0 range/SD in semitones, wpm from results.json.
import glob, json, os, sys
import numpy as np, soundfile as sf, torch, torchaudio, parselmouth
d, pub = sys.argv[1], sys.argv[2]
pred = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
res = {r["key"] + ".t%d" % r["take"]: r for r in json.load(open(os.path.join(d, "results.json")))["takes"]}
out = {}
for f in sorted(glob.glob(os.path.join(d, "*.wav"))):
    stem = os.path.basename(f)[:-4]
    y, sr = sf.read(f, dtype="float32")
    w16 = torchaudio.functional.resample(torch.from_numpy(y).unsqueeze(0), sr, 16000)
    with torch.no_grad(): mos = float(pred(w16, 16000).mean())
    p = parselmouth.Sound(y.astype("float64"), sr).to_pitch_ac(pitch_floor=70, pitch_ceiling=350)
    f0 = p.selected_array["frequency"]; f0 = f0[f0 > 0]
    st = 12 * np.log2(f0 / np.median(f0)) if len(f0) else np.array([0.0])
    r = res.get(stem, {})
    out[stem] = {"utmos": round(mos, 3), "f0_range_st": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2), "f0_sd_st": round(float(np.std(st)), 2),
                 "f0_median_hz": round(float(np.median(f0)), 1) if len(f0) else None, "dur_s": r.get("dur_s"), "wpm": r.get("wpm_native"), "span_wpm": r.get("span_wpm"),
                 "gate_pass": r.get("gate_pass"), "large_match": r.get("large_match"), "medium_match": r.get("medium_match"), "min_prob": r.get("min_prob"), "voice": r.get("voice")}
    print(stem, json.dumps(out[stem]), flush=True)
json.dump(out, open(os.path.join(pub, "measure.json"), "w"), indent=1)
