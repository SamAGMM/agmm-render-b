#!/usr/bin/env python3
"""audioscan.py <mix.wav> <mix.json> <sfx_dir> <out_dir>: where is the screech?
Per 100 ms block of the mix: level in 4 bands and the tonality of the 2-12 kHz region; per effect file (synthesised on the runner): high-band share; per mix clip: its estimated high-band level in the mix."""
import json, os, sys
import numpy as np
from scipy.io import wavfile

def load(p):
    sr, x = wavfile.read(p)
    if x.dtype == np.int16: x = x / 32768.0
    elif x.dtype == np.int32: x = x / 2147483648.0
    else: x = x.astype(np.float64)
    if x.ndim > 1: x = x.mean(1)
    return sr, np.asarray(x, dtype=np.float64)

BANDS = {"lo": (20, 1000), "mid": (1000, 4000), "hi": (4000, 8000), "air": (8000, 20000)}
def db(v): return 20 * np.log10(max(v, 1e-9))

def spec(seg, sr, nfft=8192):
    n = min(len(seg), nfft)
    w = np.hanning(n); X = np.fft.rfft(seg[:n] * w, nfft); P = np.abs(X) ** 2
    f = np.fft.rfftfreq(nfft, 1 / sr); norm = nfft * np.sum(w ** 2)
    out = {}
    for k, (a, b) in BANDS.items():
        m = (f >= a) & (f < b); out[k] = db(np.sqrt(2 * P[m].sum() / norm)) if m.any() else -180
    out["total"] = db(np.sqrt(np.mean(seg ** 2)) + 1e-12)
    m = (f >= 2000) & (f < 12000); pp = P[m]; i = int(np.argmax(pp))
    out["peak_hz"] = float(f[m][i]); out["tonal_db"] = float(10 * np.log10((pp[i] + 1e-12) / (np.median(pp) + 1e-12)))
    return out

def main(mix, mixjson, sfxdir, out):
    os.makedirs(out, exist_ok=True)
    sr, x = load(mix); B = int(sr * 0.1); rows = []
    for i in range(0, len(x) - B + 1, B):
        s = spec(x[i:i + B], sr, 8192); s["t"] = round(i / sr, 2); rows.append(s)
    json.dump(rows, open(os.path.join(out, "blocks.json"), "w"))
    L = []
    L.append("MIX %s: %.2f s, sr %d. Levels dBFS per 100 ms block. hi=4-8 kHz air=8-20 kHz tonal=peak over median in 2-12 kHz (dB) at peak_hz" % (mix, len(x) / sr, sr))
    L.append("median: total %.1f lo %.1f mid %.1f hi %.1f air %.1f" % tuple(np.median([r[k] for r in rows]) for k in ("total", "lo", "mid", "hi", "air")))
    L.append("-- top 40 blocks by hi+air energy (loudest first) --")
    for r in sorted(rows, key=lambda r: -max(r["hi"], r["air"]))[:40]:
        L.append("t %6.2f  total %6.1f  lo %6.1f mid %6.1f hi %6.1f air %6.1f  tonal %5.1f dB at %6.0f Hz" % (r["t"], r["total"], r["lo"], r["mid"], r["hi"], r["air"], r["tonal_db"], r["peak_hz"]))
    L.append("-- blocks with a tonal whine (tonal >= 25 dB and hi >= -52 dBFS) --")
    wh = [r for r in rows if r["tonal_db"] >= 25 and r["hi"] >= -52]
    L.append("%d blocks: %s" % (len(wh), ", ".join("%.1f@%.0fHz" % (r["t"], r["peak_hz"]) for r in wh[:80])))
    # per synthesised effect
    L.append("-- effect files: whole-file levels (dBFS) --")
    files = {}
    for fn in sorted(os.listdir(sfxdir)):
        if not fn.endswith(".wav"): continue
        s2, y = load(os.path.join(sfxdir, fn)); files[fn[:-4]] = (s2, y)
        sp = spec(y, s2, 1 << int(np.ceil(np.log2(max(len(y), 8192)))))
        hi = sp["hi"]; air = sp["air"]
        L.append("%-22s dur %5.2f  total %6.1f lo %6.1f mid %6.1f hi %6.1f air %6.1f  hi-total %6.1f  tonal %5.1f at %6.0f Hz" % (fn[:-4], len(y) / s2, sp["total"], sp["lo"], sp["mid"], hi, air, hi - sp["total"], sp["tonal_db"], sp["peak_hz"]))
    # per mix clip: estimated hi level in the mix
    mj = json.load(open(mixjson)); res = []
    for c in mj["clips"]:
        nm = os.path.basename(str(c.get("src"))).rsplit(".", 1)[0]
        if nm not in files: continue
        s2, y = files[nm]; a = int(c.get("in", 0) * s2); b = int(c.get("out", len(y) / s2) * s2); seg = y[a:b]
        if len(seg) < 256: continue
        sp = spec(seg, s2, 1 << int(np.ceil(np.log2(max(len(seg), 8192)))))
        g = c.get("gain_db", 0) or 0
        res.append((max(sp["hi"], sp["air"]) + g, nm, c.get("at"), c.get("out", 0) - c.get("in", 0), g, sp))
    L.append("-- mix clips ranked by estimated high-band level in the mix (file hi/air + gain) --")
    for v, nm, at, d, g, sp in sorted(res, key=lambda r: -r[0])[:30]:
        L.append("%-22s at %7.2f dur %5.2f gain %6.1f -> hi/air in mix %6.1f dBFS (file hi %6.1f air %6.1f tonal %5.1f at %5.0f Hz)" % (nm, at or 0, d, g, v, sp["hi"], sp["air"], sp["tonal_db"], sp["peak_hz"]))
    open(os.path.join(out, "audioscan.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    main(*sys.argv[1:5])
