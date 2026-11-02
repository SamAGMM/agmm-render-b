#!/usr/bin/env python3
"""Audio forensics for one file. Usage: critique_audio.py <name> <full.wav (44.1k stereo)> <vocals.wav> <novocals.wav> <out.json>
Everything runs on GitHub Actions. Levels are BS.1770 K-weighted (48 kHz coefficients), 100 ms blocks and 400 ms momentary at a 100 ms hop.
"""
import json, sys, warnings
import numpy as np, soundfile as sf
from scipy import signal
warnings.filterwarnings("ignore")

name, fullp, vocp, novp, outp = sys.argv[1:6]
SR = 48000
HB = 0.1  # block s


def load48(p):
    y, sr = sf.read(p, dtype="float32", always_2d=True)
    if sr != SR:
        y = signal.resample_poly(y, SR, sr, axis=0).astype("float32")
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y[:, :2]


def kw(y):
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    out = np.zeros_like(y)
    for c in range(y.shape[1]):
        out[:, c] = signal.lfilter(b2, a2, signal.lfilter(b1, a1, y[:, c]))
    return out


def blocks(y):
    """mean-square per 100 ms block (sum over channels) after K-weighting."""
    z = kw(y)
    n = int(SR * HB)
    nb = len(z) // n
    ms = (z[: nb * n] ** 2).reshape(nb, n, z.shape[1]).mean(axis=1).sum(axis=1)
    return ms


def lufs(ms):
    return -0.691 + 10 * np.log10(np.maximum(ms, 1e-12))


def momentary(ms):
    out = np.zeros(len(ms))
    for i in range(len(ms)):
        out[i] = ms[max(0, i - 3): i + 1].mean()
    return out


full = load48(fullp)
voc = load48(vocp)
nov = load48(novp)
dur = len(full) / SR

ms_full, ms_voc, ms_nov = blocks(full), blocks(voc), blocks(nov)
L = lambda ms: np.round(np.maximum(lufs(ms), -90), 1).tolist()
LM = lambda ms: np.round(np.maximum(lufs(momentary(ms)), -90), 1).tolist()

# integrated loudness (gated) of the full mix via simple BS.1770 gating
def integrated(ms):
    m4 = momentary(ms)
    l4 = lufs(m4)
    keep = l4 > -70
    if not keep.any():
        return -70.0
    g1 = lufs(m4[keep].mean())
    keep2 = l4 > (g1 - 10)
    return float(lufs(m4[keep2].mean())) if keep2.any() else float(g1)

res = {
    "name": name, "duration": round(dur, 3), "hop": HB,
    "integrated_lufs": round(integrated(ms_full), 2),
    "peak_dbfs": round(float(20 * np.log10(max(np.abs(full).max(), 1e-9))), 2),
    "mix_b": L(ms_full), "voc_b": L(ms_voc), "nov_b": L(ms_nov),
    "mix_m": LM(ms_full), "voc_m": LM(ms_voc), "nov_m": LM(ms_nov),
}

# ---------- voice: words, pitch ----------
mono16 = signal.resample_poly(voc.mean(axis=1), 1, 3).astype("float32")  # 16 k
words = []
try:
    from faster_whisper import WhisperModel
    m = WhisperModel("small", device="cpu", compute_type="int8")
    segs, info = m.transcribe(mono16, language="en", word_timestamps=True, vad_filter=False,
                              condition_on_previous_text=False, beam_size=5)
    for s in segs:
        for w in (s.words or []):
            words.append({"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3), "p": round(w.probability, 3)})
except Exception as e:  # noqa
    res["whisper_error"] = repr(e)

f0 = None
try:
    import librosa
    f0, vf, vp = librosa.pyin(mono16, fmin=65, fmax=450, sr=16000, frame_length=1024, hop_length=160)
    # per-100 ms median f0
    n100 = int(dur / HB)
    f0b = []
    for i in range(n100):
        seg = f0[i * 10:(i + 1) * 10]
        seg = seg[~np.isnan(seg)]
        f0b.append(round(float(np.median(seg)), 1) if len(seg) >= 3 else None)
    res["f0_b"] = f0b
    ok = f0[~np.isnan(f0)]
    if len(ok):
        med = float(np.median(ok))
        res["f0_median_hz"] = round(med, 1)
        res["f0_p5_p95_semitones"] = [round(12 * np.log2(np.percentile(ok, 5) / med), 2), round(12 * np.log2(np.percentile(ok, 95) / med), 2)]
        res["f0_range_semitones_p5_p95"] = round(12 * np.log2(np.percentile(ok, 95) / np.percentile(ok, 5)), 2)
except Exception as e:  # noqa
    res["pyin_error"] = repr(e)

# per word: rms dB of vocals, f0 median / max in semitones vs speaker median
vm = voc.mean(axis=1)
for w in words:
    a, b = int(w["s"] * SR), max(int(w["e"] * SR), int(w["s"] * SR) + 480)
    seg = vm[a:b]
    w["db"] = round(float(20 * np.log10(max(np.sqrt((seg ** 2).mean()), 1e-9))), 1) if len(seg) else None
    if f0 is not None and "f0_median_hz" in res:
        fs = f0[int(w["s"] * 100): max(int(w["e"] * 100), int(w["s"] * 100) + 1)]
        fs = fs[~np.isnan(fs)]
        if len(fs):
            w["st_med"] = round(12 * np.log2(np.median(fs) / res["f0_median_hz"]), 2)
            w["st_max"] = round(12 * np.log2(np.max(fs) / res["f0_median_hz"]), 2)
res["words"] = words

# ---------- non-voice events ----------
nvm = nov.mean(axis=1)
hop = int(0.05 * SR)
ne = len(nvm) // hop
e = np.array([20 * np.log10(max(np.sqrt((nvm[i * hop:(i + 1) * hop] ** 2).mean()), 1e-7)) for i in range(ne)])
ve = np.array([20 * np.log10(max(np.sqrt((vm[i * hop:(i + 1) * hop] ** 2).mean()), 1e-7)) for i in range(ne)])
events = []
last = -1.0
for i in range(4, ne):
    rise = e[i] - e[i - 2]
    bed = np.median(e[max(0, i - 22): i - 2]) if i > 6 else e[0]
    if rise >= 8 and e[i] - bed >= 9:
        t = i * 0.05
        if t - last < 0.15:
            continue
        # vocal onset coincidence (leakage)
        vr = ve[i] - ve[max(0, i - 3)]
        if vr >= 8 and ve[i] > -45:
            continue
        j0 = int(t / HB)
        pk = max(L(ms_nov[j0: j0 + 3])) if j0 + 1 < len(ms_nov) else -90
        events.append({"t": round(t, 2), "over_bed_db": round(float(e[i] - bed), 1), "nov_peak_lufs": pk})
        last = t
res["events"] = events

# CLAP labels
try:
    import torch
    from transformers import ClapModel, ClapProcessor
    labs = ["a whoosh sound effect", "a rising riser building tension", "a deep cinematic impact boom", "a sub bass drop",
            "a short orchestral stinger hit", "a cartoon boing", "a notification ping blip", "a camera shutter click",
            "paper rustling", "a door slam", "a crowd murmuring", "a digital glitch error sound", "keyboard typing",
            "a clock ticking", "a heartbeat", "a cash register ka-ching", "a record scratch", "a drum beat",
            "a synth pluck note", "a cymbal crash", "a reverse cymbal swell", "a sparkle chime", "a bell ding",
            "footsteps", "laughter", "applause", "hiss and room tone", "a low drone pad", "a whistle", "glass breaking"]
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    mdl = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    y48 = nvm
    y48k = y48
    # CLAP wants 48 k
    with torch.no_grad():
        ti = proc(text=labs, return_tensors="pt", padding=True)
        te = mdl.get_text_features(**ti)
        te = te / te.norm(dim=-1, keepdim=True)
        for ev in events:
            a = int(max(ev["t"] - 0.1, 0) * SR)
            seg = y48k[a: a + SR]
            ai = proc(audios=seg.astype("float32"), sampling_rate=SR, return_tensors="pt")
            ae = mdl.get_audio_features(**ai)
            ae = ae / ae.norm(dim=-1, keepdim=True)
            pr = (100 * ae @ te.T).softmax(dim=-1)[0]
            top = torch.topk(pr, 2)
            ev["label"] = [[labs[int(k)], round(float(v), 2)] for v, k in zip(top.values, top.indices)]
except Exception as e2:  # noqa
    res["clap_error"] = repr(e2)

# ---------- music character on the non-voice stem ----------
try:
    import librosa
    y22 = signal.resample_poly(nvm, 11025, SR).astype("float32")
    sr22 = 22050 // 2
    tempo, beats = librosa.beat.beat_track(y=y22, sr=sr22)
    res["tempo_bpm"] = round(float(np.atleast_1d(tempo)[0]), 1)
    res["beat_count"] = int(len(beats))
    cent = librosa.feature.spectral_centroid(y=y22, sr=sr22, hop_length=sr22 // 10)[0]
    flat = librosa.feature.spectral_flatness(y=y22, hop_length=sr22 // 10)[0]
    S = np.abs(librosa.stft(y22, n_fft=2048, hop_length=sr22 // 10)) ** 2
    freqs = librosa.fft_frequencies(sr=sr22, n_fft=2048)
    low = S[freqs < 150].sum(axis=0) / np.maximum(S.sum(axis=0), 1e-12)
    res["nov_centroid_hz_b"] = np.round(cent, 0).tolist()
    res["nov_flatness_b"] = np.round(flat, 4).tolist()
    res["nov_lowband_share_b"] = np.round(low, 3).tolist()
    chroma = librosa.feature.chroma_stft(y=y22, sr=sr22).mean(axis=1)
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    mnr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best = (-2, "")
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for k in range(12):
        for nm, prof in (("maj", maj), ("min", mnr)):
            c = np.corrcoef(np.roll(prof, k), chroma)[0, 1]
            if c > best[0]:
                best = (c, notes[k] + " " + nm)
    res["key_guess"] = best[1]
    res["key_corr"] = round(float(best[0]), 2)
except Exception as e3:  # noqa
    res["music_error"] = repr(e3)

json.dump(res, open(outp, "w"))
print(name, "dur", res["duration"], "int", res["integrated_lufs"], "words", len(words), "events", len(events), res.get("tempo_bpm"), res.get("key_guess"))
