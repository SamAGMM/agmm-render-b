#!/usr/bin/env python3
"""Praat contour helper for delivery_critic.py (19 Oct 2026, AUDIO-CRITIC). Run with ~/alfred/venvs/pitch/bin/python (the only venv with
parselmouth; the yt-engine venv must not be changed, same arrangement as voice_pitch.py).

  _dc_praat.py <wav>   ->   JSON on stdout: {"hop": 0.01, "t0": <first frame time>, "f0_st": [...], "int_db": [...]}

f0_st   pitch in semitones re 100 Hz per 10 ms frame, null where unvoiced; same autocorrelation tracker and octave-jump removal as
        voice_pitch.py (70-330 Hz, +-5 st from the local median is an octave error and is dropped)
int_db  Praat intensity (dB, 100 Hz floor so pitch-period ripple is smoothed) per 10 ms frame
"""
import json, sys
import numpy as np
import parselmouth


def main():
    snd = parselmouth.Sound(sys.argv[1])
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    p = snd.to_pitch_ac(time_step=0.01, pitch_floor=70, pitch_ceiling=330, very_accurate=True, silence_threshold=0.03, voicing_threshold=0.5)
    f = np.where(p.selected_array["frequency"] > 0, p.selected_array["frequency"], np.nan)
    v = 12 * np.log2(f / 100.0)
    idx = np.where(np.isfinite(v))[0]
    v2 = v.copy()
    for i in idx:
        w = v[max(0, i - 5): i + 6]
        w = w[np.isfinite(w)]
        if len(w) >= 4 and abs(v[i] - np.median(w)) > 5:
            v2[i] = np.nan
    it = snd.to_intensity(minimum_pitch=100, time_step=0.01)
    iv = it.values[0]
    out = {"hop": 0.01, "t0": float(p.xs()[0]), "i_t0": float(it.xs()[0]),
           "f0_st": [None if not np.isfinite(x) else round(float(x), 2) for x in v2],
           "int_db": [round(float(x), 1) if np.isfinite(x) else None for x in iv]}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
