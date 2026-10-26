#!/usr/bin/env python3
"""Spot-check word timings against the actual waveform: for N random words, verify there is
real signal (RMS above a floor) inside [start,end], and, where the previous word ends well
before this one starts, verify the gap between them is actually quieter (silence gap present,
not a mistimed boundary). Uses ffmpeg astats per-clip RMS. Prints a table; exits non-zero if
any check fails outright (word window is silent)."""
import json, random, subprocess, sys
from pathlib import Path

def rms_db(wav, start, dur):
    if dur <= 0:
        return None
    r = subprocess.run(['ffmpeg', '-nostats', '-i', str(wav), '-ss', f'{max(0,start):.3f}', '-t', f'{dur:.3f}',
                         '-af', 'astats=metadata=1:reset=1', '-f', 'null', '-'],
                        capture_output=True, text=True)
    for line in r.stderr.splitlines():
        if 'RMS level dB' in line:
            try:
                return float(line.strip().split(':')[-1])
            except ValueError:
                pass
    return None

def main():
    piece_dir = Path(sys.argv[1]).resolve()
    name = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    wav = piece_dir / f'{name}.wav'
    words = json.loads((piece_dir / f'{name}.words.json').read_text())['words']
    random.seed(42)
    sample = random.sample(range(len(words)), min(n, len(words)))
    sample.sort()
    fails = 0
    print(f'{"word":<20}{"start":>8}{"end":>8}{"word_rms_db":>14}{"pre_gap_rms_db":>16}{"gap_s":>8}  verdict')
    for i in sample:
        w = words[i]
        dur = w['end'] - w['start']
        word_rms = rms_db(wav, w['start'], dur)
        pre_gap_rms = None
        gap = None
        if i > 0:
            prev = words[i - 1]
            gap = w['start'] - prev['end']
            if gap > 0.15:
                pre_gap_rms = rms_db(wav, prev['end'] + 0.03, gap - 0.06)
        ok = word_rms is not None and word_rms > -50
        if not ok:
            fails += 1
        verdict = 'OK' if ok else 'FAIL (silent at word window)'
        print(f'{w["word"]:<20}{w["start"]:>8.2f}{w["end"]:>8.2f}{(word_rms if word_rms is not None else float("nan")):>14.1f}'
              f'{(pre_gap_rms if pre_gap_rms is not None else float("nan")):>16.1f}{(gap if gap is not None else float("nan")):>8.2f}  {verdict}')
    print(f'\n{len(sample)} words checked, {fails} failed')
    sys.exit(1 if fails else 0)

if __name__ == '__main__':
    main()
