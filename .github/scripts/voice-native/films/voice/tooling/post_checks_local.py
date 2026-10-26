#!/usr/bin/env python3
"""post_checks_local.py <piece_dir> <name>   (12 Oct 2026) the film voice checks that need no API, run on the assembled <name>.wav:
 1 WHOLE-FILE ASR   local faster-whisper small.en on the whole file, folded script-vs-heard word diff (ratio >= 0.97 and every diff listed)
 2 TIMING           assembled <name>.words.json vs the whole-file transcript's own word times (median |dt| <= 0.15 s, p95 <= 0.6 s)
 3 PACE             film wpm in the 150-165 band (assemble's tempo), per-paragraph native wpm vs the median (flag > +-30%, paragraphs >= 15 words)
 4 GAPS             every film gap (assemble gap_log) inside spec; longest silent stretch INSIDE each take (silencedetect -45 dB) <= 1.2 s
 5 NAMES            each script name occurrence heard (fuzzy) in at least one per-paragraph transcript variant; misses listed with what was heard
 6 LOUDNESS         integrated -18 +-0.5 LUFS, true peak <= -1.5 dBTP
 7 SPOT             spotcheck_words.py (15 random words have signal inside their window)
 8 LISTEN           Gemini critique: UNJUDGED (billing dead); never reported as a pass
Writes <name>.checks_local.json and RESULT.json; exit 1 if a hard check (1, 2, 4 gap_log, 6, 7) fails; soft flags are listed, not hidden."""
import difflib, json, re, statistics, subprocess, sys, wave
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_asr import fold, fold_compare
from asr_compare import _canon

def _j(o): return o.item() if hasattr(o, 'item') else str(o)

def norm_tokens(s): return re.findall(r"[a-z0-9']+", fold(s))

def main():
    piece = Path(sys.argv[1]).resolve(); name = sys.argv[2]
    band = [150.0, 165.0]
    if '--band' in sys.argv:   # 13 Oct 2026: shorts use 165-172 (run_short.py --from-existing passes it); films keep 150-165
        i = sys.argv.index('--band'); band = [float(sys.argv[i + 1]), float(sys.argv[i + 2])]
    wav = piece / f'{name}.wav'; rcpt = json.loads((piece / f'{name}.receipt.json').read_text())
    words = json.loads((piece / f'{name}.words.json').read_text())['words']
    script = rcpt['full_script_text']
    out = {'name': name, 'checks': {}, 'flags': []}
    hard_fail = []

    # 1 + 2 whole-file ASR and timing
    from faster_whisper import WhisperModel
    m = WhisperModel('small.en', compute_type='int8')
    segs, _ = m.transcribe(str(wav), word_timestamps=True, language='en', beam_size=5, condition_on_previous_text=False)
    ww = []; seg_texts = []
    for s in segs:
        seg_texts.append(s.text.strip())
        for w in (s.words or []):
            ww.append({'word': w.word.strip(), 'start': w.start, 'end': w.end})
    heard = ' '.join(seg_texts)   # segment text, not re-joined words: '$440,000' must not split into '440' '000'
    (piece / f'{name}.whole_asr.json').write_text(json.dumps({'text': heard, 'words': ww}, indent=1))
    a0 = [str(t) for t in _canon(fold(script))]; b0 = [str(t) for t in _canon(fold(heard))]
    ratio_unfolded = difflib.SequenceMatcher(a=a0, b=b0, autojunk=False).ratio()
    a, b, folds = a0, b0, []
    try:
        # runtime delta 5 (26 Oct 2026): deterministic spelling folds on BOTH sides before the ratio (asr_fold.py: fixed UK/US list, names.txt possessives, aliases.txt declarations, pound amounts)
        from asr_fold import fold_pair
        names_txt = (piece / 'names.txt').read_text(encoding='utf-8') if (piece / 'names.txt').exists() else ''
        aliases_txt = (piece / 'aliases.txt').read_text(encoding='utf-8') if (piece / 'aliases.txt').exists() else ''
        ta, tb, folds = fold_pair(fold(script), fold(heard), names_txt, _canon, aliases_txt)
        a = [str(t) for t in ta]; b = [str(t) for t in tb]
    except ImportError:
        pass
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    diffs = [{'op': op, 'script': ' '.join(a[i1:i2]), 'heard': ' '.join(b[j1:j2]), 'script_pos': i1} for op, i1, i2, j1, j2 in sm.get_opcodes() if op != 'equal']
    ratio = sm.ratio()
    out['checks']['whole_file_asr'] = {'ratio': round(ratio, 4), 'ratio_unfolded': round(ratio_unfolded, 4), 'folds_applied': folds[:60], 'n_diffs': len(diffs), 'diffs': diffs[:60], 'pass': ratio >= 0.97}
    if ratio < 0.97: hard_fail.append('whole_file_asr')
    # timing: match words by aligned tokens between assembled words.json and whole-file whisper words
    ta = [norm_tokens(w['word']) for w in words]; tb = [norm_tokens(w['word']) for w in ww]
    fa = [t[0] if t else '' for t in ta]; fb = [t[0] if t else '' for t in tb]
    sm2 = difflib.SequenceMatcher(a=fa, b=fb, autojunk=False)
    dts = []
    for blk in sm2.get_matching_blocks():
        for k in range(blk.size):
            if fa[blk.a + k]: dts.append(abs(words[blk.a + k]['start'] - ww[blk.b + k]['start']))
    dts.sort()
    med = statistics.median(dts) if dts else 99; p95 = dts[int(0.95 * (len(dts) - 1))] if dts else 99
    out['checks']['timing'] = {'matched_words': len(dts), 'median_abs_dt_s': round(med, 3), 'p95_abs_dt_s': round(p95, 3), 'pass': med <= 0.15 and p95 <= 0.6}
    if not out['checks']['timing']['pass']: hard_fail.append('timing')

    # 3 pace
    wpm_final = rcpt['wpm_speaking_final']
    paras = []
    for rc in sorted(piece.glob('P??.receipt.json')):
        r = json.loads(rc.read_text()); paras.append((rc.name[:3], len(r['text'].split()), r['wpm']))
    med_wpm = statistics.median([w for _, n, w in paras if n >= 15])
    odd = [(p, w) for p, n, w in paras if n >= 15 and abs(w - med_wpm) / med_wpm > 0.30]
    out['checks']['pace'] = {'film_wpm_speaking': wpm_final, 'band': band, 'in_band': band[0] <= wpm_final <= band[1] + 0.05, 'median_native_wpm_15plus': round(med_wpm, 1),
                             'paragraphs_off_median_30pct': odd, 'native_wpm_by_paragraph': {p: w for p, _, w in paras}}
    if not out['checks']['pace']['in_band']: out['flags'].append(f'wpm outside {band[0]:g}-{band[1]:g}')
    if odd: out['flags'].append(f'native pace off median by >30%: {odd}')

    # 4 gaps
    gl = rcpt['gap_log']
    bad_gaps = [g for g in gl if not ((0.6 <= g['gap_s'] <= 0.9) if g['section_break'] else (0.25 <= g['gap_s'] <= 0.45))]
    long_in = {}
    for p, _, _ in paras:
        r = subprocess.run(['ffmpeg', '-nostats', '-i', str(piece / f'{p}.wav'), '-af', 'silencedetect=noise=-45dB:d=0.3', '-f', 'null', '-'], capture_output=True, text=True).stderr
        starts = [float(x) for x in re.findall(r'silence_start: (-?[0-9.]+)', r)]; ends = [(float(x), float(y)) for x, y in re.findall(r'silence_end: ([0-9.]+) \| silence_duration: ([0-9.]+)', r)]
        durs = [d for _, d in ends if _ > 0.3]   # silence that ENDS after the start of the clip = internal or trailing; leading silence ends near 0
        mx = max([d for e, d in ends if e - d > 0.15] or [0])
        if mx > 1.2: long_in[p] = round(mx, 2)
    out['checks']['gaps'] = {'film_gaps': len(gl), 'out_of_spec': bad_gaps, 'internal_silence_over_1.2s': long_in, 'pass': not bad_gaps}
    if bad_gaps: hard_fail.append('gaps')
    if long_in: out['flags'].append(f'internal silence > 1.2 s in takes: {long_in}')

    # 5 names
    names = [l.strip() for l in (piece / 'names.txt').read_text().splitlines() if l.strip()]
    res = {}
    for nm in names:
        toks = norm_tokens(nm); occ = 0; ok = 0; miss = []
        for rc in sorted(piece.glob('P??.receipt.json')):
            r = json.loads(rc.read_text())
            n_in = len(re.findall(r'\b' + re.escape(nm.lower()) + r'\b', r['text'].lower()))
            if not n_in: continue
            occ += n_in
            variants = [r['asr_text']] + list((r.get('asr_variants') or {}).values())
            hit = False
            for v in variants:
                vt = norm_tokens(v)
                for i in range(len(vt) - len(toks) + 1):
                    if all(difflib.SequenceMatcher(a=vt[i + k], b=toks[k]).ratio() >= 0.85 for k in range(len(toks))): hit = True; break
                if hit: break
            if hit: ok += n_in
            else: miss.append({'paragraph': rc.name[:3], 'heard_variants': [v for v in variants][:3]})
        res[nm] = {'occurrences': occ, 'heard_ok': ok, 'misses': [x['paragraph'] for x in miss]}
        out.setdefault('names_detail', {})[nm] = miss
    out['checks']['names'] = res
    nm_miss = {k: v['misses'] for k, v in res.items() if v['misses']}
    if nm_miss: out['flags'].append(f'name occurrences not heard in any transcript variant: {nm_miss}')

    # 6 loudness
    post = rcpt.get('loudnorm_measured_post') or {}
    li = float(post.get('output_i', 0)); tp = float(post.get('output_tp', 0))
    out['checks']['loudness'] = {'integrated_lufs': li, 'true_peak_dbtp': tp, 'pass': abs(li + 18) <= 0.5 and tp <= -1.5}
    if not out['checks']['loudness']['pass']: hard_fail.append('loudness')

    # 7 spot
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'spotcheck_words.py'), str(piece), name, '15'], capture_output=True, text=True)
    (piece / f'{name}.spotcheck.txt').write_text(r.stdout)
    out['checks']['spotcheck_words'] = {'pass': r.returncode == 0, 'summary': r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}
    if r.returncode != 0: hard_fail.append('spotcheck_words')

    out['checks']['listen'] = 'UNJUDGED (Gemini billing dead; no free audio judge run) -- not a pass'
    with wave.open(str(wav)) as f: dur = f.getnframes() / f.getframerate()
    out['duration_s'] = round(dur, 3); out['hard_fail'] = hard_fail
    (piece / f'{name}.checks_local.json').write_text(json.dumps(out, indent=2, default=_j) + '\n')
    result = {'id': name, 'status': 'FINISHED_LOCAL_CHECKS_PASS' if not hard_fail else 'FAILED_LOCAL_CHECKS', 'listening_review': 'UNJUDGED',
              'hard_fail': hard_fail, 'flags': out['flags'], 'duration_s': round(dur, 3), 'wpm_final': wpm_final, 'lufs_final': li,
              'wav_path': str(wav), 'words_json_path': str(piece / f'{name}.words.json'), 'receipt_path': str(piece / f'{name}.receipt.json'),
              'checks_path': str(piece / f'{name}.checks_local.json'), 'n_paragraphs': rcpt['n_paragraphs']}
    (piece / 'RESULT.json').write_text(json.dumps(result, indent=2, default=_j) + '\n')
    print(json.dumps({k: v for k, v in out['checks'].items() if k not in ('pace',)}, indent=1, default=_j)[:3500]); print('FLAGS', out['flags']); print('HARD FAIL', hard_fail)
    sys.exit(1 if hard_fail else 0)

if __name__ == '__main__':
    main()
