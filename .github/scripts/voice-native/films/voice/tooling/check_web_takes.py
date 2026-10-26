#!/usr/bin/env python3
"""check_web_takes.py <piece_dir> [P05,P12,...]   Second ASR pass over web-route takes (after ingest_web_takes.py): three local
transcriptions per take (local_asr.transcribe_variants), PASS if any variant matches the script after folding. Writes asr_variants,
asr_match, asr_pass_variant and (on FAIL) asr_fail_diffs into PNN.receipt.json, and PNN.words.json from the passing variant
(small.en preferred). Prints a table; exit 1 if any take FAILs."""
import difflib, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_asr import fold_compare, fold, transcribe_variants
from asr_compare import _canon

def main():
    piece = Path(sys.argv[1]).resolve()
    only = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else None
    names = [l.strip() for l in (piece / 'names.txt').read_text().splitlines() if l.strip()]
    prompt = 'Names in this script: ' + ', '.join(names) + '. Spoken: agmm dot co dot uk.'
    from faster_whisper import WhisperModel
    models = {'small.en': WhisperModel('small.en', compute_type='int8'), 'base.en': WhisperModel('base.en', compute_type='int8')}
    fails = []
    for rc in sorted(piece.glob('P??.receipt.json')):
        stem = rc.name[:3]
        if only and stem not in only: continue
        r = json.loads(rc.read_text())
        var = transcribe_variants(piece / f'{stem}.wav', prompt, models)
        res = {k: fold_compare(r['text'], v['text']) for k, v in var.items()}
        passing = [k for k, (m, _) in res.items() if m]
        r['asr_variants'] = {k: v['text'] for k, v in var.items()}
        r['asr_fold_compare'] = {k: d.get('reason') for k, (m, d) in res.items()}
        r['asr_match'] = bool(passing)
        r['asr_pass_variant'] = passing[0] if passing else None
        r['asr_engine'] = 'faster-whisper small.en / small.en+names-prompt / base.en (local), folded by local_asr.FOLDS'
        use = 'small.en' if 'small.en' in passing else (passing[0] if passing else 'small.en')
        r['asr_text'] = var[use]['text']
        if not passing:
            best = min(var, key=lambda k: sum(1 for t in difflib.ndiff(_canon(fold(r['text'])).__repr__().split(), _canon(fold(var[k]['text'])).__repr__().split()) if t[0] in '+-'))
            a = [str(t) for t in _canon(fold(r['text']))]; b = [str(t) for t in _canon(fold(var[best]['text']))]
            sm = difflib.SequenceMatcher(a=a, b=b)
            r['asr_fail_diffs'] = [{'variant': best, 'op': op, 'script': ' '.join(a[i1:i2]), 'heard': ' '.join(b[j1:j2])} for op, i1, i2, j1, j2 in sm.get_opcodes() if op != 'equal']
            fails.append(stem)
        else:
            r.pop('asr_fail_diffs', None)
        (piece / f'{stem}.words.json').write_text(json.dumps({'text': var[use]['text'], 'words': var[use]['words']}, indent=2, ensure_ascii=False) + '\n')
        rc.write_text(json.dumps(r, indent=2) + '\n')
        print(stem, 'PASS ' + str(passing) if passing else 'FAIL ' + json.dumps(r['asr_fail_diffs']), flush=True)
    print('FAILS:', fails)
    sys.exit(1 if fails else 0)

if __name__ == '__main__':
    main()
