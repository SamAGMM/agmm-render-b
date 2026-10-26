#!/usr/bin/env python3
"""ingest_web_takes.py <piece_dir> <staging_dir> [inpage.json]   (12 Oct 2026, F03 voice via the Voxtral WEB demo)
The no-API twin of gen_paragraphs.py: the takes were generated one chunk at a time in the Space's Customization tab
(Mistral TTS API = 402) and downloaded as <staging_dir>/PNN.wav. For each chunk of paragraphs.txt (same split as
gen_paragraphs.py: blank-line blocks, text sent verbatim) this copies the take to <piece_dir>/PNN.wav, runs LOCAL faster-whisper
small.en (word timestamps; run it through kit/tools/heavy.py), compares with asr_compare (same comparator as the API route) and
writes PNN.words.json (flat 'words' list, the shape assemble.py falls back to) and PNN.receipt.json (same fields as
gen_paragraphs.py + route/page/time evidence). Never calls an API. Re-running overwrites with identical content."""
import datetime as dt, hashlib, json, re, shutil, sys, wave
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_compare import compare

REF_SHA = '3df9d7568c0b0569887e6c9f39bd35835cda305a09d288ea535c63d9cdc15443'

def sha(b): return hashlib.sha256(b).hexdigest()

def main():
    piece = Path(sys.argv[1]).resolve(); stage = Path(sys.argv[2]).resolve()
    inpage = json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else {}
    paras = [p.strip() for p in piece.joinpath('paragraphs.txt').read_text(encoding='utf-8').strip().split('\n\n') if p.strip()]
    from faster_whisper import WhisperModel
    model = WhisperModel('small.en', compute_type='int8')
    bad = []
    import os
    only = set(filter(None, os.environ.get('ONLY', '').split(',')))   # e.g. ONLY=P07,P09 re-ingests just those
    for i, text in enumerate(paras, 1):
        stem = f'P{i:02d}'
        if only and stem not in only: continue
        src = stage / f'{stem}.wav'
        if not src.exists():
            print(stem, 'MISSING'); bad.append(stem); continue
        dst = piece / f'{stem}.wav'
        shutil.copy2(src, dst)
        raw = dst.read_bytes()
        with wave.open(str(dst)) as f:
            info = {'channels': f.getnchannels(), 'rate': f.getframerate(), 'width': f.getsampwidth(),
                    'frames': f.getnframes(), 'duration_s': round(f.getnframes() / f.getframerate(), 6)}
        assert (info['channels'], info['rate'], info['width']) == (1, 24000, 2), info
        segs, _ = model.transcribe(str(dst), word_timestamps=True, language='en', beam_size=5)
        words, texts = [], []
        for s in segs:
            texts.append(s.text.strip())
            for w in (s.words or []):
                words.append({'word': w.word.strip(), 'start': round(w.start, 3), 'end': round(w.end, 3)})
        asr_text = ' '.join(texts)
        match, detail = compare(text, asr_text)
        ev = inpage.get(str(i), {})
        item = {
            'paragraph': stem, 'text': text, 'input_sha256': sha(text.encode()),
            'reference_sha256_only': REF_SHA, 'model': 'voxtral-mini-tts-2603', 'route': 'voxtral-web-demo (Space Customization tab, signed in as AGMMalfred)',
            'page_url': ev.get('page', 'https://mistralai-voxtral-tts-demo.hf.space/?__theme=system'),
            'generated_utc': ev.get('t') or dt.datetime.fromtimestamp(src.stat().st_mtime, dt.timezone.utc).isoformat(),
            'generated_utc_source': 'in-page download time' if ev.get('t') else 'file mtime (download time)',
            'inpage_textbox_sha256': ev.get('box_text_sha256'), 'inpage_file_sha256': ev.get('sha'),
            'inpage_file_matches_disk': (ev.get('sha') == sha(raw)) if ev.get('sha') else None,
            'space_file_url_tail': (ev.get('u') or '')[-90:] or None,
            'wav_sha256': sha(raw), 'wav_bytes': len(raw), 'wav_info': info, 'attempt': ev.get('attempt', 1),
            'asr_status': 'checked', 'asr_engine': 'faster-whisper small.en (local)', 'asr_text': asr_text,
            'asr_match': match, 'asr_compare_detail': detail,
            'wpm': round(len(text.split()) / (info['duration_s'] / 60.0), 1),
        }
        (piece / f'{stem}.words.json').write_text(json.dumps({'text': asr_text, 'words': words}, indent=2, ensure_ascii=False) + '\n')
        (piece / f'{stem}.receipt.json').write_text(json.dumps(item, indent=2) + '\n')
        print(stem, 'dur', info['duration_s'], 'wpm', item['wpm'], 'asr_match', match, '' if match else detail.get('reason'), flush=True)
    print('done', len(paras), 'missing', bad)

if __name__ == '__main__':
    main()
