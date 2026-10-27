#!/usr/bin/env python3
"""assemble_pack.py SPEC.json OUTDIR WORKDIR : build ONE voicefree pack from chosen takes of several packs.
SPEC = [{"run": 123, "label": "H02-dl-k", "key": "P1", "take": 3, "as": 1}, ...]  (packs are downloaded from this repo and decrypted into WORKDIR/<run>)
Output: <key>.t<as>.wav / .words.json / .receipt.json + results.json, in the shape build_film.py's load_takes() and the par workflow read."""
import json, os, shutil, subprocess, sys
spec = json.load(open(sys.argv[1])); out, work = sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)

def post_process(wav, words_json, post):
    """shorten gaps longer than post['max_gap'] to that length, then pitch-preserving atempo post['tempo'] (ffmpeg), then re-time the words with faster-whisper small.en"""
    import numpy as np, soundfile as sf
    x, sr = sf.read(wav, dtype="float32")
    w = json.load(open(words_json))["segments"]
    mg = float(post.get("max_gap", 0.3)); pieces, cur = [], 0.0
    for a, b in zip(w, w[1:]):
        gap = b["start"] - a["end"]
        if gap > mg:
            cut_s = a["end"] + mg / 2; cut_e = b["start"] - mg / 2
            pieces.append(x[int(cur * sr):int(cut_s * sr)]); cur = cut_e
    pieces.append(x[int(cur * sr):])
    y = np.concatenate(pieces) if len(pieces) > 1 else x
    tmp = wav + ".pre.wav"; sf.write(tmp, y, sr, subtype="PCM_16")
    t = float(post.get("tempo", 1.0))
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", tmp, "-af", "atempo=%.4f" % t, "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", wav], check=True)
    os.remove(tmp)
    from faster_whisper import WhisperModel
    m = WhisperModel("small.en", device="cpu", compute_type="int8")
    segs, _ = m.transcribe(wav, language="en", word_timestamps=True, beam_size=5, condition_on_previous_text=False)
    words = [{"type": "transcription_segment", "text": ww.word.strip(), "start": round(float(ww.start), 3), "end": round(float(ww.end), 3)} for sg in segs for ww in (sg.words or []) if ww.word.strip()]
    d = json.load(open(words_json)); d["segments"] = words; d["text"] = " ".join(v["text"] for v in words); d["model"] = "faster-whisper-small.en (post-processed)"
    json.dump(d, open(words_json, "w"), indent=2)
    print("post", os.path.basename(wav), "dur", round(len(y) / sr / t, 2), "words", len(words), d["text"][:80].replace("\n", " "), flush=True)

packs = {}
for it in spec:
    k = (it["run"], it["label"])
    if k in packs: continue
    d = os.path.join(work, str(it["run"])); os.makedirs(d, exist_ok=True)
    enc = os.path.join(work, "enc%s" % it["run"]); os.makedirs(enc, exist_ok=True)
    subprocess.run(["gh", "run", "download", str(it["run"]), "-R", os.environ["GITHUB_REPOSITORY"], "-n", "voicefree-chatterbox-%s-ENC" % it["label"], "-D", enc], check=True)
    subprocess.run(["bash", "-o", "pipefail", "-c", "openssl enc -d -aes-256-cbc -pbkdf2 -pass env:VOICEFREE_PASS -in %s/synth-pack.enc | tar -x -C %s" % (enc, d)], check=True)
    packs[k] = d
rows, sel = [], []
for it in spec:
    d = packs[(it["run"], it["label"])]; s = "%s.t%d" % (it["key"], it["take"]); a = it.get("as", 1); t = "%s.t%d" % (it["key"], a)
    for ext in ("wav", "words.json", "receipt.json"):
        shutil.copy(os.path.join(d, "%s.%s" % (s, ext)), os.path.join(out, "%s.%s" % (t, ext)))
    if it.get("post"):
        post_process(os.path.join(out, t + ".wav"), os.path.join(out, t + ".words.json"), it["post"])
    rc = json.load(open(os.path.join(out, t + ".receipt.json")))
    if it.get("post"): rc["post"] = it["post"]; rc["attempt"] = a; rc["assembled_from"] = {"run": it["run"], "label": it["label"], "take": it["take"]}
    json.dump(rc, open(os.path.join(out, t + ".receipt.json"), "w"), indent=2, ensure_ascii=False)
    res = json.load(open(os.path.join(d, "results.json")))
    row = next(r for r in res["takes"] if r["key"] == it["key"] and r["take"] == it["take"]); row = dict(row, take=a); rows.append(row)
    sel.append((it["key"], a))
json.dump({"label": "assembled", "engine": "chatterbox", "takes": rows, "assembled": spec}, open(os.path.join(out, "results.json"), "w"), indent=1)
print("assembled", len(rows), "takes")
