#!/usr/bin/env python3
"""assemble_pack.py SPEC.json OUTDIR WORKDIR : build ONE voicefree pack from chosen takes of several packs.
SPEC = [{"run": 123, "label": "H02-dl-k", "key": "P1", "take": 3, "as": 1}, ...]  (packs are downloaded from this repo and decrypted into WORKDIR/<run>)
Output: <key>.t<as>.wav / .words.json / .receipt.json + results.json, in the shape build_film.py's load_takes() and the par workflow read."""
import json, os, shutil, subprocess, sys
spec = json.load(open(sys.argv[1])); out, work = sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
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
    rc = json.load(open(os.path.join(out, t + ".receipt.json"))); rc["attempt"] = a; rc["assembled_from"] = {"run": it["run"], "label": it["label"], "take": it["take"]}
    json.dump(rc, open(os.path.join(out, t + ".receipt.json"), "w"), indent=2, ensure_ascii=False)
    res = json.load(open(os.path.join(d, "results.json")))
    row = next(r for r in res["takes"] if r["key"] == it["key"] and r["take"] == it["take"]); row = dict(row, take=a); rows.append(row)
    sel.append((it["key"], a))
json.dump({"label": "assembled", "engine": "chatterbox", "takes": rows, "assembled": spec}, open(os.path.join(out, "results.json"), "w"), indent=1)
print("assembled", len(rows), "takes")
