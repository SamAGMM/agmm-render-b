#!/usr/bin/env python3
"""fish_gate.py - licence and access gate for the Fish Audio route (voice-free 20 Oct 2026). Stdlib only, no audio, prints numbers and ids.

AGMM may use Fish weights commercially only under the Fish Audio Community License (free while annual revenue < USD 500k, self-hosted, never a hosted
service). A Hugging Face model card states its licence; weights that are CC-BY-NC / Research-only are refused here, and gated weights need an HF token
that belongs to an account that has accepted the licence (only Sam can do that). Exit 0 = allowed, 3 = refused. Writes gate.json for the run summary.
"""
import json, os, sys, urllib.error, urllib.request

repo = sys.argv[1]
out = sys.argv[2]
tok = os.environ.get("HF_TOKEN", "")
req = urllib.request.Request("https://huggingface.co/api/models/" + repo, headers=({"Authorization": "Bearer " + tok} if tok else {}))
try:
    info = json.load(urllib.request.urlopen(req, timeout=60))
except urllib.error.HTTPError as e:
    info = {"error": e.code}
card = (info.get("cardData") or {})
lic = str(card.get("license") or next((t.split(":", 1)[1] for t in info.get("tags", []) if t.startswith("license:")), "unknown")).lower()
gated = info.get("gated")
NC = ("cc-by-nc", "cc-by-nc-sa", "cc-by-nc-nd", "other", "unknown", "research")
reasons = []
if info.get("error"):
    reasons.append("model API answered HTTP %s" % info["error"])
if any(lic.startswith(p) for p in NC) and lic not in ("fish-audio-community", "fish-audio-community-license-1.0"):
    reasons.append("weights licence is %r: not the Fish Audio Community License, so no commercial use" % lic)
if gated and gated != "False" and not tok:
    reasons.append("weights are gated (%s) and no HF_TOKEN is bound: needs Sam's Hugging Face account to accept the licence" % gated)
res = {"repo": repo, "license": lic, "gated": gated, "token_present": bool(tok), "allowed": not reasons, "reasons": reasons}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
sys.exit(0 if not reasons else 3)
