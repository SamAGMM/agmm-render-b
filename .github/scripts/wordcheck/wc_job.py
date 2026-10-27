#!/usr/bin/env python3
"""wc_job.py (Actions): word gate over one or more voice packs and/or a built film's mix. Env: FILM, SCRIPT, PACKS (json [{run,label,pick?,keys?,takes?}]), MIX (json {run,label} or empty),
MODEL, OUT, GH_TOKEN, VOICEFREE_PASS, GITHUB_REPOSITORY. Public-log hygiene: prints only ids and PASS/FAIL, never script text."""
import argparse, glob, json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import word_check as W

film, script, model, out = os.environ["FILM"], os.environ["SCRIPT"], os.environ.get("MODEL") or "large-v3", os.environ["OUT"]
repo = os.environ["GITHUB_REPOSITORY"]
os.makedirs(out, exist_ok=True)


def sh(cmd):
    subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=True)


def dl(run, name, dest):
    os.makedirs(dest, exist_ok=True)
    sh("gh run download %s -R %s -n %s -D %s" % (run, repo, name, dest))


for pk in json.loads(os.environ.get("PACKS") or "[]"):
    run, label = str(pk["run"]), pk["label"]
    tmp = tempfile.mkdtemp(prefix="wcpack_")
    try:
        dl(run, "voicefree-chatterbox-%s-ENC" % label, tmp + "/enc")
        os.makedirs(tmp + "/pack")
        sh("openssl enc -d -aes-256-cbc -pbkdf2 -pass env:VOICEFREE_PASS -in %s/enc/synth-pack.enc | tar -x -C %s/pack" % (tmp, tmp))
    except subprocess.CalledProcessError:
        print("pack %s %s: download or decrypt failed (expired or other pool)" % (run, label), flush=True); continue
    a = argparse.Namespace(script=script, film=film, pack=tmp + "/pack", mix=None, stem=None, tag="%s:%s" % (run, label), pick=pk.get("pick"), keys=pk.get("keys"),
                           takes=pk.get("takes"), model=model, out="%s/words-%s.json" % (out, run))
    W.cmd_run(a)

mx = os.environ.get("MIX")
if mx:
    m = json.loads(mx)
    tmp = tempfile.mkdtemp(prefix="wcmix_")
    dl(m["run"], "hailmary-%s" % m["label"], tmp)
    mixwav = glob.glob(tmp + "/**/gate-input/mix.wav", recursive=True)
    stem = glob.glob(tmp + "/**/gate-input/stems/voice.wav", recursive=True)
    if not mixwav:
        post = glob.glob(tmp + "/**/POST1080.mp4", recursive=True) or glob.glob(tmp + "/**/FINAL.mp4", recursive=True)
        mixwav = post
    a = argparse.Namespace(script=script, film=film, pack=None, mix=mixwav[0], stem=stem[0] if stem else None, tag="mix", pick=None, keys=None, takes=None, model=model,
                           out="%s/words-mix-%s.json" % (out, m["run"]))
    W.cmd_run(a)
print("job done")
