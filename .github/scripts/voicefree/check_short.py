#!/usr/bin/env python3
"""check_short.py - the native voice check on Ubuntu for bake-off variants, without the conductor admission (voice-free 20 Oct 2026).

For each variant under <private>/voice-free/check/<tag>/<variant>/ (variant.json + files/<runtime path>): stage the frozen native tree
(.github/scripts/voice-native) at the same absolute path the hosted route uses, drop the variant's script text and takes in, then run exactly what
voice_hosted.run() runs for mode "from-existing": run_short.py --from-existing (assemble 165-172 wpm + local whisper post checks) and
voice_lead_check.py, under the same sitecustomize guard (no provider network). Then delivery_critic.py --layers signal on the assembled take.
No Mistral, no Gemini: AGMM_VOICE_MISTRAL_KEY is not set. Output: results.json (numbers) + the voicelead / RESULT / critic JSON per variant.
The real conductor route (agmm-short-voice.yml via voice_sealer) is separate and untouched.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
NATIVE = SCRIPTS / "voice-native"
BASE = Path("/Users/samwall/alfred/builds/video-program-2026-09-23/v2")


def run(cmd, env, log):
    t = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    Path(log).write_text(r.stdout + r.stderr)
    return r.returncode, round(time.time() - t, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.private) / "voice-free" / "check" / a.tag
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = []
    for vdir in sorted(p for p in root.iterdir() if (p / "variant.json").is_file()):
        meta = json.loads((vdir / "variant.json").read_text())
        sid = meta["id"]; name = vdir.name
        if BASE.exists():
            shutil.rmtree(BASE)
        home_alfred = Path.home() / "alfred"
        if home_alfred.is_symlink() or home_alfred.exists():
            home_alfred.unlink()
        shutil.copytree(NATIVE, BASE)
        for f in sorted((vdir / "files").rglob("*")):
            if f.is_file():
                dest = BASE / f.relative_to(vdir / "files")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dest)
        home_alfred.symlink_to(BASE.parents[2], target_is_directory=True)
        for rel in ("yt-engine/.venv/bin/python", "venvs/pitch/bin/python"):
            p = home_alfred / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.symlink_to(sys.executable)
        tooling = BASE / "shorts/voice/tooling"
        shutil.copyfile(HERE / "extra" / "delivery_critic.py", tooling / "delivery_critic.py")
        shutil.copyfile(HERE / "extra" / "_dc_praat.py", tooling / "_dc_praat.py")
        effects = out / (name + ".effects"); effects.mkdir(exist_ok=True)
        piece = BASE / "shorts/voice" / sid
        env = os.environ.copy()
        env.update(AGMM_VOICE_CHILD="1", AGMM_VOICE_EFFECTS=str(effects), AGMM_VOICE_PIECE=str(piece), FLEET_FREE_ONLY="1", HF_HUB_OFFLINE="1",
                   PYTHONPATH=str(SCRIPTS / "voice-site") + os.pathsep + str(SCRIPTS))
        for k in ("GH_TOKEN", "GITHUB_TOKEN", "SSH_AUTH_SOCK", "AGMM_VOICE_MISTRAL_KEY", "AGMM_VOICE_GEMINI_FREE_KEY"):
            env.pop(k, None)
        row = {"variant": name, "id": sid, "engine": meta.get("engine")}
        rc1, s1 = run([sys.executable, str(BASE / "shorts/voice/tooling/run_short.py"), str(BASE / "shorts/voice"), sid, "--from-existing"], env, out / (name + ".run_short.log"))
        row["run_short_rc"], row["run_short_s"] = rc1, s1
        rc2, s2 = (None, None)
        if rc1 == 0:
            rc2, s2 = run([sys.executable, str(BASE / "shorts/voice/tooling/voice_lead_check.py"), str(BASE / "shorts/voice"), sid], env, out / (name + ".voicelead.log"))
        row["voice_lead_check_rc"], row["voice_lead_check_s"] = rc2, s2
        lead = piece / (sid + ".voicelead.json")
        res = piece / "RESULT.json"
        if lead.is_file():
            L = json.loads(lead.read_text())
            row.update(verdict=L.get("verdict"), fails=L.get("fails"), n_warns=len(L.get("warns") or []), warns=L.get("warns"),
                       pace_wpm_speaking=L.get("pace_wpm_speaking"), pace_wpm_span=L.get("pace_wpm_span"))
            shutil.copyfile(lead, out / (name + ".voicelead.json"))
        if res.is_file():
            R = json.loads(res.read_text())
            row.update(duration_s=R.get("duration_s"), lufs=R.get("lufs_final"), result_status=R.get("status"), hard_fail=R.get("hard_fail"), local_flags=R.get("flags"))
            shutil.copyfile(res, out / (name + ".RESULT.json"))
        if (piece / (sid + ".wav")).is_file():
            rc3, s3 = run([sys.executable, str(tooling / "delivery_critic.py"), "%s::%s" % (piece, sid), "--layers", "signal", "--out", str(out / (name + ".critic.json"))],
                          env, out / (name + ".critic.log"))
            row["critic_rc"], row["critic_s"] = rc3, s3
            cj = out / (name + ".critic.json")
            if cj.is_file():
                C = json.loads(cj.read_text())
                sig = C.get("signal") or {}
                row["critic_verdict"], row["critic_score"] = sig.get("verdict"), sig.get("score")
                fl = sig.get("flags") or []
                row["critic_flags"] = ["%s(%s)" % (f.get("id"), f.get("sev")) for f in fl]
                row["critic_n_scored"] = sum(1 for f in fl if f.get("sev") in ("warn", "fail"))
        for f in ("hosted-native.log",):
            pass
        for src in (piece / "run.log",):
            if src.is_file():
                shutil.copyfile(src, out / (name + ".run.log"))
        results.append(row)
        print(json.dumps({k: row.get(k) for k in ("variant", "verdict", "pace_wpm_speaking", "pace_wpm_span", "duration_s", "critic_verdict", "run_short_rc", "voice_lead_check_rc")}), flush=True)
    json.dump(results, open(out / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
