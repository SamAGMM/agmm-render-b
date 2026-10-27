#!/usr/bin/env python3
"""synth_delivery.py: Chatterbox takes with DELIVERY control (27 Oct 2026). Same output files as synth.py (<key>.t<k>.wav / .words.json / .receipt.json / results.json)
so the par workflow can use the pack unchanged, plus a plain numbers-only <stem>.delivery.json per take (no script text) for the public artifact.

Request: like synth.py, but a paragraph may carry
  "n_takes": 5,
  "variants": [ {"over": {"exaggeration":..,"cfg_weight":..,"temperature":..},
                 "frags": [ {"t": "text voiced alone", "pause": 0.2 (silence AFTER it), "over": {...}, "tempo": 1.1 (atempo on this fragment), "gain_db": 2.0} , ...] }, ... ]
Take k uses variants[(k-1) % len]. No target_wpm slowdown/speed-up is applied: the pace is what the variant makes it.
"""
import argparse, json, os, re, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "delivery"))
import vf_common as C
import synth as S
import delivery_measure as DM

BRAND_AGM = re.compile(r"(?<![a-z0-9])a[\s.\-,]*g[\s.\-,]*m(?![a-z0-9])(?![\s.\-,]*m(?![a-z0-9]))", re.I)
BRAND_AGMM = re.compile(r"(?<![a-z0-9])a[\s.\-,]*g[\s.\-,]*m[\s.\-,]*m(?![a-z0-9])", re.I)


def atempo(a, sr, f):
    if abs(f - 1.0) < 0.005:
        return a
    import soundfile as sf
    tmp_in, tmp_out = "/tmp/_at_in.wav", "/tmp/_at_out.wav"
    sf.write(tmp_in, a, sr, subtype="FLOAT")
    chain = []
    ff = f
    while ff > 2.0:
        chain.append("atempo=2.0"); ff /= 2.0
    chain.append("atempo=%.4f" % ff)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", tmp_in, "-af", ",".join(chain), "-ar", str(sr), "-ac", "1", "-c:a", "pcm_f32le", tmp_out], check=True)
    y, _ = sf.read(tmp_out, dtype="float32")
    return y


def sanitize(m):
    """numbers only: drop every word and sentence text"""
    out = {k: v for k, v in m.items() if k not in ("words", "sentences", "gaps_ge_0p15", "numbers")}
    out["words"] = [{k: w.get(k) for k in ("i", "s", "e", "f0", "st", "stmax", "rel_db", "dbpk")} for w in m["words"]]
    out["sentences"] = [{k: v for k, v in s.items() if k not in ("text", "emphasis")} | {"emphasis_idx": [[sc, i] for sc, i, _w in s.get("emphasis", [])]} for s in m["sentences"]]
    out["gaps_ge_0p15"] = [{"t": g["t"], "gap": g["gap"]} for g in m["gaps_ge_0p15"]]
    out["numbers"] = [{k: v for k, v in n.items() if k != "num"} for n in m["numbers"]]
    sec = m.get("sections")
    if sec:
        out["sections"] = {k: ({kk: vv for kk, vv in v.items() if kk != "text"} if isinstance(v, dict) else ([[sc, i] for sc, i, _w in v] if k.endswith("emphasis") else v))
                           for k, v in sec.items() if k != "hook_numbers"}
        out["sections"]["hook_numbers"] = [{k: v for k, v in n.items() if k != "num"} for n in sec.get("hook_numbers", [])]
    if "turn_tail" in m:
        out["turn_tail"] = {k: v for k, v in m["turn_tail"].items() if k != "text"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True); ap.add_argument("--private", required=True); ap.add_argument("--out", required=True); ap.add_argument("--pub", required=True)
    a = ap.parse_args()
    req = json.load(open(a.request))
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.pub, exist_ok=True)
    P = req.get("params") or {}
    ref = os.path.join(a.private, req["ref"]); ref_sha = C.sha256_file(ref)
    t_all = time.time()
    eng = S.Chatterbox(ref, P)
    print("engine loaded in", eng.load_s, "s", flush=True)
    compare = C.load_compare()
    ref_embs = {os.path.basename(p): C.embed(os.path.join(a.private, p)) for p in (req.get("ref_sims") or [req["ref"]])}
    names = req.get("names") or []
    name_prompt = ("Names and terms in this script: " + ", ".join(names) + ".") if names else None
    budget_s = float(req.get("budget_minutes", 60)) * 60
    base_p = dict(eng.p)
    results = []
    maxtakes = max(int(p.get("n_takes", req.get("takes", 1))) for p in req["paragraphs"])
    for takei in range(1, maxtakes + 1):
        for pi, para in enumerate(req["paragraphs"], 1):
            if takei > int(para.get("n_takes", req.get("takes", 1))):
                continue
            if time.time() - t_all > budget_s:
                print("time budget reached, stopping", flush=True); break
            key, text = para["key"], para["text"].strip()
            seed = int(P.get("seed", 1234)) + 1000 * takei + pi
            vars_ = para.get("variants") or [{"frags": [{"t": (para.get("tts_texts") or [text])[0]}]}]
            var = vars_[(takei - 1) % len(vars_)]
            t0 = time.time()
            pieces, frag_times, cur = [], [], 0.0
            for fi, fr in enumerate(var["frags"]):
                eng.p = {**base_p, **(var.get("over") or {}), **(fr.get("over") or {})}
                cs = S.chunks(fr["t"], int(P.get("chunk_words", 32)))
                fa = []
                for ci, ch in enumerate(cs):
                    fa.append(S.trim(eng.synth(ch, seed + 31 * fi + 97 * ci), eng.sr))
                    if ci < len(cs) - 1:
                        fa.append(np.zeros(int(0.15 * eng.sr), dtype="float32"))
                f = np.concatenate(fa)
                f = atempo(f, eng.sr, float(fr.get("tempo", 1.0)))
                if fr.get("gain_db"):
                    f = f * (10 ** (float(fr["gain_db"]) / 20.0))
                pieces.append(f)
                frag_times.append({"i": fi, "start": round(cur, 3), "end": round(cur + len(f) / eng.sr, 3)})
                cur += len(f) / eng.sr
                pause = float(fr.get("pause", 0.0)) if fi < len(var["frags"]) - 1 else 0.0
                if pause > 0:
                    pieces.append(np.zeros(int(pause * eng.sr), dtype="float32")); cur += pause
                frag_times[-1]["pause_after"] = pause
            eng.p = dict(base_p)
            full = S.normalise_lufs(np.concatenate(pieces), eng.sr)
            render_s = round(time.time() - t0, 1)
            stem = "%s.t%d" % (key, takei)
            wav_path = os.path.join(a.out, stem + ".wav")
            C.write_wav24k(wav_path, full, eng.sr)
            info = C.wav_info(wav_path)
            resp = C.asr_words(wav_path, "medium.en", prompt=name_prompt)
            resp_s = C.asr_words(wav_path, "small.en")
            for _r in (resp, resp_s):
                _r["text"] = re.sub(r"(\d)\s+,(\d)", r"\1,\2", re.sub(r"(\d)\s+\.(\d)", r"\1.\2", _r["text"]))
            resp["segments"], align_info = C.align_words(wav_path, resp["segments"])
            ok_m, detail = compare(text, resp["text"]); ok_s, detail_s = compare(text, resp_s["text"])
            sm = "" if (ok_m or ok_s) else (S.soft_match(text, resp["text"]) or S.soft_match(text, resp_s["text"])
                                            or ("fold-equal: identical after the whole-file folds" if (S.fold_equal(text, resp["text"], names) or S.fold_equal(text, resp_s["text"], names)) else ""))
            rides = S.gain_ride(wav_path, resp["segments"])
            if rides:
                info = C.wav_info(wav_path)
            nat_ratio, nat_diffs = S.native_ratio(text, resp_s["text"], names)
            sp_bad = S.spot_fail(wav_path, resp["segments"])
            ok = bool(ok_m or ok_s or sm) and not sp_bad
            wpm_native = round(len(text.split()) / (info["duration_s"] / 60.0), 1)
            swpm = C.span_wpm(resp, text)
            emb = C.embed(wav_path)
            sims = {n: round(C.cosine(emb, e), 4) for n, e in ref_embs.items()}
            heard_all = (resp["text"] + " || " + resp_s["text"])
            brand = {"agm": bool(BRAND_AGM.search(resp["text"]) or BRAND_AGM.search(resp_s["text"])), "agmm_letters": bool(BRAND_AGMM.search(resp["text"]) or BRAND_AGMM.search(resp_s["text"]))}
            receipt = {"paragraph": key, "text": text, "input_sha256": C.sha256_bytes(text.encode()), "free_engine": True, "engine": eng.name, "engine_version": eng.version,
                       "free_route": "github-actions-cpu (public repo, encrypted artifact)", "reference_file": os.path.basename(ref), "reference_sha256": ref_sha,
                       "wav_sha256": C.sha256_file(wav_path), "wav_bytes": os.path.getsize(wav_path), "wav_info": {k: info[k] for k in ("channels", "rate", "width", "frames", "duration_s")},
                       "attempt": takei, "wpm": wpm_native, "span_wpm": swpm, "asr_status": "checked", "asr_engine": resp["model"], "asr_text": resp["text"], "asr_match": bool(ok),
                       "asr_compare_detail": detail, "asr_variants": {"small.en": resp_s["text"]}, "asr_match_by": {"medium.en": bool(ok_m), "small.en": bool(ok_s), "soft": sm or None},
                       "tempo_applied": None, "timing": align_info, "spot_fail": sp_bad, "gain_ride": rides, "native_ratio": nat_ratio, "native_diffs": nat_diffs, "tts_http_status": None,
                       "params": P, "variant": takei, "frags": frag_times, "seed": seed, "chunks": len(var["frags"]), "tts_variant": (takei - 1) % len(vars_), "tts_input_differs": True,
                       "render_seconds": render_s, "speaker_sim": sims, "speaker_sim_to_baseline": None, "brand": brand,
                       "run": {"id": os.environ.get("GITHUB_RUN_ID"), "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"), "sha": os.environ.get("GITHUB_SHA"),
                               "url": "https://github.com/%s/actions/runs/%s" % (os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GITHUB_RUN_ID", ""))}}
            json.dump(resp, open(os.path.join(a.out, stem + ".words.json"), "w"), indent=2, ensure_ascii=False)
            json.dump(receipt, open(os.path.join(a.out, stem + ".receipt.json"), "w"), indent=2, ensure_ascii=False)
            # delivery measurement (numbers only go public)
            try:
                words = [{"w": w["text"].strip(), "s": w["start"], "e": w["end"]} for w in resp["segments"] if w.get("type", "word") == "word" or "text" in w]
                m = DM.measure(wav_path, words=words, name=stem)
                pub = sanitize(m)
            except Exception as e:      # noqa: BLE001
                print("measure failed", stem, repr(e)[:200], flush=True); pub = {"error": repr(e)[:200]}
            pub.update({"key": key, "take": takei, "cls": para.get("cls"), "dur_s": info["duration_s"], "wpm_native": wpm_native, "span_wpm": swpm, "asr_match": bool(ok), "native_ratio": nat_ratio,
                        "brand": brand, "frags": frag_times, "variant_idx": (takei - 1) % len(vars_), "sim_ref": sims, "spot_fail": len(sp_bad), "render_s": render_s,
                        "n_script_words": len(text.split())})
            json.dump(pub, open(os.path.join(a.pub, stem + ".delivery.json"), "w"))
            row = {"key": key, "take": takei, "dur_s": info["duration_s"], "wpm_native": wpm_native, "span_wpm": swpm, "asr_match": bool(ok), "render_s": render_s, "sim_ref": sims, "chunks": len(var["frags"]),
                   "native_ratio": nat_ratio, "sim_baseline": None}
            results.append(row)
            print(json.dumps(row), flush=True)
    json.dump({"label": req.get("label"), "id": req.get("id"), "engine": eng.name, "engine_version": eng.version, "params": P, "reference_sha256": ref_sha, "model_load_s": eng.load_s,
               "total_s": round(time.time() - t_all, 1), "takes": results}, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print("done", len(results), "takes in", round(time.time() - t_all), "s", flush=True)


if __name__ == "__main__":
    main()
