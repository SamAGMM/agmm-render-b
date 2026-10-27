#!/usr/bin/env python3
"""delivery_pack_measure.py PACKDIR OUTDIR : measure every take of a decrypted voicefree pack (words from <stem>.words.json, aligned) with delivery_measure and write numbers-only <stem>.delivery.json."""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import delivery_measure as DM

BRAND_AGM = re.compile(r"(?<![a-z0-9])a[\s.\-,]*g[\s.\-,]*m(?![a-z0-9])(?![\s.\-,]*m(?![a-z0-9]))", re.I)
BRAND_AGMM = re.compile(r"(?<![a-z0-9])a[\s.\-,]*g[\s.\-,]*m[\s.\-,]*m(?![a-z0-9])", re.I)


CLS = {"P1": "hook", "P5": "turn", "P6": "ad", "P7": "cta"}
for _kv in (os.environ.get("CLS_MAP") or "").split(","):
    if "=" in _kv:
        _k, _v = _kv.split("="); CLS[_k.strip()] = _v.strip()


def _recall(script, a, b):
    norm = lambda t: re.findall(r"[a-z0-9']+", t.lower())
    sw = norm(script); out = {}
    for name, t in (("medium", a), ("small", b)):
        hw = norm(t); hs = set(hw)
        out[name] = {"recall": round(sum(1 for w in sw if w in hs) / max(1, len(sw)), 3), "first3": all(w in hs for w in sw[:3]), "last3": all(w in hs for w in sw[-3:])}
    return out


def sanitize(m):
    out = {k: v for k, v in m.items() if k not in ("words", "sentences", "gaps_ge_0p15", "numbers", "overall", "sections", "turn_tail", "name")}
    out["overall"] = {k: v for k, v in m["overall"].items() if k != "text"}
    out["words"] = [{k: w.get(k) for k in ("i", "s", "e", "f0", "st", "stmax", "rel_db", "dbpk")} for w in m["words"]]
    out["sentences"] = [{k: v for k, v in s.items() if k not in ("text", "emphasis")} | {"emphasis_idx": [[sc, i] for sc, i, _w in s.get("emphasis", [])]} for s in m["sentences"]]
    out["gaps_ge_0p15"] = [{"t": g["t"], "gap": g["gap"]} for g in m["gaps_ge_0p15"]]
    return out


def main():
    pack, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for rp in sorted(glob.glob(os.path.join(pack, "*.receipt.json"))):
        stem = os.path.basename(rp)[:-len(".receipt.json")]
        rc = json.load(open(rp)); wj = json.load(open(os.path.join(pack, stem + ".words.json")))
        wav = os.path.join(pack, stem + ".wav")
        words = [{"w": w["text"].strip(), "s": w["start"], "e": w["end"]} for w in wj["segments"] if w.get("text")]
        try:
            pub = sanitize(DM.measure(wav, words=words, name=stem))
        except Exception as e:      # noqa: BLE001
            print("measure failed", stem, repr(e)[:200]); pub = {"error": repr(e)[:200]}
        asr = (rc.get("asr_text") or "") + " || " + ((rc.get("asr_variants") or {}).get("small.en") or "")
        key, take = stem.split(".t")[0], int(stem.split(".t")[1])
        pub.update({"key": key, "take": take, "cls": CLS.get(key, "story"), "dur_s": rc["wav_info"]["duration_s"], "wpm_native": rc.get("wpm"), "span_wpm": rc.get("span_wpm"),
                    "asr_match": rc.get("asr_match"), "native_ratio": rc.get("native_ratio"), "brand": {"agm": bool(BRAND_AGM.search(asr)), "agmm_letters": bool(BRAND_AGMM.search(asr))},
                    "frags": rc.get("frags"), "variant_idx": rc.get("tts_variant"), "sim_ref": rc.get("speaker_sim"), "spot_fail": len(rc.get("spot_fail") or []), "n_script_words": len(rc["text"].split()), "recall": _recall(rc["text"], rc.get("asr_text") or "", (rc.get("asr_variants") or {}).get("small.en") or ""),
                    "tail_heard": (" ".join((rc.get("asr_text") or "").split()[-4:]) + " | " + " ".join(((rc.get("asr_variants") or {}).get("small.en") or "").split()[-4:])) if CLS.get(key) in ("turn", "cta") else None})
        json.dump(pub, open(os.path.join(out, stem + ".delivery.json"), "w"))
        print(stem, "ok", flush=True)


if __name__ == "__main__":
    main()
