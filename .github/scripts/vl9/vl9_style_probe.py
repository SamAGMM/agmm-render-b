#!/usr/bin/env python3
"""VL9: does an mstts:express-as style survive the free Edge endpoint? Same neutral text, plain vs styled SSML
(injected by patching edge_tts.communicate.mkssml, since Communicate escapes any SSML in the text). Compares audio bytes and F0."""
import asyncio, hashlib, json, subprocess, sys
import numpy as np, soundfile as sf, parselmouth
import edge_tts, edge_tts.communicate as C

TEXT = "One small push, and everything changes. Honestly, you would not believe how fast it happens."
ORIG = C.mkssml
SEEN = {}


def make_styled(style):
    def mk(tc, escaped_text):
        s = ORIG(tc, escaped_text)
        s = s.replace("xmlns='http://www.w3.org/2001/10/synthesis'", "xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts'", 1)
        s = s.replace("<prosody", "<mstts:express-as style='%s' styledegree='2'><prosody" % style, 1).replace("</prosody>", "</prosody></mstts:express-as>", 1)
        SEEN["ssml_has_style"] = "express-as" in s
        return s
    return mk


async def synth(voice, path):
    n = 0
    with open(path, "wb") as f:
        async for ch in edge_tts.Communicate(TEXT, voice).stream():
            if ch["type"] == "audio":
                f.write(ch["data"]); n += len(ch["data"])
    return n


def f0stats(mp3):
    wav = mp3 + ".wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp3, "-ac", "1", "-ar", "24000", wav], check=True)
    y, sr = sf.read(wav)
    p = parselmouth.Sound(y, sampling_frequency=sr).to_pitch_ac(time_step=0.01, pitch_floor=70, pitch_ceiling=500).selected_array["frequency"]
    v = p[p > 0]; st = 12 * np.log2(v / np.median(v))
    return {"dur_s": round(len(y) / sr, 3), "f0_median_hz": round(float(np.median(v)), 1), "f0_sd_st": round(float(st.std()), 2),
            "f0_range_st_p5_p95": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2)}


def main(out):
    rows = []
    for voice, style in (("en-US-AriaNeural", "cheerful"), ("en-US-JennyNeural", "excited"), ("en-US-GuyNeural", "cheerful"), ("en-US-BrianMultilingualNeural", "cheerful")):
        row = {"voice": voice, "style": style}
        try:
            C.mkssml = ORIG
            nb = asyncio.run(synth(voice, "/tmp/plain.mp3"))
            row["plain"] = {"bytes": nb, "sha": hashlib.sha256(open("/tmp/plain.mp3", "rb").read()).hexdigest()[:16], **f0stats("/tmp/plain.mp3")}
            C.mkssml = make_styled(style); SEEN.clear()
            nb = asyncio.run(synth(voice, "/tmp/styled.mp3"))
            row["styled"] = {"bytes": nb, "sha": hashlib.sha256(open("/tmp/styled.mp3", "rb").read()).hexdigest()[:16], **f0stats("/tmp/styled.mp3")}
            row["ssml_sent_with_style"] = SEEN.get("ssml_has_style", False)
            row["identical_audio"] = row["plain"]["sha"] == row["styled"]["sha"]
        except Exception as e:  # noqa: BLE001
            row["error"] = repr(e)[:200]
        finally:
            C.mkssml = ORIG
        rows.append(row); print(json.dumps(row), flush=True)
    json.dump({"edge_tts_version": getattr(edge_tts, "__version__", "?"), "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
