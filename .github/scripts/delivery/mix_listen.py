#!/usr/bin/env python3
"""mix_listen.py MIX.mp4 OUT.json : a Gemini audio listener scores the WHOLE final mix of a Hail Mary short (27 Oct 2026). Numbers plus one short issues line leave the runner."""
import base64, json, os, subprocess, sys, time, urllib.request
mp4, out = sys.argv[1], sys.argv[2]
KEYS = [k for k in (os.environ.get("GEMINI_API_KEYS", "") + "," + os.environ.get("GEMINI_API_KEY", "")).split(",") if k.strip()]
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
Q = ("You are a strict sound director listening to the FULL MIX (voice-over, music and sound effects) of a ~48 second British social video ad. It has two halves: a first-half story that is a "
     "suspenseful, curious mystery, then a turn where the narrator's voice lifts into excitement ('straight to ...'), then a fun, bright, excited second half with its own theme tune. "
     "Score honestly on 1-10 (1 = terrible, 5 = acceptable, 10 = a pro sound designer's work). Answer with ONE JSON object. Keys, all numbers: "
     '"first_half_suspense_curious" (is the first-half music suspenseful and curious, a mystery you can feel from the first second), '
     '"second_half_fun" (is the second-half music fun and upbeat), '
     '"turn_excited" (does the turn sound excited, a clear audible lift in voice and music), '
     '"hook_delivery" (is the opening line fast, curious, with the key words stressed), '
     '"cheap_or_out_of_tune" (10 = something sounds cheap, out of tune, plucky or amateur; 1 = nothing does), '
     '"flat_or_robotic" (10 = the voice sounds flat or robotic; 1 = natural and alive), '
     '"effects_fit" (do the sound effects sit well, none noisy or out of place at the start), '
     '"mix_balance" (voice always clear over music and effects), '
     '"overall" (one number for the whole mix). Also "issues": a string of at most 40 words naming the worst audible problems with rough timestamps in seconds, or "none".')
mp3 = "/tmp/mix.mp3"
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", mp4, "-vn", "-ac", "2", "-ar", "44100", "-b:a", "128k", mp3], check=True)
b64 = base64.b64encode(open(mp3, "rb").read()).decode()
body = {"generationConfig": {"temperature": 0.2, "maxOutputTokens": 4000, "responseMimeType": "application/json"},
        "contents": [{"parts": [{"text": Q}, {"inline_data": {"mime_type": "audio/mp3", "data": b64}}]}]}
res = None
for k in range(len(KEYS) * 2 + 1):
    try:
        req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % MODEL, json.dumps(body).encode(), {"Content-Type": "application/json", "x-goog-api-key": KEYS[k % len(KEYS)]})
        j = json.load(urllib.request.urlopen(req, timeout=240))
        txt = "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"] if "text" in p)
        res = json.loads(txt[txt.find("{"):txt.rfind("}") + 1]); break
    except Exception as e:   # noqa: BLE001
        print("retry", k, repr(e)[:100], flush=True); time.sleep(5)
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res))
