#!/usr/bin/env python3
"""delivery_listen.py PACKDIR "P1.t2 P1.t4 ..." OUT.json : a Gemini audio listener scores chosen takes. Numbers only leave the runner (no script text in the output)."""
import base64, json, os, subprocess, sys, time, urllib.request, urllib.error
pack, takes, out = sys.argv[1], sys.argv[2].split(), sys.argv[3]
KEY = os.environ["GEMINI_API_KEY"]
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
CLS = {"P1": "hook", "P5": "turn", "P6": "ad", "P7": "cta"}
BASE = ("You are a strict voice director listening to ONE line of a 35-second British social video ad narration (cloned human voice, text to speech). Score honestly on 1-10 "
        "(1 = terrible, 5 = acceptable, 10 = a pro narrator). Answer with a single JSON object, numbers only, no quoting of the words. ")
Q = {
 "hook": BASE + 'This is the HOOK, the first line. It must sound curious and gripping, said FAST, with the surprising number or the strange idea stressed and a short beat of intrigue, not slow or sing-song. Keys: "curious_gripping", "fast_enough" (10 = brisk and natural, 1 = slow), "emphasis_clear" (are the curious points stressed), "flat_robotic" (10 = very flat or robotic), "overall".',
 "turn": BASE + 'This is the TURN LINE. It starts in a calm curious story voice and must end with a clear, audible lift in excitement and pitch on the last words, like a presenter revealing a punchline, ending on a short name said fast as the letters A G M. Keys: "excited_lift" (10 = a clear audible excitement and tone change at the end), "fun_energy", "reveal_clear" (the last word sounds like the letters A G M, clean, not garbled or like I G M), "flat_robotic" (10 = very flat or robotic), "overall".',
 "ad": BASE + 'This is the AD HALF after the reveal. It must sound fun, bright and excited, upbeat and punchy, not a flat read. Keys: "excited_fun", "bright", "punchy_fragments", "flat_robotic" (10 = very flat or robotic), "overall".',
 "cta": BASE + 'This is the closing call to action ending with a website. All FOUR letters of the name must be spoken one by one (A, G, M, M) in the web address. Keys: "four_letters_clear" (10 = all four letters A G M M clearly heard, 1 = fewer or garbled), "excited_fun", "flat_robotic" (10 = very flat or robotic), "overall".',
 "story": BASE + 'This is a STORY line in the first half: curious, engaging, brisk, escalating, not monotone. Keys: "curious_engaging", "brisk", "flat_robotic" (10 = very flat or robotic), "overall".',
}
res = {}
for t in takes:
    key = t.split(".")[0]; cls = CLS.get(key, "story")
    wav = os.path.join(pack, t + ".wav"); mp3 = "/tmp/%s.mp3" % t
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", wav, "-ac", "1", "-b:a", "96k", mp3], check=True)
    b64 = base64.b64encode(open(mp3, "rb").read()).decode()
    body = {"generationConfig": {"temperature": 0.2, "maxOutputTokens": 600, "responseMimeType": "application/json"},
            "contents": [{"parts": [{"text": Q[cls]}, {"inline_data": {"mime_type": "audio/mp3", "data": b64}}]}]}
    r = None
    for k in range(6):
        try:
            req = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % MODEL, json.dumps(body).encode(), {"Content-Type": "application/json", "x-goog-api-key": KEY})
            j = json.load(urllib.request.urlopen(req, timeout=180))
            txt = "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"] if "text" in p)
            r = json.loads(txt[txt.find("{"):txt.rfind("}") + 1]); break
        except Exception as e:      # noqa: BLE001
            msg = repr(e)[:120]
            print(t, "retry", k, msg.replace(KEY, "***"), flush=True)
            time.sleep(15 * (k + 1))
    res[t] = {"cls": cls, "scores": r}
    print(t, cls, json.dumps(r), flush=True)
    time.sleep(4)
json.dump(res, open(out, "w"), indent=1)
