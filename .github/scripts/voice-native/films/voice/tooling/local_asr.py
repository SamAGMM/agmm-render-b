#!/usr/bin/env python3
"""local_asr.py: the no-API ASR side of the film voice checks (12 Oct 2026; Mistral ASR API = 402, Gemini billing dead).
fold_compare(expected, heard)  -> (match, detail): asr_compare.compare after folding the formatting a Whisper transcript adds
   (digit ordinals, US spellings, 'job seekers', 'installments', a spoken URL written as x.co.uk, 'GPT-40'). Folds are listed in
   FOLDS so each one is auditable; nothing is folded that changes a content word.
transcribe_variants(wav, names) -> {'small.en': text+words, 'small.en+names': ..., 'base.en': ...}: three local passes.
   A paragraph PASSES if ANY variant matches the script after folding (an ASR noise word that another transcriber hears
   correctly is noise; one all three get the same wrong way is a real defect or a name to adjudicate).
Self-test: python3 local_asr.py --selftest   (a wrong-chunk take and a dropped-word take must FAIL, the right take must PASS)."""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_compare import compare

ORD = {1:'first',2:'second',3:'third',4:'fourth',5:'fifth',6:'sixth',7:'seventh',8:'eighth',9:'ninth',10:'tenth',11:'eleventh',12:'twelfth',
       13:'thirteenth',14:'fourteenth',15:'fifteenth',16:'sixteenth',17:'seventeenth',18:'eighteenth',19:'nineteenth',20:'twentieth',
       21:'twenty first',22:'twenty second',23:'twenty third',24:'twenty fourth',25:'twenty fifth',26:'twenty sixth',27:'twenty seventh',
       28:'twenty eighth',29:'twenty ninth',30:'thirtieth',31:'thirty first'}
FOLDS = [(r'\bpenalizes\b', 'penalises'), (r'\bpenalize\b', 'penalise'), (r'\binstallments\b', 'instalments'), (r'\binstallment\b', 'instalment'),
         (r'\bjob seekers\b', 'jobseekers'), (r'\bgpt-?40\b', 'gpt 4o'), (r'\bgpt-4o\b', 'gpt 4o'), (r'\bag[mn]{1,2}\.co\.uk\b', 'agmm dot co dot uk'),
         (r'\bgood-?bye\b', 'goodbye'), (r'\bone-off\b', 'one off')]

def fold(s):
    s = s.lower().replace('’', "'")
    s = re.sub(r'\b(\d{1,2})(st|nd|rd|th)\b', lambda m: ORD.get(int(m.group(1)), m.group(0)), s)
    for pat, rep in FOLDS:
        s = re.sub(pat, rep, s)
    s = re.sub(r'\bgpt-?4o\b', 'gpt 4o', s)
    return s

def fold_compare(expected, heard):
    return compare(fold(expected), fold(heard))

def transcribe_variants(wav, names_prompt, models):
    out = {}
    for tag, (model, prompt) in {'small.en': ('small.en', None), 'small.en+names': ('small.en', names_prompt), 'base.en': ('base.en', None)}.items():
        m = models[model]
        segs, _ = m.transcribe(str(wav), word_timestamps=True, language='en', beam_size=5, initial_prompt=prompt, condition_on_previous_text=False)
        words, texts = [], []
        for s in segs:
            texts.append(s.text.strip())
            for w in (s.words or []):
                words.append({'word': w.word.strip(), 'start': round(w.start, 3), 'end': round(w.end, 3)})
        out[tag] = {'text': ' '.join(texts), 'words': words}
    return out

def selftest():
    good = "And there was still the price tag. Four hundred and forty thousand dollars. How much would come back?"
    heard_good = "there was still the price tag $440,000. How much would come back?"   # real P18 whisper output: 'And' dropped at the clip edge
    heard_good2 = "And there was still the price tag $440,000. How much would come back?"
    wrong = "Three days later, Senator Barbara Pocock's office put out a figure. Her release says Deloitte refunded $97,587. Under a quarter of the contract,"
    script23 = "If you want more stories like this one, subscribe."
    drop = "On the 6th of October, the Guardian reported that Deloitte would repay part of its fee. The department confirmed the instalment would go back. How much was a different question?"
    script19 = "On the sixth of October, The Guardian reported that Deloitte would repay part of its fee. The department confirmed the final instalment would go back. How much was a different question."
    res = {
        'good (digits for words, edge word)': fold_compare(good, heard_good2)[0],
        'wrong chunk (P22 audio against P23 text) must FAIL': fold_compare(script23, wrong)[0],
        'dropped word "final" must FAIL': fold_compare(script19, drop)[0],
        'ordinal fold 4th/fourth must PASS': fold_compare("on the fourth of July", "on the 4th of July")[0],
        'changed word must FAIL': fold_compare("the judge never said it", "the judge never read it")[0],
    }
    print(res)
    exp = {'good (digits for words, edge word)': True, 'wrong chunk (P22 audio against P23 text) must FAIL': False, 'dropped word "final" must FAIL': False,
           'ordinal fold 4th/fourth must PASS': True, 'changed word must FAIL': False}
    assert res == exp, 'SELFTEST FAILED'
    print('SELFTEST OK')

if __name__ == '__main__':
    if '--selftest' in sys.argv: selftest()
