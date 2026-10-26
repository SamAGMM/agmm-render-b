#!/usr/bin/env python3
"""Value-based ASR-vs-script comparator that tolerates number formatting differences
between spoken word-numbers ("twenty-five million dollars") and ASR digit output
("$25 million"). Reduces both sides to a token stream where number phrases become a
single NUM token (with a numeric value attached) so the comparison catches real content
problems (dropped/added/mispronounced words, wrong numeric values) without flagging
formatting noise as a defect.
"""
import re

ONES = {'zero':0,'nought':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,
        'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,
        'seventeen':17,'eighteen':18,'nineteen':19}
TENS = {'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90}
SCALES = {'hundred':100,'thousand':1000,'million':1000000,'billion':1000000000}
NUMWORDS = set(ONES) | set(TENS) | set(SCALES) | {'and', 'a', 'an', 'half', 'point'}
# 'per'+'cent' (British spoken form, pervasive in these scripts) is equivalent to the ASR's '%'
# symbol / 'percent' digit-adjacent rendering; strip all three consistently on both sides.
STRIP_WORDS = {'dollars', 'dollar', 'pounds', 'pound', 'percent', 'per', 'cent'}
BUILDER_AI = 'builderai'
# digits usable after a spoken 'point' for a decimal fraction (0-9 only, never ten..nineteen)
FRAC_DIGITS = {'zero': '0', 'nought': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
               'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'}


def _tokenize(text):
    text = text.lower()
    text = text.replace('builder.ai', BUILDER_AI)
    text = re.sub(r"(\d)-(\d)", r"\1 \2", text)  # not expected but safe
    text = text.replace(',', '')
    return re.findall(r"\d+\.\d+|[a-z]+(?:'[a-z]+)?|\d+|\$|£|%", text)


def _parse_number_run(tokens, i):
    """Given tokens and start index i (tokens[i] is a number word), consume the maximal
    valid number-word run and return (end_index_exclusive, value) or (i, None) if tokens[i]
    is not actually a number word (e.g. bare 'a' not followed by hundred/thousand/etc)."""
    j = i
    total = 0
    current = 0
    started = False
    while j < len(tokens):
        t = tokens[j]
        if t in ONES:
            current += ONES[t]; started = True; j += 1
        elif t in TENS:
            current += TENS[t]; started = True; j += 1
        elif t == 'hundred':
            current = (current or 1) * 100; started = True; j += 1
        elif t in ('thousand', 'million', 'billion'):
            total += (current or 1) * SCALES[t]
            current = 0; started = True; j += 1
        elif t == 'and':
            # only consume 'and' if a number word plausibly follows
            if j + 1 < len(tokens) and (tokens[j+1] in ONES or tokens[j+1] in TENS or tokens[j+1] == 'a'):
                j += 1
            else:
                break
        elif t in ('a', 'an'):
            if j + 1 < len(tokens) and tokens[j+1] == 'half':
                current += 0.5; started = True; j += 2
            elif j + 1 < len(tokens) and tokens[j+1] in ('hundred', 'thousand', 'million', 'billion'):
                current += 1; started = True; j += 1
            else:
                break
        elif t == 'point' and started:
            # decimal continuation, e.g. "nought point three" -> 0.3, "fifteen point nine" -> 15.9,
            # "one point three billion" -> 1.3 then billion scales it below.
            k = j + 1
            digits = []
            while k < len(tokens) and tokens[k] in FRAC_DIGITS:
                digits.append(FRAC_DIGITS[tokens[k]])
                k += 1
            if digits:
                current += float('0.' + ''.join(digits))
                j = k
            else:
                break  # bare 'point' not followed by digits -- not a decimal, stop the number run
        else:
            break
    if not started:
        return i, None
    value = total + current
    return j, value


def _canon(text):
    """Return list of tokens with number-word / digit runs collapsed to ('NUM', value)."""
    toks = _tokenize(text)
    out = []
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if t in STRIP_WORDS:
            i += 1
            continue
        if t in ('$', '£'):
            i += 1
            continue
        if t == '%':
            i += 1
            continue
        if re.match(r'^\d+(\.\d+)?$', t):
            val = float(t)
            j = i + 1
            # allow one or more trailing scale words (e.g. "1.5 billion", "25 million")
            while j < n and toks[j] in SCALES:
                val *= SCALES[toks[j]]
                j += 1
            out.append(('NUM', val))
            i = j
            continue
        if t in ONES or t in TENS or (t in ('a', 'an') and i + 1 < n and toks[i+1] in ('half', 'hundred', 'thousand', 'million', 'billion')):
            j, val = _parse_number_run(toks, i)
            if val is not None:
                out.append(('NUM', float(val)))
                i = j
                continue
        out.append(t)
        i += 1
    return out


def compare(expected_text, asr_text, num_tol=0.02):
    """Returns (match: bool, detail: dict)."""
    exp = _canon(expected_text)
    got = _canon(asr_text)
    if len(exp) != len(got):
        return False, {'reason': 'length_mismatch', 'expected_len': len(exp), 'got_len': len(got),
                        'expected': exp, 'got': got}
    for a, b in zip(exp, got):
        if isinstance(a, tuple) and isinstance(b, tuple):
            av, bv = a[1], b[1]
            if av == 0:
                if bv != 0:
                    return False, {'reason': 'num_mismatch', 'expected': av, 'got': bv}
            elif abs(av - bv) / max(abs(av), abs(bv)) > num_tol:
                return False, {'reason': 'num_mismatch', 'expected': av, 'got': bv}
        elif a != b:
            return False, {'reason': 'word_mismatch', 'expected': a, 'got': b, 'expected_seq': exp, 'got_seq': got}
    return True, {'reason': 'ok'}


if __name__ == '__main__':
    import sys, json
    a, b = sys.argv[1], sys.argv[2]
    ok, detail = compare(a, b)
    print(json.dumps({'match': ok, 'detail': detail}, indent=2))
