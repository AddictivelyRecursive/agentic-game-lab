"""Prospective parser v3: accept literal LF/CR/TAB in complete JSON strings.

No brace completion, missing-field inference, prose action extraction, or choice
among multiple decision objects. v2-valid outputs are preserved exactly. This
changes syntax acceptance only and must not rewrite the historical primary data.
"""
import json
from frozen_parser_v2 import extract_decision_json as extract_v2
from frozen_parser_v2 import validate_decision

VERSION='v3_literal_whitespace_only_20261011'

def extract_decision_json(response):
    try:
        return extract_v2(response)
    except ValueError:
        pass
    text=response
    if '</think>' in text: text=text.rsplit('</think>',1)[1]
    elif '<think>' in text: raise ValueError('Unterminated reasoning block')
    if any(ord(c)<32 and c not in '\n\r\t' for c in text):
        raise ValueError('Unsupported literal control character')
    decoder=json.JSONDecoder(strict=False); candidates=[]; offset=0
    while offset<len(text):
        start=text.find('{',offset)
        if start<0: break
        try: value,end=decoder.raw_decode(text,start)
        except json.JSONDecodeError: offset=start+1; continue
        if isinstance(value,dict) and 'a' in value: candidates.append(value)
        offset=end
    if len(candidates)!=1: raise ValueError('Expected exactly one complete decision JSON object')
    return candidates[0]
