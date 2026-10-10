"""POST-HOC / EXPLORATORY frozen high-specificity rules; no model calls.

Rules seek explicit local claims. They do not classify general intentions,
conditional alternatives, opponent predictions, or all semantic contradictions.
R1 is an explicit erroneous map for action 4; R2 ties an own/selected action
to cooperate/defect; R3 uses an own/selected numeric cooperation level;
R4 uses an explicit submitted/own chosen action number. No outcome tuning.
Known-example verification is the only permitted rule repair before prevalence.
"""
import re

ACTION4=r'(?:\baction\s*\(?\s*4\)?|\ba\s*=\s*4\b)'
COOP=r'(?:full\s+cooperation|most\s+cooperative|fully\s+cooperative|cooperative|cooperation\s+level\s+(?:of\s+)?1(?:\.0+)?\b)'
R1_PATTERNS=[
    re.compile(ACTION4+r'\s*[,):\-]?\s*(?:(?:is|represents|means|corresponds\s+to|indicates)\s+)(?:the\s+)?'+COOP,re.I),
    re.compile(COOP+r'\s+(?:action\s*)?\(\s*(?:action\s*)?4\s*\)',re.I),
    re.compile(r'\b(?:cooperation|cooperate)\s*\(\s*action\s*4\s*\)',re.I),
]
OWN=r'(?:\bI\s+(?:(?:will|should|therefore)\s+)?(?:choose|chose|select|selected|am\s+choosing)|\bmy\s+(?:chosen|selected)\s+action\s+is|\bthe\s+(?:chosen|selected|submitted)\s+action\s+is)'
CHOSEN=re.compile(OWN+r'\s+(?:action\s*|a\s*=\s*)?([0-4])\b',re.I)
SEMANTIC=re.compile(r'(?:\bI\s+(?:will\s+|choose\s+to\s+|decide\s+to\s+)?|\bmy\s+(?:decision|action)\s+is\s+(?:to\s+)?|\bthe\s+(?:chosen|selected)\s+action\s+is\s+)(cooperate|defect|full\s+cooperation|full\s+defection)\b',re.I)
NUMERIC=re.compile(r'(?:\bmy\s+(?:chosen\s+)?cooperation\s+level\s*(?:is|of|=|:)\s*|\b(?:chosen|selected)\s+action[^.;!?]{0,35}?\bcooperation\s+level\s*(?:is|of|=|:)\s*)([01](?:\.\d+)?)(\s*%)?',re.I)
NEGATION=re.compile(r'\b(?:not|never|neither|no|if|would|could|might)\b',re.I)

def allowed(text,match):
    # Avoid negated/hypothetical local assertions and opponent-owned clauses.
    prefix=text[max(0,match.start()-45):match.start()]
    clause=re.split(r'[.;!?]',prefix)[-1]
    return not NEGATION.search(clause) and not re.search(r'\b(?:opponent|other\s+player)\s+(?:will|would|might)\s*$',clause,re.I)

def detect(text,action):
    evidence={f'R{i}':[] for i in range(1,5)}
    for pattern in R1_PATTERNS:
        evidence['R1'] += [m.group(0) for m in pattern.finditer(text) if allowed(text,m)]
    for m in SEMANTIC.finditer(text):
        semantic=m.group(1).lower(); wrong=(action==0 and 'defect' in semantic) or (action==4 and ('cooperat' in semantic))
        if wrong and allowed(text,m): evidence['R2'].append(m.group(0))
    for m in NUMERIC.finditer(text):
        value=float(m.group(1))/(100 if m.group(2) else 1)
        if abs(value-(1-action/4))>1e-6 and allowed(text,m): evidence['R3'].append(m.group(0))
    for m in CHOSEN.finditer(text):
        if int(m.group(1))!=action and allowed(text,m): evidence['R4'].append(m.group(0))
    return {**{k:bool(v) for k,v in evidence.items()},'any_rule':any(evidence.values()),'evidence':evidence}
