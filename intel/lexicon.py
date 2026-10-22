"""Deterministic wording rules. The LLM checker is probabilistic, so the words we know are risky are checked by code:
evaluative adjectives, guessed timing, ordinals, and context words (substitute, red card, press, equaliser...) that
are only allowed when a detector found that fact for the moment. English only: translations go through the LLM review."""
from __future__ import annotations

import re

PH = re.compile(r"\{[^{}]*\}")
EVAL = re.compile(r"\b(crucial|vital|important|massive|huge|stunning|brilliant|superb|incredible|dramatic|fantastic|"
                  r"wonderful|spectacular|sensational|ruthless|clinical|devastating|thrilling|pivotal|decisive|"
                  r"momentum|blow|heartbreak\w*|dream)\b", re.I)
TIMING = re.compile(r"\b(just after|shortly after|moments after|minutes after|soon after|immediately|near the end|"
                    r"late in|early in|end of the (period|half|match|game)|start of the)\b", re.I)
ORDINAL = re.compile(r"\b(first|second|third|fourth|last|final|opening|only|another|again|brace|hat.?trick|winner|"
                     r"winning|so far this season)\b", re.I)
NEEDS = [  # (pattern, driver kinds that allow it, moment kinds that allow it, goal tags that allow it)
    (re.compile(r"substitut|came on|coming on|introduced|off the bench|from the bench", re.I), {"scorer_was_substitute"}, set(), set()),
    (re.compile(r"red card|sent off|dismiss|ten men|10 men|a man down|man advantage", re.I), {"red_card_earlier"}, {"red_card"}, set()),
    (re.compile(r"\bpress(ing)?\b|pressure", re.I), {"press_spike_active"}, {"press_spike"}, set()),
    (re.compile(r"equali[sz]er|draw level|level(l)?ed", re.I), set(), set(), {"equaliser"}),
    (re.compile(r"go(ing)? ahead|go.ahead|took the lead|taken the lead|goes? in front|opener", re.I), set(), set(), {"go_ahead"}),
    (re.compile(r"extends? (the|their) lead", re.I), set(), set(), {"extends_lead"}),
]


def check_wording(skeleton: str, moment: dict | None) -> list:
    text = PH.sub(" ", skeleton)
    problems = []
    for rx, label in ((EVAL, "evaluative word"), (TIMING, "timing the data does not support"),
                      (ORDINAL, "ordinal or milestone claim the data does not support")):
        m = rx.search(text)
        if m:
            problems.append(f"{label}: {m.group(0)!r}. Describe only what the claims show.")
    moment = moment or {}
    drivers = {c["kind"] for c in moment.get("context", [])}
    tag = (moment.get("facts") or {}).get("tag")
    for rx, d_ok, k_ok, t_ok in NEEDS:
        m = rx.search(text)
        if m and not (drivers & d_ok or moment.get("kind") in k_ok or tag in t_ok):
            problems.append(f"{m.group(0)!r} is not supported by the detectors for this moment")
    return problems
