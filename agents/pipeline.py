"""Orchestration of the four agents around the deterministic fact check. Pure Python, no model imports,
so the whole loop is testable with scripted agents.

Roles (each is a separate agent with its own instructions, see agents/runner.py):
  analyst   reads match facts through MCP and picks what matters and why
  narrator  turns that into structured claims plus a skeleton sentence with placeholders
  checker   code verifies every claim; an LLM pass looks for unsupported wording
  persona   rewrites the skeleton for a persona, then a translation pass for another language
Persona and translation only ever see the skeleton, never raw numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from intel import claims as C
from intel import lexicon as L
from intel.data import Match
from intel.moments import detect_moments

MAX_REWRITES = 2
MENU_METRICS = (("shots", "shots"), ("shots_on_target", "sot"), ("xg", "xg"),
                ("possession_pct", "poss"), ("pass_accuracy_pct", "passacc"))
PERSONAS = ("analyst", "casual", "player_focus", "accessibility")

# an agent call: (role, payload) -> parsed JSON dict
Runner = Callable[[str, dict], Awaitable[dict]]


@dataclass
class Commentary:
    moment_id: str
    skeleton: str
    claims: list
    verified: dict
    text: str
    catches: list = field(default_factory=list)   # every rejected draft, kept so the UI can show "caught and fixed"
    attempts: int = 1
    used_fallback: bool = False
    moment: dict | None = None


def template_for(moment: dict, match: Match):
    """Deterministic fallback sentence built only from the detector's own output."""
    k, eid = moment["kind"], (moment["evidence_event_ids"] or [None])[0]
    if k == "goal":
        return ([{"id": "c1", "kind": "event", "event_id": eid, "expect": {"type": "shot", "outcome": "goal"}}],
                "{c1.clock} GOAL! {c1.player} scores for {c1.team}. It is {c1.score}.")
    if k == "red_card":
        return ([{"id": "c1", "kind": "event", "event_id": eid, "expect": {"type": "card"}}],
                "{c1.clock} Red card for {c1.player} of {c1.team}.")
    if k == "big_chance_missed":
        return ([{"id": "c1", "kind": "event", "event_id": eid, "expect": {"type": "shot"}}],
                "{c1.clock} A big chance for {c1.player} of {c1.team}.")
    side = "home" if moment.get("team") == match.home else "away"
    if k == "press_spike":
        return ([], f"{{team:{side}}} have stepped up the pressure.")
    if k == "control_shift":
        if moment["facts"].get("label") == "dominant":
            return ([], f"{{team:{side}}} are in control.")
        return ([], "The game has become stretched and chaotic.")
    return ([], "")


def _causal_problems(moment: dict, causal: list) -> list:
    """Causal wording is only allowed when a detector found that driver for this moment."""
    have = {c["kind"] for c in moment.get("context", [])}
    return [f"causal link {c.get('driver_kind')!r} is not supported by the detectors for this moment"
            for c in causal or [] if c.get("driver_kind") not in have]


METRIC_WORDS = {"possession_pct": "possession", "xg": "expected goals (xG)", "pass_accuracy_pct": "pass accuracy",
                "shots": "shots", "shots_on_target": "shots on target", "press_intensity": "pressures per opposition pass"}


def fact_lines(verified: dict, moment: dict | None = None) -> list:
    """Plain English list of every verified fact, so the LLM reviewer can see what is already supported."""
    out = []
    for cid, v in verified.items():
        k = v["kind"]
        if k == "stat":
            who = v.get("player") or v.get("team")
            out.append(f"{{{cid}}} = {C.format_value(v['metric'], v['value'])}: {who} {METRIC_WORDS.get(v['metric'], v['metric'])}, "
                       f"from minute {int(v['from_t'] // 60)} to minute {int(v['to_t'] // 60)}")
        elif k == "event":
            out.append(f"{cid}: {v['type']} by {v['player']} for {v['team']} at {v['clock']}', outcome {v['outcome']}, score {v['score'][0]}:{v['score'][1]}")
        elif k == "state":
            out.append(f"{cid}: control label {v['label']} ({v['team']}) at {v['clock']}'")
        elif k == "threshold":
            out.append(f"{cid}: {v['player']} passed {v['level']} {v['threshold']}")
    if moment:
        mf = {a: b for a, b in (moment.get("facts") or {}).items() if b is not None}
        out.append(f"moment facts: kind {moment['kind']}, " + ", ".join(f"{a} {b}" for a, b in mf.items()))
        out += [f"moment context: {c['kind']}" for c in moment.get("context", [])]
    return out


def allowed_placeholders(menu: dict, match: Match, as_of: float) -> list:
    """Exact placeholders that are valid for this moment, built from the verified menu, so the model can pick, not guess."""
    res = C.verify_claims(match, list(menu.values()), as_of)
    out = []
    for r in res:
        if not r["ok"]:
            continue
        v = r["verified"]
        for f in sorted(C._FIELDS.get(v["kind"], set()), key=lambda x: x or ""):
            if C._field_problem(v, f) is None:
                out.append("{%s}" % r["id"] if f is None else "{%s.%s}" % (r["id"], f))
    return out


def build_menu(match: Match, moment: dict, as_of: float) -> dict:
    """Claims already computed by code for this moment. The Narrator picks from these (copying the object
    unchanged) and may add its own, which still get verified. Each id doubles as a placeholder name."""
    from intel.stats import team_stats
    menu = {}
    ts = team_stats(match, as_of)
    for side, team in (("home", match.home), ("away", match.away)):
        for metric, short in MENU_METRICS:
            cid = f"{side}_{short}"
            menu[cid] = {"id": cid, "kind": "stat", "team": team, "metric": metric, "value": round(ts[team][metric], 4),
                         "from_t": 0, "to_t": as_of}
    eid = (moment["evidence_event_ids"] or [None])[0]
    if eid is not None:
        e = match.by_id[eid]
        expect = {"type": e["type"], "team": e["team"], "player_id": e["player_id"]}
        if e["type"] == "shot":
            expect["outcome"] = e["outcome"]
        menu["ev"] = {"id": "ev", "kind": "event", "event_id": eid, "expect": expect}
    f = moment["facts"]
    if moment["kind"] == "press_spike":
        from intel.stats import team_stats as _ts
        recent = _ts(match, moment["t"], f["window_from_t"])[moment["team"]]["press_intensity"]
        before = _ts(match, f["window_from_t"], 0.0)[moment["team"]]["press_intensity"]
        menu["press_recent"] = {"id": "press_recent", "kind": "stat", "team": moment["team"], "metric": "press_intensity",
                                "value": round(recent, 4), "from_t": f["window_from_t"], "to_t": moment["t"]}
        menu["press_before"] = {"id": "press_before", "kind": "stat", "team": moment["team"], "metric": "press_intensity",
                                "value": round(before, 4), "from_t": 0, "to_t": f["window_from_t"]}
    if moment["kind"] == "control_shift":
        menu["ctrl"] = {"id": "ctrl", "kind": "state", "label": f["label"], "team": moment["team"], "at_t": moment["t"]}
    ok = {r["id"] for r in C.verify_claims(match, list(menu.values()), as_of) if r["ok"]}
    return {k: v for k, v in menu.items() if k in ok}


async def narrate_moment(match: Match, moment: dict, as_of: float, run: Runner, analyst_note: dict | None = None) -> Commentary:
    if moment["t"] > as_of + 1e-9:
        raise ValueError("cannot narrate a moment that has not happened yet at as_of")
    menu = build_menu(match, moment, as_of)
    allowed = allowed_placeholders(menu, match, as_of)
    catches, feedback = [], []
    for attempt in range(MAX_REWRITES + 1):
        draft = await run("narrator", {"moment": moment, "analyst": analyst_note, "feedback": feedback, "menu": menu,
                                       "allowed_placeholders": allowed,
                                       "match_id": match.match_id, "as_of": as_of})
        sk = draft.get("skeleton", "") or ""
        # Only claims the sentence actually uses matter. Menu ids always use code's own copy, so the model cannot
        # mangle them; any other referenced id must be a claim the model wrote itself and it is fully verified.
        used = {re.match(r"\{([A-Za-z_]\w*)", ph).group(1) for ph in C.placeholders(sk)
                if re.match(r"\{[A-Za-z_]\w*", ph) and not ph.startswith(("{player:", "{team:"))}
        own = {c.get("id"): c for c in (draft.get("claims", []) or []) if isinstance(c, dict)}
        cl = [menu[i] if i in menu else own[i] for i in sorted(used) if i in menu or i in own]
        results = C.verify_claims(match, cl, as_of)
        verified = {r["id"]: r["verified"] for r in results if r["ok"]}
        problems = [{"kind": "claim", "claim_id": r["id"], "claim": next((c for c in cl if c.get("id") == r["id"]), None),
                     "reason": r["reason"]} for r in results if not r["ok"]]
        problems += [{"kind": "wording", "reason": p, "skeleton": sk} for p in C.check_skeleton(sk, verified)]
        problems += [{"kind": "wording", "reason": p, "skeleton": sk} for p in L.check_wording(sk, moment)]
        problems += [{"kind": "causal", "reason": p} for p in _causal_problems(moment, draft.get("causal"))]
        if not problems:
            text = C.render(match, sk, verified)
            review = await run("checker", {"mode": "narration", "text": text, "facts": fact_lines(verified, moment)})
            problems += [{"kind": "unsupported", "reason": u} for u in review.get("unsupported", []) or []]
            if not problems:
                return Commentary(moment["id"], sk, cl, verified, text, catches, attempt + 1, moment=moment)
        catches.append({"attempt": attempt + 1, "rejected_skeleton": sk, "problems": problems})
        feedback = [p["reason"] for p in problems]
    cl, sk = template_for(moment, match)
    results = C.verify_claims(match, cl, as_of)
    verified = {r["id"]: r["verified"] for r in results if r["ok"]}
    if len(verified) != len(cl) or C.check_skeleton(sk, verified):
        raise RuntimeError("template fallback failed verification; this is a bug")
    return Commentary(moment["id"], sk, cl, verified, C.render(match, sk, verified), catches, MAX_REWRITES + 1, True, moment)


async def adapt(match: Match, c: Commentary, run: Runner, persona: str = "casual", locale: str = "en",
                favorite_team: str | None = None, player_id: str | None = None) -> dict:
    """Persona then translation. Both may only rearrange words around the same verified placeholders."""
    if persona not in PERSONAS:
        raise ValueError(persona)
    allowed = set(C.placeholders(c.skeleton))
    sk, notes = c.skeleton, []
    for step, payload in (("persona", {"skeleton": c.skeleton, "persona": persona, "favorite_team": favorite_team,
                                       "player_id": player_id, "allowed_placeholders": sorted(allowed)}),
                          ("translate", {"skeleton": None, "locale": locale})):
        if step == "translate" and locale == "en":
            break
        payload = dict(payload, skeleton=sk)
        ok = False
        for _ in range(2):
            out = (await run(step, payload)).get("skeleton", "")
            fine = not C.check_skeleton(out, c.verified) and set(C.placeholders(out)) <= allowed
            if step == "translate":
                fine = fine and C.same_placeholders(sk, out)
            else:
                fine = fine and not L.check_wording(out, c.moment)
            if fine:                                       # independent LLM review: did the rewrite add or change a fact?
                rev = await run("checker", {"mode": "translation" if step == "translate" else "persona", "source": sk, "candidate": out, "facts": fact_lines(c.verified, c.moment)})
                fine = not (rev.get("unsupported") or [])
            if fine:
                sk, ok = out, True
                break
        if not ok:
            notes.append(f"{step} output rejected, kept previous wording")
    return {"text": C.render(match, sk, c.verified, locale if locale in C.LOCALES else "en"), "skeleton": sk,
            "persona": persona, "locale": locale, "notes": notes}


async def commentate(match: Match, as_of: float, run: Runner, since_t: float = 0.0, top_k: int = 3) -> list:
    """One update tick: analyst chooses what matters among new moments, narrator writes each, checker gates it."""
    new = [m for m in detect_moments(match, as_of) if m["t"] > since_t]
    if not new:
        return []
    pick = await run("analyst", {"match_id": match.match_id, "as_of": as_of, "moments": new, "top_k": top_k})
    ids = [i for i in pick.get("moment_ids", []) if i in {m["id"] for m in new}] or \
          [m["id"] for m in sorted(new, key=lambda m: -m["importance"])[:top_k]]
    notes = {n.get("moment_id"): n for n in pick.get("notes", []) or []}
    out = []
    for m in new:
        if m["id"] in ids[:top_k]:
            out.append(await narrate_moment(match, m, as_of, run, notes.get(m["id"])))
    return out
