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

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from intel import claims as C
from intel.data import Match
from intel.moments import detect_moments

MAX_REWRITES = 2
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


async def narrate_moment(match: Match, moment: dict, as_of: float, run: Runner, analyst_note: dict | None = None) -> Commentary:
    if moment["t"] > as_of + 1e-9:
        raise ValueError("cannot narrate a moment that has not happened yet at as_of")
    catches, feedback = [], []
    for attempt in range(MAX_REWRITES + 1):
        draft = await run("narrator", {"moment": moment, "analyst": analyst_note, "feedback": feedback,
                                       "match_id": match.match_id, "as_of": as_of})
        cl, sk = draft.get("claims", []) or [], draft.get("skeleton", "") or ""
        results = C.verify_claims(match, cl, as_of)
        verified = {r["id"]: r["verified"] for r in results if r["ok"]}
        problems = [{"kind": "claim", "claim_id": r["id"], "claim": next((c for c in cl if c.get("id") == r["id"]), None),
                     "reason": r["reason"]} for r in results if not r["ok"]]
        problems += [{"kind": "wording", "reason": p, "skeleton": sk} for p in C.check_skeleton(sk, verified)]
        problems += [{"kind": "causal", "reason": p} for p in _causal_problems(moment, draft.get("causal"))]
        if not problems:
            text = C.render(match, sk, verified)
            review = await run("checker", {"text": text, "verified": verified, "moment": moment})
            problems += [{"kind": "unsupported", "reason": u} for u in review.get("unsupported", []) or []]
            if not problems:
                return Commentary(moment["id"], sk, cl, verified, text, catches, attempt + 1)
        catches.append({"attempt": attempt + 1, "rejected_skeleton": sk, "problems": problems})
        feedback = [p["reason"] for p in problems]
    cl, sk = template_for(moment, match)
    results = C.verify_claims(match, cl, as_of)
    verified = {r["id"]: r["verified"] for r in results if r["ok"]}
    if len(verified) != len(cl) or C.check_skeleton(sk, verified):
        raise RuntimeError("template fallback failed verification; this is a bug")
    return Commentary(moment["id"], sk, cl, verified, C.render(match, sk, verified), catches, MAX_REWRITES + 1, True)


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
