"""Claims with receipts. The Narrator proposes structured claims, code verifies each one by
recomputing it from the event log (never from the model's memory), and the final sentence is
rebuilt by code from the verified values. Persona and translation agents only touch the
skeleton text, so they cannot change a fact.

Claim kinds:
  stat       {"id","kind":"stat","team"|"player_id","metric","value","from_t","to_t"}
  event      {"id","kind":"event","event_id","expect":{"type":..,"team":..,"player_id":..,"outcome":..}}
  state      {"id","kind":"state","label":"dominant|chaotic|balanced","team":..|None,"at_t"}
  threshold  {"id","kind":"threshold","player_id","threshold":"distance_km|top_speed_kmh","level"}
"""
from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter

from .data import Match
from .stats import CFG, control_at, player_stats, team_stats, threshold_crossings

# tolerance by metric class (absolute)
PCT = {"pass_accuracy_pct", "forward_pass_pct", "possession_pct", "field_tilt_pct"}
F1 = {"avg_pass_length_m", "max_ball_speed_kmh", "max_shot_speed_kmh", "top_speed_kmh", "distance_km"}
F2 = {"xg", "avg_pass_difficulty", "press_intensity"}
_TOL = {"int": 0.0, "pct": 0.5, "f1": 0.05, "f2": 0.005}


def metric_class(metric):
    if metric in PCT:
        return "pct"
    if metric in F1:
        return "f1"
    if metric in F2:
        return "f2"
    return "int"


def _tol(metric):
    return _TOL[metric_class(metric)]


def verify_claim(match: Match, claim: dict, as_of: float) -> dict:
    """Return {"id","ok","reason","verified"}; `verified` holds the recomputed canonical facts."""
    cid = claim.get("id", "?")

    def bad(reason):
        return {"id": cid, "ok": False, "reason": reason, "verified": None}

    kind = claim.get("kind")
    try:
        if kind == "stat":
            f, t = float(claim["from_t"]), float(claim["to_t"])
            if t > as_of + 1e-9:
                return bad("future window: to_t is after the current match time")
            if f < 0 or f >= t:
                return bad("bad window")
            metric = claim["metric"]
            if "player_id" in claim:
                if f != 0:
                    return bad("player stats are only available from kickoff")
                pid = claim["player_id"]
                if pid not in match.players:
                    return bad("unknown player")
                ps = player_stats(match, t).get(pid)
                if ps is None or metric not in ps or not isinstance(ps[metric], (int, float)):
                    return bad("unknown player metric")
                true = ps[metric]
                subject = {"player_id": pid, "player": match.players[pid]["name"], "team": match.players[pid]["team"]}
            else:
                team = claim["team"]
                if team not in (match.home, match.away):
                    return bad("unknown team")
                ts = team_stats(match, t, f)[team]
                if metric not in ts:
                    return bad("unknown team metric")
                true = ts[metric]
                subject = {"team": team}
            if abs(true - float(claim["value"])) > _tol(metric):
                return bad(f"value mismatch: claimed {claim['value']}, recomputed {round(true, 3)}")
            return {"id": cid, "ok": True, "reason": "", "verified": dict(subject, kind="stat", metric=metric, value=true, from_t=f, to_t=t)}

        if kind == "event":
            e = match.by_id.get(claim["event_id"])
            if e is None and str(claim["event_id"]).lstrip("-").isdigit():
                e = match.by_id.get(int(claim["event_id"]))
            if e is None:
                return bad("unknown event id")
            if e["t"] > as_of + 1e-9:
                return bad("event has not happened yet")
            for k, v in (claim.get("expect") or {}).items():
                if e.get(k) != v:
                    return bad(f"event field {k} is {e.get(k)!r}, not {v!r}")
            return {"id": cid, "ok": True, "reason": "", "verified": {
                "kind": "event", "event_id": e["event_id"], "t": e["t"], "clock": e["clock"], "type": e["type"],
                "team": e["team"], "player": e.get("player"), "player_id": e.get("player_id"),
                "outcome": e.get("outcome"), "score": e.get("score")}}

        if kind == "state":
            at = float(claim["at_t"])
            if at > as_of + 1e-9:
                return bad("future time")
            c = control_at(match, at)
            if c["label"] != claim["label"] or (claim.get("team") or None) != (c["team"] or None):
                return bad(f"control label at that time is {c['label']} ({c['team']})")
            i = max(0, bisect_right(match.ts, at) - 1)
            return {"id": cid, "ok": True, "reason": "", "verified": {"kind": "state", "label": c["label"], "team": c["team"], "at_t": at,
                                                                      "from_t": c["from_t"], "clock": match.events[i]["clock"]}}

        if kind == "threshold":
            for x in threshold_crossings(match, as_of):
                if (x["player_id"] == claim["player_id"] and x["threshold"] == claim["threshold"]
                        and x["level"] == claim["level"]):
                    return {"id": cid, "ok": True, "reason": "", "verified": dict(x, kind="threshold")}
            return bad("that threshold has not been crossed yet by that player")
    except (KeyError, TypeError, ValueError) as ex:
        return bad(f"malformed claim: {ex!r}")
    return bad("unknown claim kind")


def verify_claims(match: Match, claims: list, as_of: float) -> list:
    seen, out = set(), []
    for c in claims:
        r = verify_claim(match, c, as_of)
        if r["id"] in seen:
            r = {"id": r["id"], "ok": False, "reason": "duplicate claim id", "verified": None}
        seen.add(r["id"])
        out.append(r)
    return out


# ---------------------------------------------------------------- skeleton rules
PH = re.compile(r"\{[^{}]*\}")
NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
                "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
                "double triple hat-trick brace").split()
_NUM_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in NUMBER_WORDS) + r")\b", re.I)
_PH_OK = re.compile(r"^\{([A-Za-z_]\w*(\.(player|team|clock|score))?|player:[\w\-]+|team:(home|away))\}$")


def placeholders(text):
    return PH.findall(text)


_FIELDS = {"stat": {None, "team", "player"}, "threshold": {None, "player", "team", "clock"},
           "event": {"player", "team", "clock", "score"}, "state": {"team", "clock"}}


def _field_problem(v, field):
    allowed = _FIELDS.get(v.get("kind"), set())
    if field not in allowed:
        names = sorted(f"{{id.{a}}}" if a else "{id}" for a in allowed)
        return f"a {v.get('kind')} claim supports only {', '.join(names)}"
    if field and not v.get(field if field != "score" else "score"):
        return f"this claim has no {field}"
    return None


def check_skeleton(text: str, verified: dict) -> list:
    """Return a list of problems (empty means the skeleton is allowed). Numbers may only appear
    through placeholders that point at verified claims."""
    problems = []
    for ph in placeholders(text):
        if not _PH_OK.match(ph):
            problems.append(f"malformed placeholder {ph}; allowed forms are {{id}}, {{id.player}}, {{id.team}}, "
                            "{id.clock}, {id.score}, {player:PLAYER_ID}, {team:home}, {team:away}")
            continue
        m = re.match(r"^\{([A-Za-z_]\w*)(?:\.(\w+))?\}$", ph)
        if m and not ph.startswith(("{player:", "{team:")):
            if m.group(1) not in verified:
                problems.append(f"placeholder {ph} refers to a claim that is missing or failed verification")
            else:
                why = _field_problem(verified[m.group(1)], m.group(2))
                if why:
                    problems.append(f"placeholder {ph}: {why}")
    bare = PH.sub(" ", text)
    if re.search(r"\d", bare) or "%" in bare:
        problems.append("digits or % outside placeholders")
    w = _NUM_RE.search(bare)
    if w:
        problems.append(f"number word outside placeholders: {w.group(0)!r}")
    return problems


def same_placeholders(a: str, b: str) -> bool:
    """A translation must keep exactly the same placeholders."""
    return Counter(placeholders(a)) == Counter(placeholders(b))


# ---------------------------------------------------------------- rendering
LOCALES = {
    "en": {"dec": ".", "pct": "{v}%", "km": "{v} km", "kmh": "{v} km/h", "m": "{v} m"},
    "de": {"dec": ",", "pct": "{v} %", "km": "{v} km", "kmh": "{v} km/h", "m": "{v} m"},
    "fr": {"dec": ",", "pct": "{v} %", "km": "{v} km", "kmh": "{v} km/h", "m": "{v} m"},
    "es": {"dec": ",", "pct": "{v} %", "km": "{v} km", "kmh": "{v} km/h", "m": "{v} m"},
    "pt": {"dec": ",", "pct": "{v}%", "km": "{v} km", "kmh": "{v} km/h", "m": "{v} m"},
}


def _num(x, nd, loc):
    s = f"{x:.{nd}f}"
    return s.replace(".", LOCALES[loc]["dec"])


def format_value(metric, value, loc="en"):
    cls = metric_class(metric)
    L = LOCALES[loc]
    if cls == "int":
        return str(int(round(value)))
    if cls == "pct":
        return L["pct"].format(v=_num(value, 0, loc))
    if cls == "f2":
        return _num(value, 2, loc)
    nd = 1
    if metric == "distance_km":
        return L["km"].format(v=_num(value, 1, loc))
    if metric in ("top_speed_kmh", "max_ball_speed_kmh", "max_shot_speed_kmh"):
        return L["kmh"].format(v=_num(value, 1, loc))
    if metric == "avg_pass_length_m":
        return L["m"].format(v=_num(value, 1, loc))
    return _num(value, nd, loc)


def _sub_text(match, ph, verified, loc):
    if ph.startswith("player:"):
        return match.players[ph.split(":", 1)[1]]["name"]
    if ph.startswith("team:"):
        return match.home if ph.endswith("home") else match.away
    cid, _, field = ph.partition(".")
    v = verified[cid]
    if field == "player":
        return v["player"]
    if field == "team":
        return v["team"]
    if field == "clock":
        return f"{v['clock']}'"
    if field == "score":
        h, a = v["score"]
        return f"{h}:{a}"
    if v["kind"] == "stat":
        return format_value(v["metric"], v["value"], loc)
    if v["kind"] == "threshold":
        return format_value(v["threshold"], v["level"], loc)
    raise ValueError(f"claim {cid} of kind {v['kind']} has no printable value; use a field like .player")


def render_parts(match: Match, skeleton: str, verified: dict, loc: str = "en") -> list:
    """Final text as parts [{"text", "claim"}]; `claim` is the id of the verified claim a piece of text came from,
    so the UI can show a receipt for it. Raises if the skeleton breaks the rules."""
    problems = check_skeleton(skeleton, verified)
    if problems:
        raise ValueError("; ".join(problems))
    parts, pos = [], 0
    for m in PH.finditer(skeleton):
        if m.start() > pos:
            parts.append({"text": skeleton[pos:m.start()], "claim": None})
        ph = m.group(0)[1:-1]
        parts.append({"text": _sub_text(match, ph, verified, loc),
                      "claim": None if ph.startswith(("player:", "team:")) else ph.partition(".")[0]})
        pos = m.end()
    if pos < len(skeleton):
        parts.append({"text": skeleton[pos:], "claim": None})
    return parts


def render(match: Match, skeleton: str, verified: dict, loc: str = "en") -> str:
    """Rebuild final text from the skeleton and verified claims only. Raises if the skeleton breaks the rules."""
    return "".join(p["text"] for p in render_parts(match, skeleton, verified, loc))
