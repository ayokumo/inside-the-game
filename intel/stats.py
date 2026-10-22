"""Deterministic match statistics. Math lives here, words come from agents.

Every function takes an `as_of` time in seconds of playing time and only ever
looks at events with t <= as_of, so no query can see the future of the replay.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .data import Match

CFG = json.loads((Path(__file__).parent / "config.json").read_text())
OWNER_TYPES = {"pass", "shot", "throw_in", "goal_kick", "corner", "free_kick", "kickoff", "possession_change"}


def possession_spans(match: Match, t0: float, t1: float):
    """(start, end, team) spans of possession clipped to [t0, t1]. A span ends when the
    next possession starts, or a few seconds after its last event (so goal celebrations
    and dead time are not credited to anyone)."""
    evs = match.window(0.0, t1)
    starts, last = [], {}
    seen = set()
    for e in evs:
        if e["type"] == "physical":
            continue
        last[e["possession_id"]] = e["t"]
        if e["type"] in OWNER_TYPES and e["possession_id"] not in seen:
            seen.add(e["possession_id"])
            starts.append((e["t"], e["team"], e["possession_id"]))
    spans = []
    for i, (t, team, pid) in enumerate(starts):
        nxt = starts[i + 1][0] if i + 1 < len(starts) else t1
        end = min(nxt, last.get(pid, t) + CFG["possession_tail_s"], t1)
        a, b = max(t, t0), min(end, t1)
        if b > a:
            spans.append((a, b, team))
    return spans


def _blank():
    return dict(passes=0, completed=0, length_sum=0.0, diff_sum=0.0, forward=0, shots=0, on_target=0, goals=0,
                xg=0.0, big_chances=0, pressures=0, tackles=0, tackles_won=0, interceptions=0, fouls=0,
                yellows=0, reds=0, corners=0, final_third_passes=0, max_ball_speed=0.0, max_shot_speed=0.0,
                sprints=0, turnovers=0)


def team_stats(match: Match, as_of: float, t0: float = 0.0):
    """Per team statistics for events in [t0, as_of]. Returns {team_name: {...}}."""
    evs = match.window(t0, as_of)
    s = {match.home: _blank(), match.away: _blank()}
    big = CFG["thresholds"]["big_chance_xg"]
    for e in evs:
        tm = e["team"]
        if tm not in s:
            continue
        d, ty = s[tm], e["type"]
        if ty == "pass":
            d["passes"] += 1
            d["length_sum"] += e["length_m"]
            d["diff_sum"] += e["difficulty"]
            d["forward"] += e["forward_m"] > 5
            d["max_ball_speed"] = max(d["max_ball_speed"], e["ball_speed_kmh"])
            if e["outcome"] == "complete":
                d["completed"] += 1
                x = e["end"][0] if match.attacks_positive(tm, e["period"]) else 105.0 - e["end"][0]
                d["final_third_passes"] += x >= CFG["final_third_x"]
        elif ty == "shot":
            d["shots"] += 1
            d["on_target"] += e["outcome"] in ("goal", "saved")
            d["goals"] += e["outcome"] == "goal"
            d["xg"] += e["xg"]
            d["big_chances"] += e["xg"] >= big
            d["max_shot_speed"] = max(d["max_shot_speed"], e["shot_speed_kmh"])
        elif ty == "pressure":
            d["pressures"] += 1
        elif ty == "tackle":
            d["tackles"] += 1
            d["tackles_won"] += e["outcome"] == "won"
        elif ty == "interception":
            d["interceptions"] += 1
        elif ty == "foul":
            d["fouls"] += 1
        elif ty == "card":
            if e["card_type"] == "yellow":
                d["yellows"] += 1
            else:
                d["reds"] += 1
        elif ty == "corner":
            d["corners"] += 1
        elif ty == "sprint":
            d["sprints"] += 1
        elif ty == "possession_change":
            d["turnovers"] += 1
    spans = possession_spans(match, t0, as_of)
    poss = {match.home: 0.0, match.away: 0.0}
    for a, b, team in spans:
        poss[team] += b - a
    tot_poss = sum(poss.values()) or 1.0
    tot_tilt = sum(d["final_third_passes"] for d in s.values()) or 0
    out = {}
    for team, d in s.items():
        o = match.other(team)
        opp_actions = s[o]["passes"] + s[o]["shots"]
        out[team] = {
            "passes": d["passes"], "completed_passes": d["completed"],
            "pass_accuracy_pct": 100.0 * d["completed"] / d["passes"] if d["passes"] else 0.0,
            "avg_pass_length_m": d["length_sum"] / d["passes"] if d["passes"] else 0.0,
            "avg_pass_difficulty": d["diff_sum"] / d["passes"] if d["passes"] else 0.0,
            "forward_pass_pct": 100.0 * d["forward"] / d["passes"] if d["passes"] else 0.0,
            "shots": d["shots"], "shots_on_target": d["on_target"], "goals": d["goals"],
            "xg": d["xg"], "big_chances": d["big_chances"],
            "pressures": d["pressures"],
            "press_intensity": d["pressures"] / opp_actions if opp_actions else 0.0,
            "tackles": d["tackles"], "tackles_won": d["tackles_won"], "interceptions": d["interceptions"],
            "fouls": d["fouls"], "yellow_cards": d["yellows"], "red_cards": d["reds"], "corners": d["corners"],
            "possession_pct": 100.0 * poss[team] / tot_poss,
            "field_tilt_pct": 100.0 * d["final_third_passes"] / tot_tilt if tot_tilt else 50.0,
            "max_ball_speed_kmh": d["max_ball_speed"], "max_shot_speed_kmh": d["max_shot_speed"],
            "sprints": d["sprints"], "possession_changes_won": d["turnovers"],
        }
    return out


def player_stats(match: Match, as_of: float):
    """Per player statistics from kickoff to as_of. Returns {player_id: {...}}."""
    out = {}

    def get(pid):
        if pid not in out:
            info = match.players[pid]
            out[pid] = dict(name=info["name"], team=info["team"], passes=0, completed_passes=0, diff_sum=0.0,
                            shots=0, goals=0, xg=0.0, assists=0, tackles_won=0, interceptions=0, fouls=0,
                            cards=0, sprints=0, top_speed_kmh=0.0, distance_km=0.0)
        return out[pid]

    for e in match.window(0.0, as_of):
        pid, ty = e["player_id"], e["type"]
        if not pid:
            continue
        if ty == "pass":
            d = get(pid)
            d["passes"] += 1
            d["diff_sum"] += e["difficulty"]
            d["completed_passes"] += e["outcome"] == "complete"
        elif ty == "shot":
            d = get(pid)
            d["shots"] += 1
            d["goals"] += e["outcome"] == "goal"
            d["xg"] += e["xg"]
            if e["assist_id"]:
                get(e["assist_id"])["assists"] += e["outcome"] == "goal"
        elif ty == "tackle" and e["outcome"] == "won":
            get(pid)["tackles_won"] += 1
        elif ty == "interception":
            get(pid)["interceptions"] += 1
        elif ty == "foul":
            get(pid)["fouls"] += 1
        elif ty == "card":
            get(pid)["cards"] += 1
        elif ty == "sprint":
            d = get(pid)
            d["sprints"] += 1
            d["top_speed_kmh"] = max(d["top_speed_kmh"], e["speed_kmh"])
        elif ty == "physical":
            d = get(pid)
            d["distance_km"] = e["distance_km"]
            d["top_speed_kmh"] = max(d["top_speed_kmh"], e["top_speed_kmh"])
    for d in out.values():
        d["pass_accuracy_pct"] = 100.0 * d["completed_passes"] / d["passes"] if d["passes"] else 0.0
        d["avg_pass_difficulty"] = d["diff_sum"] / d["passes"] if d["passes"] else 0.0
        del d["diff_sum"]
    return out


def threshold_crossings(match: Match, as_of: float):
    """Moments a player crossed a configured distance or speed threshold, in time order."""
    th = CFG["thresholds"]
    dist_last, top_seen, out = {}, {}, []
    for e in match.window(0.0, as_of):
        pid = e["player_id"]
        if e["type"] == "physical":
            prev = dist_last.get(pid, 0.0)
            for lvl in th["distance_km"]:
                if prev < lvl <= e["distance_km"]:
                    out.append({"t": e["t"], "clock": e["clock"], "player_id": pid, "player": e["player"],
                                "team": e["team"], "threshold": "distance_km", "level": lvl,
                                "value": e["distance_km"], "event_id": e["event_id"]})
            dist_last[pid] = e["distance_km"]
        if e["type"] in ("sprint", "physical"):
            v = e["speed_kmh"] if e["type"] == "sprint" else e["top_speed_kmh"]
            prev = top_seen.get(pid, 0.0)
            for lvl in th["top_speed_kmh"]:
                if prev < lvl <= v:
                    out.append({"t": e["t"], "clock": e["clock"], "player_id": pid, "player": e["player"],
                                "team": e["team"], "threshold": "top_speed_kmh", "level": lvl, "value": v,
                                "event_id": e["event_id"]})
            top_seen[pid] = max(prev, v)
    return out


def window_features(match: Match, t0: float, t1: float):
    """Rolling window picture used to explain control versus chaos."""
    ts = team_stats(match, t1, t0)
    mins = max((t1 - t0) / 60.0, 1e-9)
    turnovers = sum(d["possession_changes_won"] for d in ts.values())
    return {"from_t": t0, "to_t": t1, "teams": ts, "turnovers_per_min": turnovers / mins}


def control_label(feat, baseline_turnovers=None):
    """Rule based label for a window: one team dominant, the game chaotic, or balanced.
    Chaos is relative to this match's own turnover rate so far (baseline_turnovers),
    not an absolute number. Thresholds live in config.json so they are visible and tunable."""
    c = CFG["control"]
    for team, d in feat["teams"].items():
        if d["possession_pct"] >= c["dominant_possession_pct"] and d["field_tilt_pct"] >= c["dominant_tilt_pct"]:
            return {"label": "dominant", "team": team}
    if baseline_turnovers and feat["turnovers_per_min"] >= c["chaotic_ratio_to_match_baseline"] * baseline_turnovers:
        return {"label": "chaotic", "team": None}
    return {"label": "balanced", "team": None}


def control_at(match: Match, as_of: float):
    """Control label for the window ending at as_of, using only data up to as_of."""
    w = CFG["control"]["window_s"]
    feat = window_features(match, max(0.0, as_of - w), as_of)
    base = None
    if as_of >= CFG["control"]["min_baseline_s"]:
        base = window_features(match, 0.0, as_of)["turnovers_per_min"]
    return dict(control_label(feat, base), from_t=feat["from_t"], to_t=as_of, features=feat)


def poisson_upper_tail(n, lam):
    """P(X >= n) for X ~ Poisson(lam). Used to ask: is this pressure count unusual for the baseline rate?"""
    if lam <= 0:
        return 1.0
    s, term = 0.0, math.exp(-lam)
    for k in range(n):
        s += term
        term *= lam / (k + 1)
    return max(0.0, 1.0 - s)


def press_height(match: Match, team: str, t0: float, t1: float):
    """(mean attacking frame x in metres of the team's pressures, count). Higher means pressing further up the pitch."""
    xs = []
    for e in match.window(t0, t1):
        if e["type"] == "pressure" and e["team"] == team:
            x = e["start"][0]
            xs.append(x if match.attacks_positive(team, e["period"]) else 105.0 - x)
    return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)
