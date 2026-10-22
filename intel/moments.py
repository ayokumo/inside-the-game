"""Key moment detection and scoring. Fully deterministic and causal: a moment at time t
is detected using only events up to t, so moments(as_of=T1) is always a prefix of
moments(as_of=T2) for T1 < T2. The Analyst agent explains these moments, it does not find them.
"""
from __future__ import annotations

from .data import Match
from .stats import CFG, control_at, poisson_upper_tail, press_height, window_features

LATE_S = 75 * 60


def detect_moments(match: Match, as_of: float):
    evs = match.window(0.0, as_of)
    moments = []

    def add(t, kind, importance, team=None, player=None, facts=None, evidence=(), clock=None):
        moments.append({"t": t, "clock": clock or "", "kind": kind, "importance": importance, "team": team,
                        "player_id": player["player_id"] if player else None,
                        "player": player["player"] if player else None,
                        "facts": facts or {}, "evidence_event_ids": list(evidence)})

    # ---- event driven moments: goals, red cards, big chances
    for e in evs:
        if e["type"] == "shot" and e["outcome"] == "goal":
            h, a = e["score"]
            scorer_home = e["team"] == match.home
            own, opp = (h, a) if scorer_home else (a, h)
            before_own, before_opp = own - 1, opp
            imp = 10.0
            tag = "extends_lead"
            if before_own == before_opp:
                tag, imp = ("go_ahead", imp + 3) if own > opp else ("equaliser", imp + 4)
            elif before_own < before_opp:
                tag = "equaliser" if own == opp else "pulls_one_back"
                imp += 4 if own == opp else 2
            if e["t"] >= LATE_S:
                imp += 2
            add(e["t"], "goal", imp, e["team"], e, {"score": f"{h}:{a}", "tag": tag, "xg": e["xg"],
                "assist": e["assist"], "body_part": e["body_part"], "shot_type": e["shot_type"]},
                [e["event_id"]], e["clock"])
        elif e["type"] == "card" and e["card_type"] != "yellow":
            imp = 8.0 + (2.0 if e["t"] < 3600 else 0.0)
            add(e["t"], "red_card", imp, e["team"], e, {"card_type": e["card_type"]}, [e["event_id"]], e["clock"])
        elif e["type"] == "shot" and e["xg"] >= CFG["thresholds"]["big_chance_xg"]:
            add(e["t"], "big_chance_missed", 4.0, e["team"], e,
                {"xg": e["xg"], "outcome": e["outcome"]}, [e["event_id"]], e["clock"])

    # ---- window driven moments, checked each minute using only past data
    ps = CFG["press_spike"]
    last_spike, last_ctrl, prev_label = {}, -1e9, {"label": "balanced", "team": None}
    spikes = []
    minute = 10
    while minute * 60 <= as_of:
        t = minute * 60.0
        if t - ps["window_s"] >= 300:
            rec = window_features(match, t - ps["window_s"], t)["teams"]
            base = window_features(match, 0.0, t - ps["window_s"])["teams"]
            for team in (match.home, match.away):
                o = match.other(team)
                opp_actions = rec[o]["passes"] + rec[o]["shots"]
                b = base[team]["press_intensity"]
                cur = rec[team]["press_intensity"]
                hr, nr = press_height(match, team, t - ps["window_s"], t)
                hb, _ = press_height(match, team, 0.0, t - ps["window_s"])
                gain = (hr - hb) if hr is not None and hb is not None and nr >= ps["min_pressures_for_height"] else None
                if (opp_actions >= ps["min_opp_actions"] and b > 0.01 and cur / b >= ps["ratio"]
                        and poisson_upper_tail(rec[team]["pressures"], b * opp_actions) <= ps["alpha"]
                        and gain is not None and gain >= ps["min_height_gain_m"]
                        and t - last_spike.get(team, -1e9) >= ps["debounce_s"]):
                    last_spike[team] = t
                    spike = {"team": team, "from_t": t - ps["window_s"], "detected_t": t,
                             "ratio": round(cur / b, 2)}
                    spikes.append(spike)
                    add(t, "press_spike", 5.0, team, None,
                        {"press_intensity_recent": round(rec[team]["press_intensity"], 3),
                         "press_intensity_before": round(b, 3), "ratio": spike["ratio"],
                         "press_height_gain_m": round(gain, 1), "window_from_t": spike["from_t"]}, [], _clock_of(match, t))
        c = control_at(match, t)
        key = (c["label"], c["team"])
        if key != (prev_label["label"], prev_label["team"]) and c["label"] != "balanced" and t - last_ctrl >= 600:
            last_ctrl = t
            add(t, "control_shift", 3.0, c["team"], None,
                {"label": c["label"], "possession_pct": round(c["features"]["teams"][c["team"]]["possession_pct"], 1)
                 if c["team"] else None, "turnovers_per_min": round(c["features"]["turnovers_per_min"], 2)},
                [], _clock_of(match, t))
        prev_label = c
        minute += 1

    # ---- attach context drivers to event moments (only things that happened before them)
    reds = [m for m in moments if m["kind"] == "red_card"]
    subs = [e for e in evs if e["type"] == "substitution"]
    for m in moments:
        ctx = []
        if m["kind"] in ("goal", "big_chance_missed", "red_card"):
            for r in reds:
                if r is not m and r["t"] < m["t"] and m["t"] - r["t"] <= 1500:
                    ctx.append({"kind": "red_card_earlier", "team": r["team"], "minutes_ago": round((m["t"] - r["t"]) / 60)})
            for sp in spikes:
                if sp["detected_t"] <= m["t"] and m["t"] - sp["detected_t"] <= 900:
                    ctx.append({"kind": "press_spike_active", "team": sp["team"], "ratio": sp["ratio"]})
            if m["player_id"]:
                for s in subs:
                    if s["player_in_id"] == m["player_id"] and s["t"] < m["t"]:
                        ctx.append({"kind": "scorer_was_substitute", "minutes_on_pitch": round((m["t"] - s["t"]) / 60)})
        m["context"] = ctx

    moments.sort(key=lambda m: (m["t"], -m["importance"]))
    for i, m in enumerate(moments, start=1):
        m["id"] = f"{match.match_id}:M{i}"
    return moments


def _clock_of(match, t):
    from bisect import bisect_right
    i = max(0, bisect_right(match.ts, t) - 1)
    return match.events[i]["clock"]


def top_moments(match: Match, as_of: float, k: int = 5, since_t: float = 0.0):
    ms = [m for m in detect_moments(match, as_of) if m["t"] >= since_t]
    return sorted(ms, key=lambda m: -m["importance"])[:k]
