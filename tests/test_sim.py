"""Invariant tests: the simulator must never emit an impossible event."""
import statistics as st

import pytest

from matchsim import simulate
from matchsim.metrics import match_metrics
from matchsim.xg import xg_model

SEEDS = list(range(8))
ACTION_TYPES = {"pass", "shot", "tackle", "interception", "pressure", "foul", "clearance", "sprint",
                "possession_change", "throw_in", "goal_kick", "corner", "free_kick", "kickoff"}


@pytest.fixture(scope="module")
def matches():
    return [simulate("NBR", "HLW", s, f"SYN-T{s}") for s in SEEDS]


def test_ids_contiguous_and_time_monotonic(matches):
    for r in matches:
        ev = r["events"]
        assert [e["event_id"] for e in ev] == list(range(1, len(ev) + 1))
        ts = [e["t"] for e in ev]
        assert all(b >= a for a, b in zip(ts, ts[1:]))
        assert ev[-1]["type"] == "full_time"
        assert 5400 <= ev[-1]["t"] <= 6600


def test_coordinates_inside_pitch(matches):
    for r in matches:
        for e in r["events"]:
            for k in ("start", "end"):
                if e.get(k):
                    x, y = e[k]
                    assert 0 <= x <= 105 and 0 <= y <= 68, e


def test_score_matches_goals(matches):
    for r in matches:
        h = a = 0
        for e in r["events"]:
            if e["type"] == "shot" and e["outcome"] == "goal":
                if e["team"] == r["meta"]["home"]["name"]:
                    h += 1
                else:
                    a += 1
                assert e["score"] == [h, a]
        assert r["meta"]["final_score"] == {"home": h, "away": a}
        assert r["events"][-1]["score"] == [h, a]


def _intervals(r):
    """player_id -> (t_in, t_out) while on the pitch."""
    iv = {}
    for side in ("home", "away"):
        for p in r["meta"][side]["players"]:
            iv[p["player_id"]] = [0.0 if p["started"] else None, float("inf")]
    for e in r["events"]:
        if e["type"] == "substitution":
            iv[e["player_in_id"]][0] = e["t"]
            iv[e["player_out_id"]][1] = e["t"]
        elif e["type"] == "card" and e["card_type"] != "yellow":
            iv[e["player_id"]][1] = e["t"]
    return iv


def test_players_only_act_while_on_pitch(matches):
    for r in matches:
        iv = _intervals(r)
        for e in r["events"]:
            if e["type"] in ACTION_TYPES and e["player_id"]:
                t_in, t_out = iv[e["player_id"]]
                assert t_in is not None and t_in - 1e-6 <= e["t"] <= t_out + 1e-6, e
            if e["type"] == "pass":
                t_in, t_out = iv[e["recipient_id"]]
                assert t_in is not None and t_in - 1e-6 <= e["t"] <= t_out + 1e-6, e


def test_passes_stay_within_team_and_next_actor_is_recipient(matches):
    for r in matches:
        by_team = {r["meta"][s]["name"]: {p["player_id"] for p in r["meta"][s]["players"]} for s in ("home", "away")}
        ev = r["events"]
        for i, e in enumerate(ev):
            if e["type"] != "pass":
                continue
            assert e["player_id"] in by_team[e["team"]] and e["recipient_id"] in by_team[e["team"]]
            assert e["player_id"] != e["recipient_id"]
            assert 0.0 <= e["difficulty"] <= 1.0
            if e["outcome"] != "complete":
                continue
            for nxt in ev[i + 1:]:
                if nxt["possession_id"] != e["possession_id"]:
                    break
                if nxt["team"] == e["team"] and nxt["type"] in ("pass", "shot"):
                    assert nxt["player_id"] == e["recipient_id"], (e, nxt)
                    break


def test_shots_and_xg(matches):
    for r in matches:
        for e in r["events"]:
            if e["type"] == "shot":
                assert 0 < e["xg"] < 1
                assert e["outcome"] in ("goal", "saved", "blocked", "off_target", "post")
                assert 30 <= e["shot_speed_kmh"] <= 140
    assert xg_model(94, 34) > xg_model(80, 34) > xg_model(70, 34)
    assert xg_model(94, 34) > xg_model(94, 60)
    assert 0.08 < xg_model(94, 34) < 0.2
    assert xg_model(104, 34) <= 0.65


def test_physical_distance_monotone_and_plausible(matches):
    for r in matches:
        last = {}
        for e in r["events"]:
            if e["type"] == "physical":
                assert e["distance_km"] >= last.get(e["player_id"], 0) - 1e-9
                last[e["player_id"]] = e["distance_km"]
        outfield = [v for k, v in last.items() if not k.endswith("01")]
        assert 3.0 < max(outfield) < 14.0


def test_no_latent_state_leaks_into_events(matches):
    banned = ("fatigue", "mode", "window", "stamina", "tactical", "truth")
    for r in matches:
        for e in r["events"]:
            for k in e:
                assert not any(b in k for b in banned), k


def test_truth_log_is_consistent(matches):
    for r in matches:
        tr = r["truth"]
        for w in tr["tactical_windows"]:
            assert w["end_t"] is not None and w["start_t"] < w["end_t"]
            assert {"press_intensity_in", "shots_per_min_in", "opp_pass_completion_in"} <= set(w["measured"])
        assert len(tr["substitutions"]) <= 10
        for s in tr["substitutions"]:
            assert 0.0 <= s["out_fatigue"] <= 1.0


def test_deterministic_given_seed():
    a = simulate("NBR", "HLW", 5, "SYN-T")
    b = simulate("NBR", "HLW", 5, "SYN-T")
    c = simulate("NBR", "HLW", 6, "SYN-T")
    assert a["events"] == b["events"]
    assert a["events"] != c["events"]


def test_script_forces_red_card_and_mode():
    script = [{"at_min": 30, "type": "red_card", "team": "home"},
              {"at_min": 60, "type": "mode", "team": "away", "mode": "high_press", "duration_min": 15}]
    r = simulate("NBR", "HLW", 3, "SYN-T", script)
    reds = r["truth"]["red_cards"]
    assert reds and reds[0]["team"] == "Northbridge FC" and reds[0]["t"] >= 30 * 60
    assert any(w["reason"] == "script" and w["mode"] == "high_press" for w in r["truth"]["tactical_windows"])


def test_aggregate_realism_sanity_bands():
    """Loose sanity bands, NOT proof of realism. The real comparison is
    tools/wyscout_benchmark.py against the public Wyscout data."""
    ms = [match_metrics(*(lambda r: (r["events"], r["meta"]))(simulate("NBR", "HLW", s, "SYN-T")))["total"]
          for s in range(100, 130)]
    assert 750 <= st.mean(m["passes"] for m in ms) <= 1150
    assert 0.76 <= st.mean(m["completion"] for m in ms) <= 0.88
    assert 2.0 <= st.mean(m["goals"] for m in ms) <= 3.4
    assert 17 <= st.mean(m["shots"] for m in ms) <= 30
    assert abs(st.mean(m["goals"] for m in ms) - st.mean(m["xg"] for m in ms)) < 0.5
    assert 0.44 <= st.mean(m["possession_home"] for m in ms) <= 0.58


def test_matches_differ_in_tempo():
    """Real matches vary a lot in pass volume. A simulator whose matches all look alike is not realistic."""
    ps = [match_metrics(*(lambda r: (r["events"], r["meta"]))(simulate("NBR", "HLW", s, "SYN-T")))["total"]["passes"]
          for s in range(200, 230)]
    assert st.pstdev(ps) > 60
