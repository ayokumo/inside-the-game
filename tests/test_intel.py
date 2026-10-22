import pathlib
import re

import pytest

from intel import claims as C
from intel import store
from intel.data import Match, load_match, list_matches
from intel.moments import detect_moments
from intel.stats import team_stats, threshold_crossings, control_at

IDS = [m["match_id"] for m in list_matches()]
ROOT = pathlib.Path(__file__).resolve().parents[1]


def truncated(m, t):
    return Match([e for e in m.events if e["t"] <= t], m.meta)


@pytest.mark.parametrize("mid", IDS)
def test_no_lookahead_stats_and_moments(mid):
    m = load_match(mid)
    for t in (1500.0, 3300.0, 4800.0):
        cut = truncated(m, t)
        assert team_stats(m, t) == team_stats(cut, t)
        assert detect_moments(m, t) == detect_moments(cut, t)
        assert control_at(m, t)["label"] == control_at(cut, t)["label"]


@pytest.mark.parametrize("mid", IDS)
def test_moments_prefix_property(mid):
    m = load_match(mid)
    a, b = detect_moments(m, 2400.0), detect_moments(m, 5000.0)
    key = lambda x: (x["t"], x["kind"], x["team"])
    assert [key(x) for x in a] == [key(x) for x in b if x["t"] <= 2400.0]


@pytest.mark.parametrize("mid", IDS)
def test_threshold_monotonic(mid):
    m = load_match(mid)
    seen = set()
    for x in threshold_crossings(m, m.duration):
        k = (x["player_id"], x["threshold"], x["level"])
        assert k not in seen
        seen.add(k)


def _stat(m, **kw):
    base = {"id": "c1", "kind": "stat", "team": m.home, "metric": "shots", "from_t": 0, "to_t": 3000.0}
    base.update(kw)
    return base


def test_claim_accept_and_reject():
    m = load_match("SYN-0001")
    true = team_stats(m, 3000.0)[m.home]["shots"]
    assert C.verify_claim(m, _stat(m, value=true), 3000.0)["ok"]
    assert not C.verify_claim(m, _stat(m, value=true + 1), 3000.0)["ok"]
    assert not C.verify_claim(m, _stat(m, value=true, to_t=3600.0), 3000.0)["ok"]       # future
    assert not C.verify_claim(m, _stat(m, value=true, team="Nobody FC"), 3000.0)["ok"]
    assert not C.verify_claim(m, _stat(m, value=true, metric="bogus"), 3000.0)["ok"]
    assert not C.verify_claim(m, {"id": "x", "kind": "event", "event_id": "nope"}, 3000.0)["ok"]
    assert not C.verify_claim(m, {"id": "x", "kind": "wat"}, 3000.0)["ok"]
    assert not C.verify_claim(m, {"id": "x", "kind": "stat"}, 3000.0)["ok"]              # malformed


def test_event_claim_wrong_player_and_future():
    m = load_match("SYN-0001")
    goal = next(e for e in m.events if e["type"] == "shot" and e["outcome"] == "goal")
    ok = {"id": "g", "kind": "event", "event_id": goal["event_id"], "expect": {"type": "shot", "player_id": goal["player_id"]}}
    assert C.verify_claim(m, ok, goal["t"])["ok"]
    wrong = dict(ok, expect={"player_id": "someone-else"})
    assert not C.verify_claim(m, wrong, goal["t"])["ok"]
    assert not C.verify_claim(m, ok, goal["t"] - 5)["ok"]                                  # not happened yet


def test_duplicate_ids_rejected():
    m = load_match("SYN-0001")
    v = team_stats(m, 3000.0)[m.home]["shots"]
    r = C.verify_claims(m, [_stat(m, value=v), _stat(m, value=v)], 3000.0)
    assert r[0]["ok"] and not r[1]["ok"]


def test_skeleton_rules():
    v = {"c1": {"kind": "stat", "metric": "shots", "value": 5}}
    assert C.check_skeleton("They had {c1} shots", v) == []
    assert C.check_skeleton("They had 5 shots", v)
    assert C.check_skeleton("They had five shots", v)
    assert C.check_skeleton("{c2} shots", v)
    assert C.check_skeleton("{c1; DROP}", v)
    assert C.same_placeholders("a {c1} b {team:home}", "x {team:home} y {c1}")
    assert not C.same_placeholders("a {c1}", "a {c1} {c1}")


def test_render_locale_and_fact_stability():
    m = load_match("SYN-0001")
    t = 3000.0
    poss = team_stats(m, t)[m.home]["possession_pct"]
    r = C.verify_claims(m, [{"id": "c1", "kind": "stat", "team": m.home, "metric": "avg_pass_length_m",
                             "value": team_stats(m, t)[m.home]["avg_pass_length_m"], "from_t": 0, "to_t": t}], t)
    v = {x["id"]: x["verified"] for x in r}
    en, de = C.render(m, "{c1}", v, "en"), C.render(m, "{c1}", v, "de")
    assert "." in en and "," in de and en.replace(".", ",") == de
    assert re.sub(r"\D", "", en) == re.sub(r"\D", "", de)
    # rendering uses recomputed values even if a claim was nudged within tolerance
    assert f"{team_stats(m, t)[m.home]['avg_pass_length_m']:.1f}" in en


def test_truth_isolation():
    import ast
    for folder in ("intel", "mcp_server"):
        for p in (ROOT / folder).rglob("*.py"):
            tree = ast.parse(p.read_text())
            docs = set()
            for n in ast.walk(tree):
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body \
                        and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant):
                    docs.add(id(n.body[0].value))
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
                    assert "truth" not in n.value.lower(), f"{p}: string literal mentions truth"
                if isinstance(n, ast.Name) and "truth" in n.id.lower():
                    raise AssertionError(f"{p}: identifier {n.id}")
                if isinstance(n, ast.Attribute) and "truth" in n.attr.lower():
                    raise AssertionError(f"{p}: attribute {n.attr}")
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in n.names] + [getattr(n, "module", "") or ""]
                    assert not any("truth" in x for x in names)


def test_store_versioning(tmp_path):
    assert store.state_get("a", "k", tmp_path)["version"] == 0
    assert store.state_put("a", "k", {"x": 1}, 0, tmp_path)["ok"]
    assert not store.state_put("a", "k", {"x": 2}, 0, tmp_path)["ok"]
    assert store.state_put("a", "k", {"x": 3}, 1, tmp_path)["version"] == 2
    with pytest.raises(ValueError):
        store.state_put("../evil", "k", {}, None, tmp_path)
