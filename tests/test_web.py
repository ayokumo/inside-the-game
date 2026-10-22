import asyncio

import pytest
from starlette.testclient import TestClient

from web import app as webapp
from web.service import Service


@pytest.fixture()
def client(tmp_path):
    webapp.svc = Service(cache_dir=tmp_path)
    return TestClient(webapp.app)


def test_matches_and_state(client):
    ms = client.get("/api/matches").json()
    assert len(ms) >= 3 and ms[0]["home_players"]
    s = client.get(f"/api/match/{ms[0]['match_id']}/state", params={"as_of": 3300}).json()
    assert set(s["stats"]) == {"home", "away"} and all(x["t"] <= 3300 for x in s["moments"])


def test_state_cannot_see_the_future(client):
    a = client.get("/api/match/SYN-0001/state", params={"as_of": 1000}).json()
    b = client.get("/api/match/SYN-0001/state", params={"as_of": 99999}).json()
    assert b["as_of"] <= 5700 and len(b["moments"]) >= len(a["moments"])
    assert client.get("/api/match/SYN-0001/state", params={"as_of": "x"}).status_code == 400
    assert client.get("/api/match/NOPE/state").status_code == 404


def test_future_moment_refused(client):
    s = client.get("/api/match/SYN-0001/state", params={"as_of": 99999}).json()
    late = s["moments"][-1]
    r = client.get(f"/api/match/SYN-0001/moment/{late['id']}", params={"as_of": late["t"] - 10})
    assert r.status_code == 404


def test_offline_view_has_receipts_and_no_invented_numbers(client):
    s = client.get("/api/match/SYN-0001/state", params={"as_of": 99999}).json()
    goal = next(x for x in s["moments"] if x["kind"] == "goal")
    v = client.get(f"/api/match/SYN-0001/moment/{goal['id']}", params={"as_of": goal["t"] + 1, "persona": "analyst"}).json()
    assert v["source"].startswith("template") and v["used_fallback"] and v["text"]
    assert any(p["claim"] for p in v["parts"]) and v["receipts"]
    r = client.get(f"/api/match/SYN-0001/moment/{goal['id']}", params={"as_of": goal["t"] + 1, "locale": "de"}).json()
    assert r["locale"] == "en" and r["notes"]


def test_bad_params_rejected(client):
    base = "/api/match/SYN-0001/moment/SYN-0001:M1"
    assert client.get(base, params={"as_of": 5000, "persona": "hacker"}).status_code in (400, 404)
    assert client.get(base, params={"as_of": 5000, "favorite_team": "Nobody"}).status_code in (400, 404)
    assert client.get("/api/match/SYN-0001/events").json()["events"]


def test_pitch_events_endpoint_compact(client):
    d = client.get("/api/match/SYN-0001/events").json()
    assert d["events"][0][2] in ("pass", "shot") and d["home"] == "Northbridge FC"
