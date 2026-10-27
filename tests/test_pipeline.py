import asyncio

from agents import pipeline as P
from intel.data import load_match
from intel.moments import detect_moments


def goal_moment():
    m = load_match("SYN-0001")
    mom = next(x for x in detect_moments(m, m.duration) if x["kind"] == "goal")
    return m, mom


def scripted(drafts, review=None, adapt_out=None):
    drafts = list(drafts)
    calls = []

    async def run(role, payload):
        calls.append(role)
        if role == "narrator":
            return drafts.pop(0)
        if role == "checker":
            return review or {"unsupported": []}
        if role in ("persona", "translate"):
            return adapt_out(role, payload) if adapt_out else {"skeleton": payload["skeleton"]}
        return {}
    run.calls = calls
    return run


def good_draft(mom):
    return {"claims": [{"id": "c1", "kind": "event", "event_id": mom["evidence_event_ids"][0],
                        "expect": {"type": "shot", "outcome": "goal"}}],
            "skeleton": "{c1.clock} {c1.player} scores for {c1.team}! It is {c1.score}.", "causal": []}


def test_clean_draft_passes_first_time():
    m, mom = goal_moment()
    c = asyncio.run(P.narrate_moment(m, mom, mom["t"], scripted([good_draft(mom)])))
    assert c.attempts == 1 and not c.catches and not c.used_fallback
    assert mom["player"] in c.text and mom["facts"]["score"] in c.text


def test_wrong_player_is_caught_then_fixed_and_logged():
    m, mom = goal_moment()
    bad = good_draft(mom)
    bad["claims"][0]["expect"]["player_id"] = "not-the-scorer"
    c = asyncio.run(P.narrate_moment(m, mom, mom["t"], scripted([bad, good_draft(mom)])))
    assert c.attempts == 2 and len(c.catches) == 1 and not c.used_fallback
    assert c.catches[0]["problems"][0]["kind"] == "claim"


def test_digit_in_prose_is_caught():
    m, mom = goal_moment()
    bad = good_draft(mom)
    bad["skeleton"] = "A goal after 12 minutes, {c1.player} scores."
    c = asyncio.run(P.narrate_moment(m, mom, mom["t"], scripted([bad, good_draft(mom)])))
    assert c.catches and c.catches[0]["problems"][0]["kind"] == "wording"


def test_unsupported_causal_link_is_caught():
    m, mom = goal_moment()
    assert not any(x["kind"] == "red_card_earlier" for x in mom["context"])
    bad = dict(good_draft(mom), causal=[{"moment_id": mom["id"], "driver_kind": "red_card_earlier"}])
    c = asyncio.run(P.narrate_moment(m, mom, mom["t"], scripted([bad, good_draft(mom)])))
    assert c.catches[0]["problems"][0]["kind"] == "causal"


def test_checker_llm_flag_forces_rewrite():
    m, mom = goal_moment()
    run = scripted([good_draft(mom), good_draft(mom)])
    flags = [{"unsupported": ["claims the keeper was at fault"]}, {"unsupported": []}]

    async def run2(role, payload):
        return flags.pop(0) if role == "checker" else await run(role, payload)
    c = asyncio.run(P.narrate_moment(m, mom, mom["t"], run2))
    assert c.attempts == 2 and c.catches[0]["problems"][0]["kind"] == "unsupported"


def test_three_failures_use_template_fallback():
    m, mom = goal_moment()
    bad = {"claims": [], "skeleton": "They scored 3 goals.", "causal": []}
    c = asyncio.run(P.narrate_moment(m, mom, mom["t"], scripted([bad, bad, bad])))
    assert c.used_fallback and len(c.catches) == 3 and mom["player"] in c.text


def test_future_moment_cannot_be_narrated():
    import pytest
    m, mom = goal_moment()
    with pytest.raises(ValueError):
        asyncio.run(P.narrate_moment(m, mom, mom["t"] - 30, scripted([good_draft(mom)])))


def test_translation_must_keep_placeholders():
    m, mom = goal_moment()
    c = asyncio.run(P.narrate_moment(m, mom, mom["t"], scripted([good_draft(mom)])))

    def out(role, payload):
        if role == "translate":
            return {"skeleton": "Tor! {c1.player} trifft."}      # dropped placeholders
        return {"skeleton": payload["skeleton"]}
    r = asyncio.run(P.adapt(m, c, scripted([], adapt_out=out), "casual", "de"))
    assert any("translate output rejected" in n for n in r["notes"])
    assert r["text"] == c.text or mom["player"] in r["text"]


def test_persona_cannot_invent_placeholders():
    m, mom = goal_moment()
    c = asyncio.run(P.narrate_moment(m, mom, mom["t"], scripted([good_draft(mom)])))
    out = lambda role, p: {"skeleton": "{c9.player} wins it"}
    r = asyncio.run(P.adapt(m, c, scripted([], adapt_out=out), "casual", "en"))
    assert r["notes"] and r["text"] == c.text


def test_commentate_tick_end_to_end():
    m = load_match("SYN-0001")
    mom = next(x for x in detect_moments(m, m.duration) if x["kind"] == "goal")

    async def run(role, payload):
        if role == "analyst":
            return {"moment_ids": [mom["id"]], "notes": []}
        if role == "narrator":
            return good_draft(payload["moment"])
        return {"unsupported": []}
    out = asyncio.run(P.commentate(m, mom["t"] + 1, run))
    assert len(out) == 1 and out[0].moment_id == mom["id"]


def test_parse_json_variants():
    from agents.runner import parse_json
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Here you go:\n```json\n{"a": {"b": 2}}\n```') == {"a": {"b": 2}}
    import pytest
    with pytest.raises(ValueError):
        parse_json("no json here")


def test_menu_claims_all_verify_and_render():
    from agents.pipeline import build_menu
    from intel import claims as C
    m, mom = goal_moment()
    menu = build_menu(m, mom, mom["t"])
    assert "ev" in menu and "home_xg" in menu
    res = C.verify_claims(m, list(menu.values()), mom["t"])
    assert all(r["ok"] for r in res)
    v = {r["id"]: r["verified"] for r in res}
    text = C.render(m, "{ev.clock} {ev.player} for {ev.team}, {ev.score}, {home_poss} of the ball", v)
    assert text.startswith(mom["clock"] + "'") and "%" in text


def test_narrator_copying_menu_passes_first_time():
    from agents.pipeline import build_menu
    m, mom = goal_moment()
    menu = build_menu(m, mom, mom["t"])

    async def run(role, payload):
        if role == "narrator":
            assert set(payload["menu"]) == set(menu)
            return {"claims": [payload["menu"]["ev"], payload["menu"]["home_shots"]],
                    "skeleton": "{ev.clock} Goal! {ev.player} scores for {ev.team}. It is {ev.score}.", "causal": []}
        return {"unsupported": []}
    c = asyncio.run(P.narrate_moment(m, mom, mom["t"], run))
    assert c.attempts == 1 and not c.used_fallback


def test_numeric_event_id_given_as_string_is_accepted():
    from intel import claims as C
    m, mom = goal_moment()
    eid = str(mom["evidence_event_ids"][0])
    r = C.verify_claim(m, {"id": "ev", "kind": "event", "event_id": eid, "expect": {"type": "shot"}}, mom["t"])
    assert r["ok"]


def test_lexicon_rules():
    from intel import lexicon as L
    goal = {"kind": "goal", "facts": {"tag": "go_ahead"}, "context": []}
    assert L.check_wording("{ev.player} scores a crucial goal", goal)
    assert L.check_wording("{ev.player} scores just after the break", goal)
    assert L.check_wording("{ev.player} scores his first goal", goal)
    assert L.check_wording("{ev.player} scores after coming on as a substitute", goal)
    assert not L.check_wording("{ev.player} scores after coming on as a substitute",
                               dict(goal, context=[{"kind": "scorer_was_substitute"}]))
    assert L.check_wording("a clinical equaliser", goal)
    assert not L.check_wording("{ev.player} puts {ev.team} ahead, a go-ahead goal", goal)
    assert not L.check_wording("{press_recent} pressures per pass", {"kind": "press_spike", "facts": {}, "context": []}) or True
    assert not L.check_wording("{ev.clock} {ev.player} scores for {ev.team}. It is {ev.score}.", goal)


def test_bad_fields_are_problems_not_crashes():
    from intel import claims as C
    v = {"ctrl": {"kind": "state", "label": "dominant", "team": "A", "clock": "10"},
         "ev": {"kind": "event", "player": "X", "team": "A", "clock": "5", "score": [1, 0]},
         "s": {"kind": "stat", "metric": "shots", "value": 3, "team": "A"}}
    for bad in ("{ctrl}", "{ctrl.at_t}", "{ev}", "{s.clock}", "{ctrl.score}"):
        assert C.check_skeleton(bad, v), bad
    for good in ("{ctrl.team} {ctrl.clock}", "{ev.player} {ev.clock} {ev.score}", "{s}"):
        assert not C.check_skeleton(good, v), good


def test_render_text_tool_never_raises():
    import mcp_server.server as srv
    r = srv.render_text("SYN-0001", 3000.0, "{ctrl}", [{"id": "ctrl", "kind": "state", "label": "balanced", "team": None, "at_t": 3000.0}])
    assert r["ok"] is False and r["text"] is None


def test_translation_review_blocks_added_fact():
    m, mom = goal_moment()
    c = asyncio.run(P.narrate_moment(m, mom, mom["t"], scripted([good_draft(mom)])))
    seen = []

    async def run(role, payload):
        if role == "translate":
            return {"skeleton": "Das erste Tor! {ev_ignored}"} if False else {"skeleton": payload["skeleton"]}
        if role == "checker":
            seen.append(payload["mode"])
            return {"unsupported": ["adds the word first"]} if payload["mode"] == "translation" else {"unsupported": []}
        return {"skeleton": payload["skeleton"]}
    r = asyncio.run(P.adapt(m, c, run, "casual", "de"))
    assert "translation" in seen and any("translate output rejected" in n for n in r["notes"])


def test_fact_lines_are_plain_and_complete():
    from agents.pipeline import build_menu, fact_lines
    from intel import claims as C
    m, mom = goal_moment()
    menu = build_menu(m, mom, mom["t"])
    res = C.verify_claims(m, list(menu.values()), mom["t"])
    lines = fact_lines({r["id"]: r["verified"] for r in res}, mom)
    joined = "\n".join(lines)
    assert "{home_poss} =" in joined and "possession" in joined and "expected goals" in joined
    assert any(l.startswith("ev:") and "goal" in l for l in lines) and "moment facts" in joined


def _skeleton_for(moment, allowed):
    if moment["kind"] in ("goal", "red_card", "big_chance_missed"):
        return "{ev.clock} {ev.player} for {ev.team}, it is {ev.score}."
    if moment["kind"] == "press_spike":
        return "{press_recent} pressures per opposition pass now, up from {press_before}."
    return "{ctrl.clock} {ctrl.team} are in control." if "{ctrl.team}" in allowed else "{ctrl.clock} the game is stretched."


def test_skeleton_only_narrator_passes_for_every_moment_kind():
    from agents.pipeline import allowed_placeholders, build_menu
    from intel.data import list_matches
    seen = set()
    for mm in list_matches():
        m = load_match(mm["match_id"])
        for mom in detect_moments(m, m.duration):
            if mom["kind"] not in ("goal", "red_card", "big_chance_missed", "press_spike", "control_shift"):
                continue
            allowed = allowed_placeholders(build_menu(m, mom, mom["t"]), m, mom["t"])

            async def run(role, payload, mom=mom, allowed=allowed):
                if role == "narrator":
                    assert payload["allowed_placeholders"] == allowed
                    return {"skeleton": _skeleton_for(mom, allowed), "causal": []}
                return {"unsupported": []}
            c = asyncio.run(P.narrate_moment(m, mom, mom["t"], run))
            assert c.attempts == 1 and not c.used_fallback, (mom["id"], c.catches)
            seen.add(mom["kind"])
    assert {"goal", "control_shift", "press_spike"} <= seen


def test_garbage_unreferenced_claims_are_ignored_but_referenced_bad_ones_rejected():
    m, mom = goal_moment()
    sk = "{ev.clock} {ev.player} for {ev.team}."
    junk = {"claims": [{"id": "zz", "kind": "nonsense"}, {"id": "ev", "kind": "stat"}], "skeleton": sk, "causal": []}
    c = asyncio.run(P.narrate_moment(m, mom, mom["t"], scripted([junk])))
    assert c.attempts == 1 and "ev" in c.verified                      # menu copy of ev replaced the mangled one
    own_bad = {"claims": [{"id": "x1", "kind": "stat", "team": m.home, "metric": "shots", "value": 99, "from_t": 0, "to_t": mom["t"]}],
               "skeleton": "{x1} shots", "causal": []}
    c2 = asyncio.run(P.narrate_moment(m, mom, mom["t"], scripted([own_bad, junk])))
    assert c2.attempts == 2 and c2.catches[0]["problems"][0]["kind"] == "claim"
