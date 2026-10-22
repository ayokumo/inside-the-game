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
