import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import wyscout_benchmark as wb  # noqa: E402


def ev(match, name, sub="", tags=()):
    return {"matchId": match, "eventName": name, "subEventName": sub, "tags": [{"id": t} for t in tags]}


def test_parser_counts_on_a_fixture():
    events = [ev(1, "Pass", "Simple pass", [1801]), ev(1, "Pass", "Simple pass", [1802]),
              ev(1, "Pass", "Cross", [1801]), ev(1, "Shot", "Shot", [101]), ev(1, "Shot", "Shot", [1801]),
              ev(1, "Foul", "Foul"), ev(1, "Free Kick", "Corner"), ev(1, "Shot", "Shot", [101, 102]),
              ev(2, "Pass", "Simple pass", [1801])]
    res = wb.summarise(wb.per_match(events))
    assert res["_matches"] == 2
    m1 = wb.per_match(events)[1]
    assert (m1["passes"], m1["accurate"], m1["shots"], m1["goals"], m1["fouls"], m1["corners"]) == (3, 2, 3, 1, 1, 1)
