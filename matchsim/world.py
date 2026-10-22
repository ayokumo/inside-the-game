"""Fictional clubs and squads. Every club, player and result is invented.

Club names are provisional: rename them here and nothing else changes.
Player names are generated from a pool, any resemblance to a real person is
coincidental.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# Slot name -> (x, y) home anchor in the team's attacking frame (goal at x=105).
FORMATIONS = {
    "4-3-3": [
        ("GK", 5, 34), ("LB", 30, 10), ("LCB", 26, 24), ("RCB", 26, 44), ("RB", 30, 58),
        ("DM", 42, 34), ("LCM", 52, 22), ("RCM", 52, 46),
        ("LW", 74, 12), ("ST", 82, 34), ("RW", 74, 56),
    ],
    "4-4-2": [
        ("GK", 5, 34), ("LB", 30, 10), ("LCB", 26, 24), ("RCB", 26, 44), ("RB", 30, 58),
        ("LM", 52, 10), ("LCM", 50, 26), ("RCM", 50, 42), ("RM", 52, 58),
        ("ST1", 76, 26), ("ST2", 76, 42),
    ],
}

GROUP_OF_SLOT = {
    "GK": "GK", "LB": "DEF", "LCB": "DEF", "RCB": "DEF", "RB": "DEF",
    "DM": "MID", "LCM": "MID", "RCM": "MID", "LM": "MID", "RM": "MID",
    "LW": "FWD", "RW": "FWD", "ST": "FWD", "ST1": "FWD", "ST2": "FWD",
}

CLUBS = {
    "NBR": {"name": "Northbridge FC", "seed": 11, "strength": 0.64, "formation": "4-3-3"},
    "HLW": {"name": "Harlow United", "seed": 23, "strength": 0.60, "formation": "4-4-2"},
    "CAS": {"name": "Castlemere Athletic", "seed": 37, "strength": 0.58, "formation": "4-3-3"},
    "PAV": {"name": "Port Avalon", "seed": 41, "strength": 0.62, "formation": "4-4-2"},
}

_INITIALS = list("ABCDEFGHJKLMNOPRSTWY")
_SURNAMES = [
    "Okafor", "Mensah", "Moreau", "Kowalski", "Haddad", "Tanaka", "Lindqvist", "Costa",
    "Novak", "Adeyemi", "Rossi", "Petrov", "Duarte", "Fofana", "Bianchi", "Halvorsen",
    "Ibarra", "Chukwu", "Varga", "Sorensen", "Dlamini", "Marchetti", "Osei", "Brandt",
    "Karimi", "Delacroix", "Nwosu", "Jankowski", "Almeida", "Sato", "Reyes", "Volkov",
    "Abara", "Lefevre", "Ng", "Bergstrom", "Quaye", "Castellano", "Yilmaz", "Ferreira",
    "Obi", "Hartmann", "Zielinski", "Mbeki", "Santoro", "Laurent", "Takahashi", "Eze",
    "Ostrowski", "Camara", "Dimitrov", "Vidal", "Okonkwo", "Fischer", "Benitez", "Asante",
    "Lund", "Mercier", "Idowu", "Kovac", "Salazar", "Boateng", "Rinaldi", "Wallin",
]


@dataclass
class Player:
    pid: str
    name: str
    group: str
    passing: float
    shooting: float
    defending: float
    stamina: float
    pace: float
    slot: str | None = None
    home: tuple = (0.0, 0.0)
    fatigue: float = 0.0
    dist_m: float = 0.0
    top_speed: float = 0.0
    t_on: float = 0.0
    on_pitch: bool = False
    yellow: int = 0
    sent_off: bool = False
    started: bool = False
    subbed_off: bool = False

    def anchor_home(self):
        return self.home


def _clip(v, lo=0.3, hi=0.92):
    return max(lo, min(hi, v))


def build_squad(club_id: str):
    """Return (starters, bench) for a club. Deterministic per club."""
    c = CLUBS[club_id]
    rng = random.Random(c["seed"])
    names = _SURNAMES[:]
    rng.shuffle(names)
    formation = FORMATIONS[c["formation"]]
    base = c["strength"]

    def make(i, group):
        g_bonus = {
            "GK": dict(passing=-0.05, shooting=-0.3, defending=0.12),
            "DEF": dict(passing=0.0, shooting=-0.12, defending=0.10),
            "MID": dict(passing=0.08, shooting=0.0, defending=0.0),
            "FWD": dict(passing=-0.02, shooting=0.12, defending=-0.12),
        }[group]
        return Player(
            pid=f"{club_id}{i:02d}",
            name=f"{rng.choice(_INITIALS)}. {names[i]}",
            group=group,
            passing=_clip(base + g_bonus["passing"] + rng.gauss(0, 0.07)),
            shooting=_clip(base + g_bonus["shooting"] + rng.gauss(0, 0.07)),
            defending=_clip(base + g_bonus["defending"] + rng.gauss(0, 0.07)),
            stamina=_clip(0.62 + rng.gauss(0, 0.09), 0.4, 0.9),
            pace=_clip(0.55 + rng.gauss(0, 0.12), 0.3, 0.95),
        )

    starters = []
    for i, (slot, x, y) in enumerate(formation):
        p = make(i + 1, GROUP_OF_SLOT[slot])
        p.slot, p.home, p.on_pitch, p.started = slot, (float(x), float(y)), True, True
        starters.append(p)
    bench = [make(12 + j, g) for j, g in enumerate(["GK", "DEF", "MID", "MID", "FWD"])]
    return starters, bench
