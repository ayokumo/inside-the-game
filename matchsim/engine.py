"""Rule based match simulator.

Possession chain model: the ball is always with one team, one action at a time
(pass, shot, carry duel, restart). Every outcome is drawn from probabilities that
depend on player attributes, pressure, fatigue, numbers on the pitch and the
team's tactical mode.

Hidden latent causes (tactical windows, fatigue, red cards, substitutions) are
written to a separate truth log. They are NEVER part of the event stream, so an
analyst agent can be scored against the real cause of a swing.

Coordinates: metres on a 105 x 68 pitch. Event coordinates are absolute. Home
attacks towards x=105 in period 1 and towards x=0 in period 2.
Internally everything is computed in the possessing team's attacking frame
(goal at x=105) and converted on output.
"""
from __future__ import annotations

import math
import random

from .world import CLUBS, build_squad
from .xg import shot_geometry, xg_model

L, W = 105.0, 68.0
H_BASE = 2700.0

MODES = {
    "balanced":       dict(shift=0.0,  press=0.30, zone="even", shot=1.00, fwd=1.00, load=1.00),
    "high_press":     dict(shift=10.0, press=0.52, zone="high", shot=1.00, fwd=1.05, load=1.30),
    "low_block":      dict(shift=-9.0, press=0.22, zone="low",  shot=0.85, fwd=1.15, load=0.85),
    "all_out_attack": dict(shift=12.0, press=0.35, zone="high", shot=1.35, fwd=1.25, load=1.20),
}
SHOT_BASE, SHOT_DECAY, GOAL_SCALE = 0.44, 9.5, 0.88  # tuned by tools/tune_shots.py
RUN_MS = {"GK": 0.40, "DEF": 1.55, "MID": 1.90, "FWD": 1.70}  # average running speed m/s


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class TeamState:
    def __init__(self, idx, club_id, rng):
        c = CLUBS[club_id]
        self.idx, self.club_id, self.name = idx, club_id, c["name"]
        self.strength = c["strength"]
        self.side = "home" if idx == 0 else "away"
        starters, bench = build_squad(club_id)
        self.all = starters + bench
        self.mode = "balanced"
        self.mode_until = None
        self.window = None
        self.reds = 0
        self.subs_used = 0
        self.chase_done = False
        self.protect_done = False
        self.sub_times = sorted(
            [rng.uniform(3000, 3700), rng.uniform(3700, 4400), rng.uniform(4400, 5000)]
            + ([rng.uniform(5000, 5300)] if rng.random() < 0.4 else [])
        )

    def on(self):
        return [p for p in self.all if p.on_pitch]

    def gk(self):
        for p in self.on():
            if p.group == "GK":
                return p
        return self.on()[0]

    def anchor(self, p, bx=None):
        """Where a player positions himself: home slot, shifted by tactical mode and
        by the ball (the team moves up and down the pitch with the ball)."""
        if p.group == "GK":
            return p.home
        k = {"DEF": 0.8, "MID": 1.0, "FWD": 0.7}[p.group]
        follow = {"DEF": 0.18, "MID": 0.30, "FWD": 0.40}[p.group]
        ball_shift = follow * ((52.5 if bx is None else bx) - 52.5)
        return (clamp(p.home[0] + MODES[self.mode]["shift"] * k + ball_shift, 3.0, 97.5), p.home[1])

    def mean_fatigue(self):
        o = [p.fatigue for p in self.on() if p.group != "GK"]
        return sum(o) / len(o) if o else 0.0


class Sim:
    def __init__(self, home_id, away_id, seed, match_id="SYN-0001", script=None):
        self.rng = random.Random(seed)
        self.seed, self.match_id, self.script = seed, match_id, [dict(s, applied=False) for s in (script or [])]
        self.teams = [TeamState(0, home_id, self.rng), TeamState(1, away_id, self.rng)]
        # Match level latent variables: how fast the game is played, and how open it is.
        self.tempo = self.rng.uniform(0.78, 1.28)
        self.openness = self.rng.uniform(0.85, 1.20)
        self.events = []
        self.t = 0.0
        self.period = 1
        self.p2_start = None
        self.p_end = H_BASE
        self.p_extended = False
        self.score = [0, 0]
        self.poss, self.poss_id = 0, 0
        self.ball = (52.5, 34.0)
        self.carrier = None
        self.set_piece = None
        self.last_passer = None  # (player, t)
        self.next_tick = 60.0
        self.next_phys = 300.0
        self.windows = []
        self._wid = 0
        self.pending_reds = []
        self.truth = {"seed": seed, "match_context": {"tempo": round(self.tempo, 3), "openness": round(self.openness, 3)},
                      "red_cards": [], "substitutions": [], "script": list(script or []),
                      "fatigue_curve": [], "goals": []}

    # ---------------------------------------------------------------- helpers
    def opp(self, tm):
        return self.teams[1 - tm.idx]

    def abs_xy(self, frame_idx, x, y):
        x, y = clamp(x, 0.0, L), clamp(y, 0.0, W)
        flip = (frame_idx == 0 and self.period == 2) or (frame_idx == 1 and self.period == 1)
        if flip:
            x, y = L - x, W - y
        return [round(x, 1), round(y, 1)]

    def clock(self):
        pt = self.t if self.period == 1 else self.t - self.p2_start
        minute = int(pt // 60) + (0 if self.period == 1 else 45)
        second = int(pt % 60)
        lim = 45 * self.period
        disp = f"{lim}+{minute - lim + 1}" if minute >= lim else str(minute + 1)
        return minute, second, disp

    def emit(self, typ, team=None, player=None, **kw):
        minute, second, disp = self.clock()
        ev = {"match_id": self.match_id, "event_id": len(self.events) + 1, "t": round(self.t, 1),
              "period": self.period, "minute": minute, "second": second, "clock": disp, "type": typ,
              "team": team.name if team else None,
              "player": player.name if player else None,
              "player_id": player.pid if player else None,
              "score": list(self.score), "possession_id": self.poss_id}
        ev.update(kw)
        self.events.append(ev)
        return ev

    def pick_near(self, tm, pos, exclude=(), sigma=18.0, allow_gk=False):
        cands = [p for p in tm.on() if p not in exclude and (allow_gk or p.group != "GK")]
        ws = []
        for p in cands:
            ax, ay = tm.anchor(p, pos[0])
            d2 = (ax - pos[0]) ** 2 + (ay - pos[1]) ** 2
            ws.append(math.exp(-d2 / (2 * sigma * sigma)) + 1e-6)
        return self.rng.choices(cands, weights=ws)[0]

    # --------------------------------------------------------- tactics / truth
    def set_mode(self, tm, mode, reason, until=None):
        if tm.mode == mode:
            tm.mode_until = until
            return
        if tm.window is not None:
            if self.t - tm.window["start_t"] < 45.0:  # too short to be a real tactical phase
                self.windows.remove(tm.window)
            else:
                tm.window["end_t"] = round(self.t, 1)
            tm.window = None
        tm.mode, tm.mode_until = mode, until
        if mode != "balanced":
            self._wid += 1
            w = {"id": f"W{self._wid}", "team": tm.name, "side": tm.side, "mode": mode,
                 "reason": reason, "start_t": round(self.t, 1), "end_t": None}
            self.windows.append(w)
            tm.window = w

    def tactics_tick(self):
        for s in self.script:
            if not s["applied"] and self.t >= s["at_min"] * 60:
                s["applied"] = True
                tm = self.teams[0 if s["team"] == "home" else 1]
                if s["type"] == "mode":
                    self.set_mode(tm, s["mode"], "script", self.t + s.get("duration_min", 15) * 60)
                elif s["type"] == "red_card":
                    self.pending_reds.append(tm.idx)
        for tm in self.teams:
            if tm.mode_until is not None and self.t >= tm.mode_until:
                self.set_mode(tm, "balanced", "window_ended")
            diff = self.score[tm.idx] - self.score[1 - tm.idx]
            if tm.mode == "all_out_attack" and tm.window and tm.window["reason"] == "chasing_game" and diff >= 0:
                self.set_mode(tm, "balanced", "equalised")
            if tm.mode == "balanced":
                if self.t > 3900 and diff < 0 and not tm.chase_done:
                    tm.chase_done = True
                    self.set_mode(tm, "all_out_attack", "chasing_game")
                elif self.t > 4500 and diff == 1 and not tm.protect_done and self.rng.random() < 0.7:
                    tm.protect_done = True
                    self.set_mode(tm, "low_block", "protecting_lead")
                elif tm.reds == 0 and self.rng.random() < 0.011:
                    self.set_mode(tm, "high_press", "style_shift", self.t + self.rng.uniform(480, 1080))
                elif self.rng.random() < 0.006:
                    self.set_mode(tm, "low_block", "style_shift", self.t + self.rng.uniform(480, 900))
        self.truth["fatigue_curve"].append({
            "t": round(self.t), "home": round(self.teams[0].mean_fatigue(), 3),
            "away": round(self.teams[1].mean_fatigue(), 3)})

    def process_subs(self):
        # scripted red cards are applied at a dead ball so the ball carrier is never removed mid action
        while self.pending_reds:
            tm = self.teams[self.pending_reds.pop(0)]
            cand = [p for p in tm.on() if p.group != "GK"]
            if cand:
                self.send_off(tm, self.rng.choice(cand), "red")
        for tm in self.teams:
            while tm.sub_times and self.t >= tm.sub_times[0] and tm.subs_used < 5:
                tm.sub_times.pop(0)
                outs = [p for p in tm.on() if p.group != "GK"]
                bench = [p for p in tm.all if not p.on_pitch and not p.sent_off and not p.started
                         and not p.subbed_off and p.group != "GK"]
                if not outs or not bench:
                    break
                out = max(outs, key=lambda p: p.fatigue)
                same = [b for b in bench if b.group == out.group]
                inn = self.rng.choice(same or bench)
                out.on_pitch, out.subbed_off = False, True
                inn.on_pitch, inn.slot, inn.home, inn.t_on, inn.fatigue = True, out.slot, out.home, self.t, 0.0
                tm.subs_used += 1
                self.emit("substitution", tm, inn, player_out=out.name, player_out_id=out.pid,
                          player_in=inn.name, player_in_id=inn.pid)
                self.truth["substitutions"].append({
                    "t": round(self.t, 1), "team": tm.name, "out": out.pid, "in": inn.pid,
                    "out_fatigue": round(out.fatigue, 3)})

    def send_off(self, tm, player, why):
        player.on_pitch, player.sent_off = False, True
        tm.reds += 1
        self.emit("card", tm, player, card_type=why if why in ("second_yellow",) else "red",
                  start=None)
        self.truth["red_cards"].append({"t": round(self.t, 1), "team": tm.name, "player": player.pid, "kind": why})
        self.set_mode(tm, "low_block", "red_card")
        other = self.opp(tm)
        if other.mode == "balanced" and self.rng.random() < 0.6:
            self.set_mode(other, "high_press", "man_advantage", self.t + 900)

    # ------------------------------------------------------------ fatigue/phys
    def advance(self, dt):
        for tm in self.teams:
            load = MODES[tm.mode]["load"]
            for p in tm.on():
                p.fatigue = min(1.0, p.fatigue + (1.1 - p.stamina) * 0.75 / 5400.0 * load * dt)
                run = RUN_MS[p.group] * load * (1 - 0.18 * p.fatigue) * dt
                p.dist_m += run * self.rng.uniform(0.85, 1.15)
        self.t += dt
        while self.t >= self.next_phys:
            self.next_phys += 300.0
            for tm in self.teams:
                for p in tm.on():
                    if p.top_speed == 0.0:
                        p.top_speed = round(24 + 8 * p.pace * self.rng.uniform(0.9, 1.0), 1)
                    self.emit("physical", tm, p, distance_km=round(p.dist_m / 1000, 2),
                              top_speed_kmh=p.top_speed)

    # ------------------------------------------------------------ possessions
    def to_possession(self, tm, pos, cause, carrier=None):
        """Ball goes to team tm at pos (tm's frame). Emits a possession_change."""
        self.poss = tm.idx
        self.poss_id += 1
        self.ball = (clamp(pos[0], 0.5, 104.5), clamp(pos[1], 0.5, 67.5))
        self.carrier = carrier or self.pick_near(tm, self.ball, allow_gk=self.ball[0] < 20)
        self.emit("possession_change", tm, self.carrier, cause=cause,
                  start=self.abs_xy(tm.idx, *self.ball))

    def restart(self, tm, kind, pos):
        self.process_subs()
        self.poss = tm.idx
        self.poss_id += 1
        self.ball = (clamp(pos[0], 0.5, 104.5), clamp(pos[1], 0.5, 67.5))
        if kind == "goal_kick":
            taker = tm.gk()
        elif kind == "kickoff":
            taker = max((p for p in tm.on() if p.group == "FWD"), key=lambda p: p.passing, default=tm.on()[0])
        elif kind in ("corner", "free_kick"):
            taker = max((p for p in tm.on() if p.group in ("MID", "FWD")), key=lambda p: p.passing + self.rng.random() * 0.1)
        else:  # throw_in
            taker = self.pick_near(tm, self.ball, sigma=14.0)
        self.carrier = taker
        self.set_piece = kind
        self.emit(kind, tm, taker, start=self.abs_xy(tm.idx, *self.ball))

    def pressure_check(self, tp, to, carrier):
        m = MODES[to.mode]
        x = self.ball[0]
        z = {"high": 1.4 if x < 50 else (1.0 if x < 75 else 0.7),
             "low": 0.6 if x < 50 else (1.0 if x < 70 else 1.5), "even": 1.0}[m["zone"]]
        eff = clamp(1.2 - 1.6 * to.mean_fatigue(), 0.5, 1.2)
        num = (len(to.on()) / 11.0) / (len(tp.on()) / 11.0)
        p = clamp(m["press"] * z * eff * num * (1.25 - carrier.passing), 0.02, 0.9)
        return self.rng.random() < p, p

    # ------------------------------------------------------------------ steps
    def step(self):
        tp = self.teams[self.poss]
        to = self.opp(tp)
        carrier = self.carrier
        x, y = self.ball
        sp, self.set_piece = self.set_piece, None

        if sp == "corner":
            return self.do_corner(tp, to, carrier)
        if sp == "free_kick" and x > 82 and abs(y - 34) < 18 and self.rng.random() < 0.25:
            return self.do_shot(tp, to, carrier, False, "free_kick")

        pressed, level = (False, 0.0) if sp else self.pressure_check(tp, to, carrier)
        presser = None
        if pressed:
            presser = self.pick_near(to, (L - x, W - y))
            if self.rng.random() < 0.6:
                self.emit("pressure", to, presser, start=self.abs_xy(tp.idx, x, y),
                          pressure_level=round(level, 2), pressed_player=carrier.name,
                          pressed_player_id=carrier.pid)
            if self.rng.random() < 0.07:
                return self.do_foul(to, presser, tp, carrier)
            if self.rng.random() < 0.11:
                return self.do_duel(tp, to, carrier, presser)

        dist, angle = shot_geometry(x, y)
        base = clamp(SHOT_BASE * math.exp(-(dist - 6.0) / SHOT_DECAY), 0.004, SHOT_BASE)
        if angle < 0.25:
            base *= 0.5
        p_shot = base * MODES[tp.mode]["shot"] * (0.8 + 0.4 * carrier.shooting) * (0.7 if pressed else 1.0) * self.openness
        if sp not in ("goal_kick", "throw_in") and self.rng.random() < p_shot:
            return self.do_shot(tp, to, carrier, pressed, "open_play")
        return self.do_pass(tp, to, carrier, pressed, sp)

    def do_foul(self, fouler_tm, fouler, victim_tm, victim):
        x, y = self.ball
        self.emit("foul", fouler_tm, fouler, start=self.abs_xy(victim_tm.idx, x, y), victim=victim.name,
                  victim_id=victim.pid)
        r = self.rng.random()
        if r < 0.004:
            self.send_off(fouler_tm, fouler, "red")
        elif r < 0.004 + 0.15:
            if fouler.yellow == 1:
                fouler.yellow = 2
                self.send_off(fouler_tm, fouler, "second_yellow")
            else:
                fouler.yellow += 1
                self.emit("card", fouler_tm, fouler, card_type="yellow", start=None)
        self.advance(self.rng.uniform(15, 30))
        self.restart(victim_tm, "free_kick", (x, y))

    def do_duel(self, tp, to, carrier, tackler):
        x, y = self.ball
        win = clamp(0.40 + 0.5 * (tackler.defending - 0.6) - 0.3 * (carrier.passing - 0.6), 0.15, 0.7)
        won = self.rng.random() < win
        self.emit("tackle", to, tackler, start=self.abs_xy(tp.idx, x, y), outcome="won" if won else "lost",
                  tackled_player=carrier.name, tackled_player_id=carrier.pid,
                  under_pressure=True)
        self.advance(self.rng.uniform(1.0, 2.5))
        if won:
            if self.rng.random() < 0.30:
                return self.do_foul(to, tackler, tp, carrier)
            self.to_possession(to, (L - x, W - y), "tackle", carrier=tackler)

    def do_pass(self, tp, to, carrier, pressed, sp):
        rng = self.rng
        x, y = self.ball
        m = MODES[tp.mode]
        cands = [p for p in tp.on() if p is not carrier]
        ws = []
        for p in cands:
            ax, ay = tp.anchor(p, x)
            d2 = (ax - x) ** 2 + (ay - y) ** 2
            w = math.exp(-d2 / (2 * 18.0 ** 2)) * math.exp(0.45 * m["fwd"] * (ax - x) / 20.0)
            if p.group == "GK" and x > 35 and not pressed:
                w *= 0.05
            if sp == "goal_kick":
                w = (3.0 if p.group in ("MID", "FWD") else 0.3) * (1 if p.group != "GK" else 0.0)
            ws.append(w + 1e-9)
        r = rng.choices(cands, weights=ws)[0]
        ax, ay = tp.anchor(r, x)
        if sp == "goal_kick":
            tx, ty = rng.uniform(38, 70), rng.uniform(10, 58)
        else:
            tx = clamp(ax + rng.gauss(0, 5), 1.0, 100.0)
            ty = clamp(ay + 0.2 * (y - 34) + rng.gauss(0, 6), 1.0, 67.0)
        dist = math.hypot(tx - x, ty - y)
        fwd = tx - x
        ptype = "long" if dist > 32 else ("forward" if fwd > 12 else ("back" if fwd < -8 else "short"))
        man = (len(tp.on()) - len(to.on())) / 11.0
        z = (2.45 + 3.2 * (carrier.passing - 0.6) + 1.2 * (r.passing - 0.6)
             - 0.05 * max(0.0, dist - 14.0) - 1.0 * pressed - 0.012 * max(0.0, fwd)
             - 0.5 * (x > 78) - 0.9 * carrier.fatigue + 2.0 * man + (0.6 if sp == "throw_in" else 0.0))
        p_ok = sigmoid(z)
        speed = clamp(18.0 + 1.15 * dist + rng.gauss(0, 4), 18.0, 95.0)
        flight = dist / (speed / 3.6)
        base = dict(recipient=r.name, recipient_id=r.pid, length_m=round(dist, 1), forward_m=round(fwd, 1),
                    pass_type=ptype, ball_speed_kmh=round(speed, 1), difficulty=round(1 - p_ok, 3),
                    under_pressure=bool(pressed), set_piece=sp)
        start = self.abs_xy(tp.idx, x, y)
        if rng.random() < p_ok:
            self.emit("pass", tp, carrier, start=start, end=self.abs_xy(tp.idx, tx, ty), outcome="complete", **base)
            self.advance(rng.uniform(2.4, 5.4) * self.tempo + flight)
            self.ball, self.carrier = (tx, ty), r
            self.last_passer = (carrier, self.t)
            if dist > 24 and fwd > 18 and rng.random() < 0.4:
                sp_kmh = clamp(24 + 11 * r.pace + rng.gauss(0, 1.5), 22.0, 37.5) * (1 - 0.12 * r.fatigue)
                r.top_speed = max(r.top_speed, round(sp_kmh, 1))
                self.emit("sprint", tp, r, start=self.abs_xy(tp.idx, tx, ty), speed_kmh=round(sp_kmh, 1),
                          distance_m=round(rng.uniform(18, 40), 1))
            return
        kind = rng.choices(["interception", "out", "lost"], weights=[0.20, 0.30, 0.50])[0]
        frac = rng.uniform(0.6, 0.95)
        ex, ey = x + (tx - x) * frac, y + (ty - y) * frac
        self.advance(rng.uniform(2.4, 5.4) * self.tempo + flight * frac)
        self.last_passer = None
        if kind == "out":
            side = 0.5 if ey < 34 else 67.5
            if tx > 97:
                self.emit("pass", tp, carrier, start=start, end=self.abs_xy(tp.idx, 105.0, ey), outcome="out", **base)
                if rng.random() < 0.4:  # deflected behind off a defender
                    return self.restart(tp, "corner", (104.5, 0.5 if ey < 34 else 67.5))
                return self.restart(to, "goal_kick", (5.5, 34.0))
            self.emit("pass", tp, carrier, start=start, end=self.abs_xy(tp.idx, ex, side), outcome="out", **base)
            return self.restart(to, "throw_in", (L - ex, W - side))
        self.emit("pass", tp, carrier, start=start, end=self.abs_xy(tp.idx, ex, ey), outcome="incomplete", **base)
        pos_to = (L - ex, W - ey)
        if kind == "interception":
            who = self.pick_near(to, pos_to, allow_gk=pos_to[0] < 20)
            self.emit("interception", to, who, start=self.abs_xy(tp.idx, ex, ey))
            self.to_possession(to, pos_to, "interception", carrier=who)
        else:
            self.to_possession(to, pos_to, "loss")

    def do_shot(self, tp, to, shooter, pressed, sp):
        rng = self.rng
        x, y = self.ball
        header = (sp == "corner_header") or (sp == "open_play" and x > 88 and abs(y - 34) < 14 and rng.random() < 0.12)
        xg = xg_model(x, y, header, pressed, "free_kick" if sp == "free_kick" else "open_play")
        finish = 0.8 + (shooter.shooting - 0.4) * 0.8
        gk = to.gk()
        gkf = 1.12 - (gk.defending - 0.5) * 0.5
        p_goal = clamp(xg * finish * gkf * GOAL_SCALE, 0.001, 0.95)  # GOAL_SCALE recentres goals on xG
        dist, _ = shot_geometry(x, y)
        speed = clamp(rng.gauss(55, 8) if header else rng.gauss(98 - 0.4 * max(0.0, dist - 12), 10), 35.0, 135.0)
        assist = None
        if self.last_passer and self.last_passer[0] is not shooter and self.t - self.last_passer[1] < 12:
            assist = self.last_passer[0]
        r = rng.random()
        if r < p_goal:
            outcome = "goal"
        else:
            outcome = rng.choices(["blocked", "saved", "off_target", "post"], weights=[0.30, 0.27, 0.39, 0.04])[0]
        ey = {"goal": 34 + rng.uniform(-3.2, 3.2), "saved": 34 + rng.uniform(-3.0, 3.0),
              "off_target": 34 + rng.choice([-1, 1]) * rng.uniform(4.2, 9.0), "post": 34 + rng.choice([-1, 1]) * 3.66,
              "blocked": y + (34 - y) * 0.25}[outcome]
        ex = x + (dist * 0.25) if outcome == "blocked" else 105.0
        pre_score = list(self.score)
        if outcome == "goal":
            self.score[tp.idx] += 1
        self.emit("shot", tp, shooter, start=self.abs_xy(tp.idx, x, y), end=self.abs_xy(tp.idx, ex, ey),
                  outcome=outcome, xg=round(xg, 3), shot_speed_kmh=round(speed, 1),
                  body_part="head" if header else "foot", shot_type=sp if sp != "corner_header" else "corner",
                  under_pressure=bool(pressed), assist=assist.name if assist else None,
                  assist_id=assist.pid if assist else None, distance_m=round(dist, 1))
        self.advance(rng.uniform(1.0, 2.5))
        self.last_passer = None
        if outcome == "goal":
            self.truth["goals"].append({
                "t": round(self.t, 1), "team": tp.name, "scorer": shooter.pid, "xg": round(xg, 3),
                "windows_active": [w["id"] for w in self.windows if w["end_t"] is None],
                "score_before": pre_score})
            self.advance(rng.uniform(30, 50))
            return self.restart(to, "kickoff", (52.5, 34.0))
        if outcome == "off_target" or (outcome == "post" and rng.random() >= 0.35):
            return self.restart(to, "goal_kick", (5.5, 34.0))
        if outcome == "post":
            self.carrier = self.pick_near(tp, (x + 2, y), exclude=(shooter,))
            self.ball = (clamp(x + 2, 0.5, 104.5), y)
            return
        if outcome == "blocked":
            if rng.random() < 0.40:
                return self.restart(tp, "corner", (104.5, 0.5 if y < 34 else 67.5))
            return self.to_possession(to, (L - ex, W - ey), "block")
        if rng.random() < 0.55:  # saved
            return self.restart(tp, "corner", (104.5, 0.5 if y < 34 else 67.5))
        self.to_possession(to, (5.5, 34.0 + rng.uniform(-6, 6)), "save", carrier=gk)

    def do_corner(self, tp, to, taker):
        rng = self.rng
        x, y = self.ball
        tx, ty = rng.uniform(92, 101), rng.uniform(26, 42)
        target = self.pick_near(tp, (tx, ty), exclude=(taker,))
        dist = math.hypot(tx - x, ty - y)
        p_ok = clamp(0.30 + 0.5 * (taker.passing - 0.6), 0.15, 0.5)
        speed = clamp(rng.gauss(70, 8), 45, 95)
        base = dict(recipient=target.name, recipient_id=target.pid, length_m=round(dist, 1), forward_m=round(tx - x, 1),
                    pass_type="cross", ball_speed_kmh=round(speed, 1), difficulty=round(1 - p_ok, 3),
                    under_pressure=False, set_piece="corner")
        self.advance(rng.uniform(4, 8) + dist / (speed / 3.6))
        if rng.random() < p_ok:
            self.emit("pass", tp, taker, start=self.abs_xy(tp.idx, x, y), end=self.abs_xy(tp.idx, tx, ty),
                      outcome="complete", **base)
            self.last_passer = (taker, self.t)
            self.ball, self.carrier = (tx, ty), target
            if rng.random() < 0.5:
                return self.do_shot(tp, to, target, False, "corner_header")
            return
        self.emit("pass", tp, taker, start=self.abs_xy(tp.idx, x, y), end=self.abs_xy(tp.idx, tx, ty),
                  outcome="incomplete", **base)
        who = self.pick_near(to, (L - tx, W - ty))
        self.emit("clearance", to, who, start=self.abs_xy(tp.idx, tx, ty))
        self.to_possession(to, (L - tx + rng.uniform(5, 20), W - ty + rng.uniform(-10, 10)), "clearance")

    # -------------------------------------------------------------------- run
    def play_period(self):
        guard = 0
        while self.t < self.p_end and guard < 6000:
            guard += 1
            if self.t >= self.next_tick:
                self.next_tick += 60.0
                self.tactics_tick()
            self.step()

    def run(self):
        for tm in self.teams:
            for p in tm.all:
                p.fatigue = 0.0
        self.restart(self.teams[0], "kickoff", (52.5, 34.0))
        # first half
        h1_end = H_BASE + 60.0 * (1 + self.rng.randint(0, 3))
        self.p_end = h1_end
        self.play_period()
        self.emit("period_end", None, None)
        self.p2_start = self.t
        self.period = 2
        for tm in self.teams:
            for p in tm.all:
                p.fatigue *= 0.85
            if tm.mode != "balanced":
                self.set_mode(tm, "balanced", "halftime")
        self.p_end = self.t + H_BASE + 60.0 * (2 + self.rng.randint(0, 4)) + 20.0 * sum(self.score)
        self.poss_id += 1
        self.emit("period_start", None, None)
        self.restart(self.teams[1], "kickoff", (52.5, 34.0))
        self.play_period()
        self.emit("full_time", None, None)
        for tm in self.teams:
            if tm.window is not None:
                tm.window["end_t"] = round(self.t, 1)
        return self.finish()

    def finish(self):
        self.truth["tactical_windows"] = self.windows
        for w in self.windows:
            w["measured"] = measure_window(self.events, w, self.teams)
        meta = {
            "match_id": self.match_id, "seed": self.seed, "synthetic": True,
            "pitch": {"length_m": L, "width_m": W},
            "coordinate_system": "absolute metres; home attacks +x in period 1, -x in period 2",
            "kickoff_local": "2026-08-15T15:00:00",
            "home": self._team_meta(self.teams[0]), "away": self._team_meta(self.teams[1]),
            "final_score": {"home": self.score[0], "away": self.score[1]},
            "duration_s": round(self.t, 1),
        }
        return {"events": self.events, "meta": meta, "truth": self.truth}

    def _team_meta(self, tm):
        c = CLUBS[tm.club_id]
        return {"club_id": tm.club_id, "name": tm.name, "formation": c["formation"],
                "players": [{"player_id": p.pid, "name": p.name, "group": p.group, "started": p.started}
                            for p in tm.all]}


def measure_window(events, w, teams):
    """Ground truth effect of a tactical window: team metrics inside vs outside it.
    Press intensity is pressures per opponent possession action, so it is not
    confounded by how much of the ball each side has."""
    team = w["team"]
    opp = [t.name for t in teams if t.name != team][0]
    inside = lambda e: w["start_t"] <= e["t"] < w["end_t"]
    dur_in = max(1.0, (w["end_t"] - w["start_t"]) / 60.0)
    dur_out = max(1.0, events[-1]["t"] / 60.0 - dur_in)

    def count(pred, ins):
        return sum(1 for e in events if pred(e) and inside(e) == ins)

    def intensity(ins):
        n = count(lambda e: e["type"] == "pressure" and e["team"] == team, ins)
        d = count(lambda e: e["type"] in ("pass", "shot") and e["team"] == opp, ins)
        return round(n / d, 3) if d else None

    def comp(name, ins):
        ps = [e for e in events if e["type"] == "pass" and e["team"] == name and inside(e) == ins]
        return round(sum(e["outcome"] == "complete" for e in ps) / len(ps), 3) if ps else None

    return {
        "press_intensity_in": intensity(True), "press_intensity_out": intensity(False),
        "shots_per_min_in": round(count(lambda e: e["type"] == "shot" and e["team"] == team, True) / dur_in, 3),
        "shots_per_min_out": round(count(lambda e: e["type"] == "shot" and e["team"] == team, False) / dur_out, 3),
        "pass_completion_in": comp(team, True), "pass_completion_out": comp(team, False),
        "opp_pass_completion_in": comp(opp, True), "opp_pass_completion_out": comp(opp, False),
    }


def simulate(home_id, away_id, seed, match_id="SYN-0001", script=None):
    return Sim(home_id, away_id, seed, match_id, script).run()
