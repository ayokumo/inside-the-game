# Synthetic match data

All clubs, players and results are invented. No real match data is shipped or used to generate events.
Player names are drawn from a pool, any resemblance to a real person is coincidental.

## What is modelled

A possession chain simulator. At every step the team with the ball makes one action (pass, shot, carry duel
or restart) and the outcome is drawn from probabilities that depend on:

* player attributes (passing, shooting, defending, stamina, pace)
* pressure from the opposition, which depends on their tactical mode and how tired they are
* numbers on the pitch (red cards)
* the tactical mode of each team: balanced, high press, low block, all out attack
* match level tempo and openness

Shot quality uses a logistic xG curve on distance and goal mouth angle (`matchsim/xg.py`, anchor fit in
`tools/fit_xg.py`). Finishing and goalkeeping skill move the true goal probability around the reported xG, and
goals are recentred so total goals equal total xG on average.

## Event types

| type | notable fields |
| --- | --- |
| pass | recipient, start, end, outcome (complete, incomplete, out), length_m, forward_m, pass_type, ball_speed_kmh, difficulty (0 to 1, the modelled chance of failure), under_pressure, set_piece |
| shot | start, end, outcome (goal, saved, blocked, off_target, post), xg, shot_speed_kmh, body_part, shot_type, assist, distance_m |
| pressure | pressed_player, pressure_level |
| tackle | outcome (won, lost), tackled_player |
| interception, clearance | location |
| possession_change | cause (tackle, interception, loss, block, save, clearance) |
| foul, card | victim, card_type (yellow, second_yellow, red) |
| substitution | player_out, player_in |
| throw_in, goal_kick, corner, free_kick, kickoff | taker, start |
| sprint | speed_kmh, distance_m |
| physical | distance_km (cumulative), top_speed_kmh, emitted every 5 minutes per player |
| period_start, period_end, full_time | markers |

Every event also carries `match_id`, `event_id`, `t` (seconds of playing time), `period`, `minute`, `second`,
`clock` (display string such as `45+2`), `team`, `player`, `player_id`, `score` and `possession_id`.

Coordinates are absolute metres on a 105 by 68 pitch. Home attacks towards x=105 in period 1 and towards x=0 in
period 2.

## Hidden ground truth

`data/truth/<match_id>.json` records the real causes behind each swing: tactical windows (which team, which mode,
why it started and the measured effect on press intensity, shots and pass completion), red cards, substitutions
with the true fatigue of the player taken off, goals and the windows active when they were scored, and a per
minute fatigue curve.

This file is never part of the event stream and no agent tool reads it. It exists so that the Analyst agent's
explanations can be scored against the true cause. Real match data cannot offer this.

## Known limitations

* Speeds and distances are modelled, not measured. There is no tracking data and no video, so the
  auto eventing idea in the challenge is out of scope.
* No penalties, offsides, injuries or VAR. Open play xG is capped at 0.65.
* Players are not located continuously. Positions are formation anchors that shift with the ball and tactical
  mode, so off the ball movement and passing lanes are not modelled.
* Calibration is by anchor points and aggregate checks. The comparison against real data is in
  `data/realism_report.md` and needs the Wyscout download to be complete.
* The three demo matches were chosen from 150 seeds per fixture by a narrative score (`data/selection_log.json`).
  The matches themselves are emergent, not scripted, but the selection is curation and is disclosed as such.

## Reproduce

```
python -m pytest                      # 14 invariant and sanity tests
python -m matchsim.cli select         # rescan seeds and rewrite the three demo matches
python tools/wyscout_benchmark.py /path/to/wyscout/events
python tools/realism.py
```
