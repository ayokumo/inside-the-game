"""Expected goals model used by the simulator.

Logistic on shot distance and the angle the goal mouth subtends. Coefficients come
from tools/fit_xg.py (anchor fit, an engineering approximation, not a real data fit).
"""
import math

GOAL_X, GOAL_Y, POST_HALF = 105.0, 34.0, 3.66
B0, B1, B2 = -3.1328, -0.0454, 2.7059


def shot_geometry(x: float, y: float):
    """Return (distance_m, goal_mouth_angle_rad) for a shot from (x, y) in the
    shooting team's attacking frame (goal at x=105)."""
    dx = max(GOAL_X - x, 0.5)
    dy = y - GOAL_Y
    dist = math.hypot(dx, dy)
    a1 = math.atan2(POST_HALF - dy, dx)
    a2 = math.atan2(-POST_HALF - dy, dx)
    return dist, abs(a1 - a2)


def xg_model(x: float, y: float, header: bool = False, under_pressure: bool = False,
             set_piece: str = "open_play") -> float:
    dist, angle = shot_geometry(x, y)
    z = B0 + B1 * dist + B2 * angle
    p = min(1.0 / (1.0 + math.exp(-z)), 0.65)  # open play ceiling, penalties are not modelled
    if header:
        p *= 0.75
    if under_pressure:
        p *= 0.85
    if set_piece == "free_kick":
        p = min(p, 0.08)
    return max(0.001, min(0.95, p))
