import numpy as np
from foosball import Ball, Pitch

EPS = 1e-8          # numerical guard

# ── Tunable constants ─────────────────────────────────────────────────────────
SLOPE_WEIGHT = 0.3  # strength of global "tilt" in field
BOUNDARY_A   = 80.0 # wall repulsion amplitude
BOUNDARY_W   = 3.0  # wall decay width [m] — significant only within ~3m of edge

#── Mathematical functs ────────────────────────────────────────────────────────

#PRE: ball.team is 0,1 or None ; own_team is 0 or 1
#POST: Will return 'attack' or 'defense' or 'neutral'
def determine_game_state(ball, own_team: int) -> str:
    if ball.team == own_team:
        return 'attack'
    elif ball.team is None:
        return 'neutral'
    else:
        return 'defense'

#PRE:   pos and center of gradient are 2D-Vectors; sigma > 0
#POST:  gradient of v (bell curve); amplitude > 0 -> repulsive hill, amplitude < 0 -> attractive well
def gauss_grad(pos: np.ndarray, center: np.ndarray, amplitude: float, sigma: float) -> np.ndarray:
    diff = pos - center                        # vector pointing away from center
    r2   = np.dot(diff, diff)                  # difference squared (scalar)
    V    = amplitude * np.exp(-r2 / (2.0 * sigma**2))
    return V * (-diff / sigma**2)              


#PRE:   pos and center of gradient are 2D-Vectors, scalar b > 0 
#       first attractive ring at: solve b = π/(2·target_r).
#POST:  returns gradient of V(r) = A * sin (b * r)/r
#       singularity at 0 guarded by early return 
def sombrero_grad(pos: np.ndarray, center: np.ndarray, amplitude: float, b: float) -> np.ndarray:
    diff = pos - center
    r = np.sqrt(np.dot(diff, diff)) #scalar distance
    if r < EPS:                     #dealing with singularity at zero with early zero
        return np.zeros(2)
    r_hat = diff/r                  #unit radial vector
    dV_dr  = amplitude * (b * np.cos(b * r) / r  -  np.sin(b * r) / r**2)
    return dV_dr * r_hat

#PRE:   takes own_team in {0,1} and game_state in {'attack', 'defense', 'neutral'}
#POST:  'neutral' -> no slope
#       'attack'  -> pull towards enemy goal
#       'defense' -> pull towards own goal
def slope_grad(own_team: int, game_state: str) -> np.ndarray:
    if game_state == 'neutral':
        return np.zeros(2)
    sign = -1.0 if own_team == 0 else 1.0           #team 1 is on leftmost bondary, team 0 at rightmost boundary
    if game_state == 'defense':
        sign *= -1.0                                #pull back to own goal
    return np.array([sign * SLOPE_WEIGHT, 0.0])

#PRE:   pos 2D-Vector, Works outside pitch too (even steeper slope)
#POST:  gradient from four exponential walls, points toward nearest wall
def boundary_grad(pos: np.ndarray) -> np.ndarray:
    x, y = pos[0], pos[1]
    a_w  = BOUNDARY_A / BOUNDARY_W

    d_left   = x + Pitch.X_BOUND
    d_right  = Pitch.X_BOUND - x
    d_bottom = y + Pitch.Y_BOUND
    d_top    = Pitch.Y_BOUND - y

    gx = -a_w * np.exp(-d_left   / BOUNDARY_W) \
         +a_w * np.exp(-d_right  / BOUNDARY_W)
    gy = -a_w * np.exp(-d_bottom / BOUNDARY_W) \
         +a_w * np.exp(-d_top    / BOUNDARY_W)

    return np.array([gx, gy])
