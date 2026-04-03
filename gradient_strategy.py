import numpy as np
from foosball import Ball, Pitch, DistanceLimits

EPS = 1e-8          # numerical guard

# ── Amplitude constants (to tune)  ────────────────────────────────
SIGMA_PLAYER    = 8.0       # spatial spread of player potentials [m]
SIGMA_BALL      = 20.0       # spatial spread of ball potential [m]
AMP_OWN_REPEL   = 2.0       # own-player repulsion (to spread out)
AMP_OPP_ATTACK  = 3.0       # opponent repulsion in attack (avoid blockers)
AMP_OPP_NEUTRAL = 2.5       # opponent repulsion in neutral
AMP_BALL_ATTACK = -3.0      # ball attraction in attack (weak — carrier has it)
AMP_BALL_NEUTRAL= -20.0      # ball attraction in neutral (strong — go get it)
SOMBRERO_B      = 0.22      # ring distance ≈ π/(2·0.22) ≈ 7m
SOMBRERO_AMP    = 1.5       # might need to be tuned to make it stronger than ball attraction in attack

SLOPE_WEIGHT    = 0.3       # strength of global "tilt" in field
BOUNDARY_A      = 80.0      # wall repulsion amplitude
BOUNDARY_W      = 3.0       # wall decay width [m] — significant only within ~3m of edge

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

#── Field composition ────────────────────────────────────────────────────────

# PRE:  all arrays valid, game_state in {'attack', 'defense', 'neutral'}
#       own_coords is the committed-moves array, not the original snapshot.
# POST: 2D-Vector as summed gradient; must add negative sign to get descent direction when called
def total_gradient(
    pos:         np.ndarray,  # 2D-vector — position of the player we're computing for
    own_coords:  np.ndarray,  # 5 2D-Vectors — use new_coords, NOT original own positions (to base calculation on future positions)
    opp_coords:  np.ndarray,  # 5 2D-Vectors — opponent positions
    ball_coords: np.ndarray,  # 2D-vector - position of ball
    game_state:  str,         # 'attack' | 'neutral' | 'defense'
    own_team:    int,
    player_idx:  int          # skip self-repulsion
) -> np.ndarray:

    grad = np.zeros(2)

    # ── Global slope ──────────────────────────────────────────────────────────
    grad += slope_grad(own_team, game_state)

    # ── Own players ───────────────────────────────────────────────────────────
    for j, coord in enumerate(own_coords):    # iterating over players
        if j == player_idx:
            continue                          # no self-interaction
        if game_state == 'attack':
            grad += sombrero_grad(pos, coord, SOMBRERO_AMP, SOMBRERO_B)
        else:
            grad += gauss_grad(pos, coord, AMP_OWN_REPEL, SIGMA_PLAYER)

    # ── Opponent players ──────────────────────────────────────────────────────
    for coord in opp_coords:
        if game_state == 'defense':
            grad += sombrero_grad(pos, coord, SOMBRERO_AMP, SOMBRERO_B)
        elif game_state == 'neutral':
            grad += gauss_grad(pos, coord, AMP_OPP_NEUTRAL, SIGMA_PLAYER)
        else:  # attack
            grad += gauss_grad(pos, coord, AMP_OPP_ATTACK, SIGMA_PLAYER)

    # ── Ball ──────────────────────────────────────────────────────────────────
    if game_state == 'neutral':
        grad += gauss_grad(pos, ball_coords, AMP_BALL_NEUTRAL, SIGMA_BALL)
    elif game_state == 'attack':
        grad += gauss_grad(pos, ball_coords, AMP_BALL_ATTACK, SIGMA_BALL)
    # defense: ignore ball — position relative to opponents, not ball

    # ── Boundary walls ────────────────────────────────────────────────────────
    grad += boundary_grad(pos)

    return grad

 #RIGHT NOW THE PLAYER will not "see a ball through a wall of defenders" -> increase constant?


def sort_players_by_distance(own_coords: np.ndarray,
        ball_coords: np.ndarray) -> np.ndarray:
        # PRE:  own_coords shape (5, 2), ball_coords shape (2,)
        # POST: shape (5,) integer array of player indices, ascending distance to ball.
        #       Index 0 of the result = index of closest player in own_coords.
        diffs     = own_coords - ball_coords          # shape (5, 2), broadcast
        distances = np.linalg.norm(diffs, axis=1)     # shape (5,) — one distance per row
        return np.argsort(distances)                  # shape (5,) — sorted indices
#TODo: implement this algo using the fucntions provided in foosball.py (like norm)



def propose_move(pos:        np.ndarray,
                 grad:       np.ndarray,
                 ball_coords:np.ndarray) -> np.ndarray:
    # PRE:  pos shape (2,), grad shape (2,) — the raw gradient at pos.
    #       ball_coords shape (2,) — used as fallback direction only.
    # POST: candidate position = pos + MAX_RUNNING_DISTANCE · (−grad/‖grad‖)
    #       Moves full step in gradient descent direction.
    #       No constraint enforcement here — that's steps 9 and 10.
    grad_norm = np.linalg.norm(grad)

    if grad_norm < EPS:
        # Flat region — gradient gives no information.
        # Fall back: move toward ball (always a sensible default).
        direction = ball_coords - pos
        dist      = np.linalg.norm(direction)
        if dist < EPS:
            direction = np.array([1.0, 0.0])  # last resort: move right
        else:
            direction = direction / dist
    else:
        direction = -grad / grad_norm         # descend: negate and normalise

    step = DistanceLimits.MAX_RUNNING_DISTANCE - 1e-5 # Because we always move the full 10m, player might "overshoot" the perfect spot.
    return pos + direction * step

   