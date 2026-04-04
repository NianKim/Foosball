import numpy as np
from foosball import Ball, Pitch, DistanceLimits

EPS = 1e-8  # numerical guard for division and distance checks

# ── Tunable constants ─────────────────────────────────────────────────────────
SIGMA_PLAYER     = 8.0    # spatial spread of player potentials [m]
SIGMA_BALL       = 20.0   # spatial spread of ball potential [m]
AMP_OWN_REPEL    = 2.0    # own-player repulsion (spread out)
AMP_OPP_ATTACK   = 3.0    # opponent repulsion in attack (avoid blockers)
AMP_OPP_NEUTRAL  = 2.5    # opponent repulsion in neutral
AMP_BALL_ATTACK  = -3.0   # ball attraction in attack (weak — carrier has it)
AMP_BALL_NEUTRAL = -20.0  # ball attraction in neutral (strong — go get it)
SOMBRERO_B       = 0.22   # ring distance ≈ π/(2·0.22) ≈ 7m
SOMBRERO_AMP     = 1.5    # sombrero strength — tune relative to ball attraction
SLOPE_WEIGHT     = 0.3    # strength of global forward/backward tilt
BOUNDARY_A       = 80.0   # wall repulsion amplitude
BOUNDARY_W       = 3.0    # wall decay width [m] — kicks in within ~3m of edge


# ── Mathematical functions ────────────────────────────────────────────────────

# PRE:  ball.team is 0, 1, or None. own_team is 0 or 1.
# POST: returns 'attack' | 'neutral' | 'defense'
def determine_game_state(ball, own_team: int) -> str:
    if ball.team == own_team:
        return 'attack'
    elif ball.team is None:
        return 'neutral'
    else:
        return 'defense'


# PRE:  pos and center are 2D points on the pitch. sigma > 0.
# POST: gradient of a bell-shaped potential centered at center.
#       amplitude > 0 → repulsive hill, amplitude < 0 → attractive well.
def gauss_grad(pos: np.ndarray, center: np.ndarray,
               amplitude: float, sigma: float) -> np.ndarray:
    diff = pos - center
    r2   = np.dot(diff, diff)
    V    = amplitude * np.exp(-r2 / (2.0 * sigma**2))
    return V * (-diff / sigma**2)


# PRE:  pos and center are 2D points. b > 0.
#       First attractive ring sits at r ≈ π/(2·b).
#       To target a specific ring distance d: set b = π / (2·d).
# POST: gradient of a ripple-shaped potential V(r) = A·sin(b·r)/r.
#       Singularity at r=0 is mathematically removable — guarded by early return.
def sombrero_grad(pos: np.ndarray, center: np.ndarray,
                  amplitude: float, b: float) -> np.ndarray:
    diff  = pos - center
    r     = np.sqrt(np.dot(diff, diff))
    if r < EPS:
        return np.zeros(2)
    r_hat = diff / r
    dV_dr = amplitude * (b * np.cos(b * r) / r - np.sin(b * r) / r**2)
    return dV_dr * r_hat


# PRE:  own_team is 0 or 1. game_state is 'attack', 'neutral', or 'defense'.
# POST: constant gradient tilting the whole field in one direction.
#       neutral  → no tilt.
#       attack   → pulls players toward the enemy goal.
#       defense  → pulls players back toward their own goal.
def slope_grad(own_team: int, game_state: str) -> np.ndarray:
    if game_state == 'neutral':
        return np.zeros(2)
    sign = -1.0 if own_team == 0 else 1.0
    if game_state == 'defense':
        sign *= -1.0
    return np.array([sign * SLOPE_WEIGHT, 0.0])


# PRE:  pos is a 2D point. Works outside the pitch too — walls get steeper.
# POST: gradient pointing toward the nearest wall.
#       Descent direction (negative of this) pushes players away from edges.
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


# ── Field composition ─────────────────────────────────────────────────────────

# PRE:  own_coords must be the committed-moves array (new_coords), not the
#       original snapshot — later players see earlier players' decided positions.
#       game_state is 'attack', 'neutral', or 'defense'.
# POST: summed gradient at pos from all field sources.
#       Caller must negate to get the descent (movement) direction.
#       See strategy table in README.md
def total_gradient(
    pos:         np.ndarray,  # 2D position of the player being computed
    own_coords:  np.ndarray,  # 5 x 2D positions — committed moves so far
    opp_coords:  np.ndarray,  # 5 x 2D positions — opponents
    ball_coords: np.ndarray,  # 2D position of the ball
    game_state:  str,
    own_team:    int,
    player_idx:  int          # this player's index — skipped in own-player loop
) -> np.ndarray:

    grad = np.zeros(2)

    grad += slope_grad(own_team, game_state)

    for j, coord in enumerate(own_coords):
        if j == player_idx:
            continue
        if game_state == 'attack':
            grad += sombrero_grad(pos, coord, SOMBRERO_AMP, SOMBRERO_B)
        else:
            grad += gauss_grad(pos, coord, AMP_OWN_REPEL, SIGMA_PLAYER)

    for coord in opp_coords:
        if game_state == 'defense':
            grad += sombrero_grad(pos, coord, SOMBRERO_AMP, SOMBRERO_B)
        elif game_state == 'neutral':
            grad += gauss_grad(pos, coord, AMP_OPP_NEUTRAL, SIGMA_PLAYER)
        else:
            grad += gauss_grad(pos, coord, AMP_OPP_ATTACK, SIGMA_PLAYER)

    if game_state == 'neutral':
        grad += gauss_grad(pos, ball_coords, AMP_BALL_NEUTRAL, SIGMA_BALL)
    elif game_state == 'attack':
        grad += gauss_grad(pos, ball_coords, AMP_BALL_ATTACK, SIGMA_BALL)

    grad += boundary_grad(pos)

    return grad


# PRE:  own_coords holds 5 x 2D positions. ball_coords is a 2D point.
# POST: the 5 player indices sorted by distance to ball, closest first.
def sort_players_by_distance(own_coords: np.ndarray,
                             ball_coords: np.ndarray) -> np.ndarray:
    diffs     = own_coords - ball_coords      # one difference vector per player
    distances = np.linalg.norm(diffs, axis=1) # one scalar distance per player
    return np.argsort(distances)


# PRE:  pos and ball_coords are 2D points. grad is the raw gradient at pos.
# POST: candidate position one full step (MAX_RUNNING_DISTANCE) away from pos
#       in the gradient descent direction.
#       If gradient is flat, falls back toward the ball instead.
#       No constraint checking — that is steps 9 and 10.
def propose_move(pos:         np.ndarray,
                 grad:        np.ndarray,
                 ball_coords: np.ndarray) -> np.ndarray:
    grad_norm = np.linalg.norm(grad)

    if grad_norm < EPS:
        direction = ball_coords - pos
        dist      = np.linalg.norm(direction)
        if dist < EPS:
            direction = np.array([1.0, 0.0])
        else:
            direction = direction / dist
    else:
        direction = -grad / grad_norm

    step = DistanceLimits.MAX_RUNNING_DISTANCE - 1e-5
    return pos + direction * step


# PRE:  pos_from and pos_to are 2D points. min_d >= 0, max_d > min_d.
# POST: a point at the same direction from pos_from as pos_to,
#       but with distance clamped to [min_d, max_d].
#       If pos_to == pos_from (zero direction), nudges along +x as fallback.
def enforce_run_distance(pos_from: np.ndarray, pos_to: np.ndarray,
                         min_d: float, max_d: float) -> np.ndarray:
    diff = pos_to - pos_from
    d    = np.linalg.norm(diff)

    if d < EPS:
        diff = np.array([1.0, 0.0])  # no direction info — nudge along +x
        d    = 1.0

    d_clamped = np.clip(d, min_d, max_d)
    return pos_from + (diff / d) * d_clamped

# PRE:  pos is a 2D point. ball_coords is the current ball position.
#       min_dist is MIN_OWN_BALL_DISTANCE or MIN_OPP_BALL_DISTANCE.
#       Only call this when ball.team is not None.
# POST: if pos is within min_dist of the ball, pushes it to min_dist + small buffer
#       along the same radial direction. Otherwise returns pos unchanged.
def enforce_ball_clearance(pos: np.ndarray, ball_coords: np.ndarray,
                           min_dist: float) -> np.ndarray:
    diff = pos - ball_coords
    d    = np.linalg.norm(diff)

    if d >= min_dist:
        return pos                           # already clear — nothing to do

    if d < EPS:
        diff = np.array([0.0, 1.0])          # directly on ball — push sideways
        d    = 1.0

    return ball_coords + (diff / d) * (min_dist + 0.05)

