# Python >= 3.10
# Mathrix MindPhair 2026 — Foosball Strategy
# by NianKim
#
# Approach: Artificial Potential Fields (APF)
#   Every object on the pitch (players, ball, walls) generates a potential field.
#   Players follow the negative gradient (steepest descent) of the superposed field.
#   Three game states (attack / neutral / defense) switch which fields are active.

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


EPS = 1e-8  # numerical guard for division and distance checks

# ── Tunable constants ─────────────────────────────────────────────────────────
# First set: hand-tuned, 91% win rate vs easy_strategy
SIGMA_PLAYER     = 8.0    # spatial spread of player potentials [m]
SIGMA_BALL       = 14.0   # spatial spread of ball potential [m]
AMP_OWN_REPEL    = 2.1    # own-player repulsion (spread out)
AMP_OPP_ATTACK   = 3.5    # opponent repulsion in attack (avoid blockers)
AMP_OPP_NEUTRAL  = 1.8    # opponent repulsion in neutral
AMP_BALL_ATTACK  = -5.0   # ball attraction in attack (weak — carrier has it)
AMP_BALL_NEUTRAL = -50.0  # ball attraction in neutral (strong — go get it)
AMP_BALL_DEFENSE = -40.0  # ball attraction in defense (chase the carrier)
SOMBRERO_B       = 0.2    # first attractive ring at r ≈ π/(2b) ≈ 7.8m
SOMBRERO_AMP     = 4.0    # sombrero strength — shapes passing lanes in attack
SLOPE_WEIGHT     = 0.6    # strength of global forward/backward tilt
BOUNDARY_A       = 50.0   # wall repulsion amplitude
BOUNDARY_W       = 2.3    # wall decay width [m] — kicks in within ~3m of edge
LAT_PASS_PENALTY = 0.5   # penalty per metre of lateral distance when choosing pass target


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
#       First attractive ring at r ≈ π/(2·b). To target distance d: b = π/(2·d).
# POST: gradient of ripple potential V(r) = A·sin(b·r)/r.
#       Singularity at r=0 is removable — guarded by early return.
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
#       neutral → no tilt. attack → toward enemy goal. defense → toward own goal.
#       Team 0 attacks +x (right boundary), Team 1 attacks -x (left boundary).
def slope_grad(own_team: int, game_state: str) -> np.ndarray:
    if game_state == 'neutral':
        return np.zeros(2)
    sign = -1.0 if own_team == 0 else 1.0   # team 0: ∇V = -x → descent goes +x ✓
    if game_state == 'defense':
        sign *= -1.0
    return np.array([sign * SLOPE_WEIGHT, 0.0])


# PRE:  pos is a 2D point. Works outside pitch too — walls get steeper.
# POST: gradient pointing toward nearest wall.
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
#
#       State   | Own players | Opponents  | Ball
#       --------|-------------|------------|---------------------------
#       attack  | sombrero    | gauss +    | gauss − (weak attraction)
#       neutral | gauss +     | gauss +    | gauss − (strong attraction)
#       defense | gauss +     | sombrero   | gauss − (chase carrier)
def total_gradient(
    pos:         np.ndarray,  # 2D position of the player being computed
    own_coords:  np.ndarray,  # 5 x 2D — committed moves so far this turn
    opp_coords:  np.ndarray,  # 5 x 2D — opponent positions
    ball_coords: np.ndarray,  # 2D position of the ball
    game_state:  str,
    own_team:    int,
    player_idx:  int          # skipped in own-player loop (no self-interaction)
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
    elif game_state == 'defense':
        grad += gauss_grad(pos, ball_coords, AMP_BALL_DEFENSE, SIGMA_BALL)

    grad += boundary_grad(pos)

    return grad


# ── Movement helpers ──────────────────────────────────────────────────────────

# PRE:  own_coords holds 5 x 2D positions. ball_coords is a 2D point.
#       carrier_idx is ball.player when own team has the ball, else -1.
# POST: the 5 player indices sorted by distance to ball, closest first.
#       Carrier pinned to back so teammates commit moves before carrier passes.
def sort_players_by_distance(own_coords: np.ndarray,
                             ball_coords: np.ndarray,
                             carrier_idx: int = -1) -> np.ndarray:
    diffs     = own_coords - ball_coords
    distances = np.linalg.norm(diffs, axis=1)
    order     = np.argsort(distances)
    if carrier_idx >= 0:
        order = np.append(order[order != carrier_idx], carrier_idx)
    return order


# PRE:  pos and ball_coords are 2D points. grad is the raw gradient at pos.
# POST: candidate position one full step (MAX_RUNNING_DISTANCE) in descent direction.
#       Falls back toward ball if gradient is flat (saddle point).
#       No constraint checking — that is enforce_ball_clearance + enforce_run_distance.
def propose_move(pos:         np.ndarray,
                 grad:        np.ndarray,
                 ball_coords: np.ndarray) -> np.ndarray:
    grad_norm = np.linalg.norm(grad)
    if grad_norm < EPS:
        direction = ball_coords - pos
        dist      = np.linalg.norm(direction)
        direction = np.array([1.0, 0.0]) if dist < EPS else direction / dist
    else:
        direction = -grad / grad_norm
    return pos + direction * (DistanceLimits.MAX_RUNNING_DISTANCE - 1e-5)


# PRE:  pos_from and pos_to are 2D points. min_d >= 0, max_d > min_d.
# POST: same direction from pos_from as pos_to, distance clamped to [min_d, max_d].
#       If direction is zero, nudges along +x as fallback.
def enforce_run_distance(pos_from: np.ndarray, pos_to: np.ndarray,
                         min_d: float, max_d: float) -> np.ndarray:
    diff = pos_to - pos_from
    d    = np.linalg.norm(diff)
    if d < EPS:
        diff, d = np.array([1.0, 0.0]), 1.0
    return pos_from + (diff / d) * np.clip(d, min_d, max_d)


# PRE:  pos is a 2D point. ball_coords is the current ball position.
#       min_dist is MIN_OWN_BALL_DISTANCE (5m) or MIN_OPP_BALL_DISTANCE (4m).
#       Only call this when ball.team is not None.
# POST: if pos is within min_dist of ball, pushes it to min_dist + 0.05m radially.
#       Otherwise returns pos unchanged.
def enforce_ball_clearance(pos: np.ndarray, ball_coords: np.ndarray,
                           min_dist: float) -> np.ndarray:
    diff = pos - ball_coords
    d    = np.linalg.norm(diff)
    if d >= min_dist:
        return pos
    if d < EPS:
        diff, d = np.array([0.0, 1.0]), 1.0
    return ball_coords + (diff / d) * (min_dist + 0.05)


# PRE:  pos is a 2D point, possibly outside pitch bounds.
# POST: clipped to pitch interior with 0.05m margin from each edge.
def clamp_to_pitch(pos: np.ndarray, margin: float = 0.05) -> np.ndarray:
    return np.clip(pos,
                   [-Pitch.X_BOUND + margin, -Pitch.Y_BOUND + margin],
                   [ Pitch.X_BOUND - margin,  Pitch.Y_BOUND - margin])


# ── Ball carrier ──────────────────────────────────────────────────────────────

# PRE:  ball_coords is the carrier's current 2D position.
#       own_coords should be new_coords (committed future positions of teammates).
#       own_team is 0 or 1.
# POST: target 2D point for the carrier's move, always 3–20m from ball_coords.
#       Priority: (1) shoot if goal in range, (2) pass to most forward non-own-goal
#       teammate, (3) emergency clearance kick to open space.
#       No dribbling — carrier must move the ball every turn.
def ball_carrier_action(ball_coords: np.ndarray, own_coords: np.ndarray,
                        own_team: int) -> np.ndarray:
    sign       = 1.0 if own_team == 0 else -1.0
    goal_x     = Pitch.X_BOUND * sign
    own_goal_x = -Pitch.X_BOUND * sign
    MIN_PASS   = DistanceLimits.MIN_SHOOTING_DISTANCE + 0.05
    MAX_PASS   = DistanceLimits.MAX_SHOOTING_DISTANCE - 0.05

    # Priority 1: shoot — aim past the line to guarantee crossing
    if abs(goal_x - ball_coords[0]) <= MAX_PASS:
        target  = np.array([goal_x * 1.1, ball_coords[1]])
        to_goal = target - ball_coords
        return ball_coords + (to_goal / np.linalg.norm(to_goal)) * MAX_PASS

    # Priority 2: pass to most forward teammate in legal range
    #             skip any teammate whose position would score an own goal
    in_range = []
    for coord in own_coords:
        d = np.linalg.norm(coord - ball_coords)
        if MIN_PASS < d < MAX_PASS:
            if (coord[0] - own_goal_x) * sign < 0:
                continue                        # would land behind own goal — skip
            forward_component = coord[0] * sign
            lateral_penalty   = abs(coord[1] - ball_coords[1]) * LAT_PASS_PENALTY
            in_range.append((forward_component - lateral_penalty, coord))
    if in_range:
        in_range.sort(key=lambda x: -x[0])
        return in_range[0][1].copy()

    # Priority 3: emergency clearance — y clamped, x free to cross boundary
    for direction in [
        np.array([ sign,  0.0]),
        np.array([ sign,  0.8]),
        np.array([ sign, -0.8]),
        np.array([ 0.0,   1.0]),
        np.array([ 0.0,  -1.0]),
        np.array([-sign,  0.0]),
    ]:
        direction = direction / np.linalg.norm(direction)
        target    = ball_coords + direction * MAX_PASS
        target[1] = np.clip(target[1], -Pitch.Y_BOUND + 0.05, Pitch.Y_BOUND - 0.05)
        if np.linalg.norm(target - ball_coords) >= MIN_PASS:
            return target

    return ball_coords + np.array([sign, 0.0]) * MAX_PASS


# ── Strategy assembly ─────────────────────────────────────────────────────────

# PRE:  valid StrategyInput. Must return within TIME_LIMIT = 0.01s.
# POST: StrategyOutput with 5 committed 2D positions, all constraints satisfied.
#       state stores kickoff direction so consecutive kickoffs alternate sides.
def gradient_strategy(strat_input: StrategyInput) -> StrategyOutput:

    own_team    = strat_input.team
    opp_team    = 1 - own_team
    own_coords  = strat_input.player_coords[own_team].copy()
    opp_coords  = strat_input.player_coords[opp_team].copy()
    ball        = strat_input.ball
    ball_coords = ball.coords.copy()

    game_state  = determine_game_state(ball, own_team)
    carrier_idx = ball.player if ball.team == own_team else -1
    new_coords  = own_coords.copy()

    # ── Kickoff: pass back to own winger, alternate sides ────────────────────
    # Detected by: we have the ball, it's at centre, our carrier is at centre.
    if (ball.team == own_team
            and np.linalg.norm(ball_coords) < 1.0
            and abs(own_coords[ball.player][0]) < 1.0):
        sign = 1.0 if own_team == 0 else -1.0

        # flip y each kickoff using prev_state so opponent can't camp one side
        prev        = strat_input.prev_state
        flip        = -1.0 if (prev == 'kickoff_up') else 1.0
        kickoff_dir = np.array([-sign * 0.8, flip * 1.0])   # backward + sideways
        kickoff_dir = kickoff_dir / np.linalg.norm(kickoff_dir)

        target    = ball_coords + kickoff_dir * (DistanceLimits.MAX_SHOOTING_DISTANCE - 0.05)
        target[1] = np.clip(target[1], -Pitch.Y_BOUND + 0.05, Pitch.Y_BOUND - 0.05)
        new_coords[ball.player] = target

        order = sort_players_by_distance(own_coords, ball_coords, carrier_idx)
        for i in order:
            if i == carrier_idx:
                continue
            pos       = own_coords[i]
            grad      = total_gradient(pos, new_coords, opp_coords,
                                       target, 'neutral', own_team, i)
            candidate = propose_move(pos, grad, target)
            candidate = enforce_run_distance(pos, candidate,
                                             DistanceLimits.MIN_RUNNING_DISTANCE + 0.05,
                                             DistanceLimits.MAX_RUNNING_DISTANCE - 0.05)
            new_coords[i] = clamp_to_pitch(candidate)

        next_state = 'kickoff_down' if flip > 0 else 'kickoff_up'
        return StrategyOutput(new_coords, next_state)

    # ── Main loop ─────────────────────────────────────────────────────────────
    order = sort_players_by_distance(own_coords, ball_coords, carrier_idx)

    for i in order:

        # ball carrier: shoot or pass, no gradient
        if ball.team == own_team and i == carrier_idx:
            new_coords[i] = ball_carrier_action(ball_coords, new_coords, own_team)
            continue

        # everyone else: gradient descent + constraint chain
        pos       = own_coords[i]
        grad      = total_gradient(pos, new_coords, opp_coords,
                                   ball_coords, game_state, own_team, i)
        candidate = propose_move(pos, grad, ball_coords)

        # if ball is free and reachable this turn, stop at the ball not past it
        if ball.team is None:
            dist_to_ball = np.linalg.norm(pos - ball_coords)
            if dist_to_ball < DistanceLimits.MAX_RUNNING_DISTANCE:
                candidate = ball_coords.copy()

        if ball.team is not None:
            min_dist  = (DistanceLimits.MIN_OWN_BALL_DISTANCE
                         if ball.team == own_team
                         else DistanceLimits.MIN_OPP_BALL_DISTANCE)
            candidate = enforce_ball_clearance(candidate, ball_coords, min_dist)

        candidate     = enforce_run_distance(pos, candidate,
                                             DistanceLimits.MIN_RUNNING_DISTANCE + 0.05,
                                             DistanceLimits.MAX_RUNNING_DISTANCE - 0.05)
        new_coords[i] = clamp_to_pitch(candidate)

    return StrategyOutput(new_coords, game_state)
