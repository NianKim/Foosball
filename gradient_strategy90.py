import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from foosball import Ball, Pitch, DistanceLimits, StrategyInput, StrategyOutput

EPS = 1e-8  # numerical guard for division and distance checks

# ── Tunable constants ─────────────────────────────────────────────────────────
SIGMA_PLAYER     = 8.0    # spatial spread of player potentials [m]
SIGMA_BALL       = 14.0   # spatial spread of ball potential [m]
AMP_OWN_REPEL    = 2.1    # own-player repulsion (spread out)
AMP_OPP_ATTACK   = 3.5    # opponent repulsion in attack (avoid blockers)
AMP_OPP_NEUTRAL  = 1.8    # opponent repulsion in neutral
AMP_BALL_ATTACK  = -5.0   # ball attraction in attack (weak — carrier has it)
AMP_BALL_NEUTRAL = -50.0  # ball attraction in neutral (strong — go get it)
AMP_BALL_DEFENSE = -40.0  # ball attraction in defense (chase the carrier)
SOMBRERO_B       = 0.2    # first attractive ring at r ≈ π/(2·0.22) ≈ 7m
SOMBRERO_AMP     = 4.0    # sombrero strength — shapes passing lanes in attack
SLOPE_WEIGHT     = 0.6    # strength of global forward/backward tilt
BOUNDARY_A       = 50.0   # wall repulsion amplitude
BOUNDARY_W       = 2.3    # wall decay width [m] — kicks in within ~3m of edge

LAT_PASS_PENALTY = 0.5    #inside ball_carrier (was 0.45 for 88 percent)

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
#       Team 0 attacks +x, Team 1 attacks -x.
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
#       See strategy table in README.md.
#
#       State   | Own players | Opponents  | Ball
#       --------|-------------|------------|---------------------------
#       attack  | sombrero    | gauss +    | gauss − (weak attraction)
#       neutral | gauss +     | gauss +    | gauss − (strong attraction)
#       defense | gauss +     | sombrero   | gauss − (chase carrier)
def total_gradient(
    pos:         np.ndarray,
    own_coords:  np.ndarray,
    opp_coords:  np.ndarray,
    ball_coords: np.ndarray,
    game_state:  str,
    own_team:    int,
    player_idx:  int
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
#       Priority: (1) shoot if goal in range, (2) pass to most forward teammate,
#       (3) emergency clearance kick to open space.
#       No dribbling — carrier must move the ball every turn.
def ball_carrier_action(ball_coords: np.ndarray, own_coords: np.ndarray,
                        own_team: int) -> np.ndarray:
    sign     = 1.0 if own_team == 0 else -1.0
    goal_x   = Pitch.X_BOUND * sign
    own_goal_x = -Pitch.X_BOUND * sign
    MIN_PASS = DistanceLimits.MIN_SHOOTING_DISTANCE + 0.05
    MAX_PASS = DistanceLimits.MAX_SHOOTING_DISTANCE - 0.05

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
                continue                       # would land behind own goal — skip
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

# ── Kickoff: alternate direction each kickoff using prev_state ────────────
    if (ball.team == own_team
            and np.linalg.norm(ball_coords) < 1.0
            and abs(own_coords[ball.player][0]) < 1.0):
        sign = 1.0 if own_team == 0 else -1.0

        # flip y direction each kickoff, always pass BACK to own winger
        prev  = strat_input.prev_state
        flip  = -1.0 if (prev == 'kickoff_up') else 1.0

        kickoff_dir = np.array([-sign * 0.8, flip * 1.0])   # backward + sideways
        kickoff_dir = kickoff_dir / np.linalg.norm(kickoff_dir)

        ickoff_dir = np.array([sign * 1.2, flip * 0.8])
        kickoff_dir = kickoff_dir / np.linalg.norm(kickoff_dir)
        target      = ball_coords + kickoff_dir * (DistanceLimits.MAX_SHOOTING_DISTANCE - 0.05)
        target[1]   = np.clip(target[1], -Pitch.Y_BOUND + 0.05, Pitch.Y_BOUND - 0.05)
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

        # store direction so next kickoff goes the other way
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


def benchmark(strategy, n=200):
    state   = SessionState(kickoff_team=0)
    times   = []
    for turn in range(n):
        inp   = state.get_strategy_input(0, state.strategy_states[0])
        start = time.perf_counter()
        strategy(inp)
        times.append(time.perf_counter() - start)
        state.perform_iteration([strategy, easy_strategy], seed=turn)
    times = np.array(times)
    print(f"mean: {times.mean()*1000:.3f}ms  "
          f"max: {times.max()*1000:.3f}ms  "
          f"limit: 10.000ms  "
          f"{'OK' if times.max() < 0.01 else 'OVER LIMIT'}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    from foosball import SessionState, easy_strategy
    benchmark(gradient_strategy)

    strategies = [gradient_strategy, easy_strategy]

    frames  = []
    points  = [0, 0]
    state   = SessionState(kickoff_team=0)

    for turn in range(1000):
        frames.append((
            state.player_coords[0].copy(),
            state.player_coords[1].copy(),
            state.ball.coords.copy(),
            f"Turn {turn}  |  Blue {points[0]} – {points[1]} Red"
        ))
        winner = state.perform_iteration(strategies, seed=turn)
        if winner in (0, 1):
            points[winner] += 1
            frames.append((
                state.player_coords[0].copy(),
                state.player_coords[1].copy(),
                state.ball.coords.copy(),
                f"GOAL — {'Blue' if winner==0 else 'Red'} scores!  "
                f"|  Blue {points[0]} – {points[1]} Red"
            ))
            state = SessionState(kickoff_team=1 - winner)

    # animation setup as before...
    state  = SessionState(kickoff_team=0)
    points = [0, 0]
    for turn in range(2000):
        winner = state.perform_iteration(strategies, seed=turn)
        if winner in (0, 1):
            points[winner] += 1
            state = SessionState(kickoff_team=1 - winner)
    total = points[0] + points[1]
    print(f"Final: {points[0]}-{points[1]}  ({round(100*points[0]/total)}% win rate)")
    # ── Animation ─────────────────────────────────────────────────────────────
    s0, s1 = gradient_strategy, easy_strategy
    frames  = []
    points  = [0, 0]
    state   = SessionState(kickoff_team=0)

    for turn in range(1000):
        frames.append((
            state.player_coords[0].copy(),
            state.player_coords[1].copy(),
            state.ball.coords.copy(),
            f"Turn {turn}  |  Blue {points[0]} – {points[1]} Red"
        ))
        winner = state.perform_iteration([s0, s1], seed=turn)
        if winner in (0, 1):
            points[winner] += 1
            frames.append((
                state.player_coords[0].copy(),
                state.player_coords[1].copy(),
                state.ball.coords.copy(),
                f"GOAL — {'Blue' if winner == 0 else 'Red'} scores!  "
                f"|  Blue {points[0]} – {points[1]} Red"
            ))
            state = SessionState(kickoff_team=1 - winner)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-50, 50)
    ax.set_ylim(-25, 25)
    ax.set_facecolor('#4a7c3f')
    ax.axvline(0,   color='white', linewidth=0.8, linestyle='--')
    ax.axvline(-50, color='white', linewidth=1.5)
    ax.axvline( 50, color='white', linewidth=1.5)

    scat0  = ax.scatter([], [], s=150, color='blue',  zorder=3, label='Team 0 (gradient)')
    scat1  = ax.scatter([], [], s=150, color='red',   zorder=3, label='Team 1 (easy)')
    ball_s = ax.scatter([], [], s=200, color='white', zorder=4,
                        edgecolors='black', linewidths=2)
    title  = ax.set_title('')
    ax.legend(loc='upper right')
    plt.tight_layout()

    ctrl = {'paused': False, 'frame': 0}

    def update(f):
        ctrl['frame'] = f
        coords0, coords1, ball_coords, label = frames[f]
        scat0.set_offsets(coords0)
        scat1.set_offsets(coords1)
        ball_s.set_offsets([ball_coords])
        title.set_text(label)
        return scat0, scat1, ball_s, title

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=200, blit=True, repeat=False)

    from matplotlib.widgets import Button
    ax_pause  = fig.add_axes([0.4,  0.01, 0.1,  0.05])
    ax_skip   = fig.add_axes([0.52, 0.01, 0.2,  0.05])
    btn_pause = Button(ax_pause, 'Pause')
    btn_skip  = Button(ax_skip,  'Skip 50 frames')

    def on_pause(event):
        if ctrl['paused']:
            ani.resume()
            btn_pause.label.set_text('Pause')
        else:
            ani.pause()
            btn_pause.label.set_text('Resume')
        ctrl['paused'] = not ctrl['paused']
        fig.canvas.draw_idle()

    def on_skip(event):
        target = min(ctrl['frame'] + 50, len(frames) - 1)
        ani.pause()
        ctrl['paused'] = True
        btn_pause.label.set_text('Resume')
        update(target)
        fig.canvas.draw_idle()

    btn_pause.on_clicked(on_pause)
    btn_skip.on_clicked(on_skip)
    plt.show()