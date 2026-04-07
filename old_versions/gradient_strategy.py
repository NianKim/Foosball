import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from foosball import Ball, Pitch, DistanceLimits, StrategyInput, StrategyOutput

EPS = 1e-8

# ── Tunable constants ─────────────────────────────────────────────────────────
SIGMA_PLAYER     = 8.0
SIGMA_BALL       = 10.0
AMP_OWN_REPEL    = 0.8
AMP_OPP_ATTACK   = 3.0
AMP_OPP_NEUTRAL  = 0.8
AMP_BALL_ATTACK  = -3.0
AMP_BALL_NEUTRAL = -50.0
AMP_BALL_DEFENSE = -40.0
SOMBRERO_B       = 0.22
SOMBRERO_AMP     = 3.0
SLOPE_WEIGHT     = 0.15
BOUNDARY_A       = 80.0
BOUNDARY_W       = 3.0


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


# PRE:  pos and center are 2D points. sigma > 0.
# POST: gradient of a bell-shaped potential. amplitude > 0 repels, < 0 attracts.
def gauss_grad(pos: np.ndarray, center: np.ndarray,
               amplitude: float, sigma: float) -> np.ndarray:
    diff = pos - center
    r2   = np.dot(diff, diff)
    V    = amplitude * np.exp(-r2 / (2.0 * sigma**2))
    return V * (-diff / sigma**2)


# PRE:  pos and center are 2D points. b > 0.
#       First attractive ring at r ≈ π/(2·b). For d metres: b = π/(2·d).
# POST: gradient of ripple potential V(r) = A·sin(b·r)/r.
def sombrero_grad(pos: np.ndarray, center: np.ndarray,
                  amplitude: float, b: float) -> np.ndarray:
    diff  = pos - center
    r     = np.sqrt(np.dot(diff, diff))
    if r < EPS:
        return np.zeros(2)
    r_hat = diff / r
    dV_dr = amplitude * (b * np.cos(b * r) / r - np.sin(b * r) / r**2)
    return dV_dr * r_hat


# PRE:  own_team in {0,1}. game_state in {'attack','neutral','defense'}.
# POST: constant gradient. neutral→zero. attack→toward enemy goal. defense→own goal.
def slope_grad(own_team: int, game_state: str) -> np.ndarray:
    if game_state == 'neutral':
        return np.zeros(2)
    sign = -1.0 if own_team == 0 else 1.0
    if game_state == 'defense':
        sign *= -1.0
    return np.array([sign * SLOPE_WEIGHT, 0.0])


# PRE:  pos is a 2D point.
# POST: gradient pointing toward nearest wall. Descent pushes away from edges.
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

# PRE:  own_coords is the committed-moves array (new_coords), not original snapshot.
# POST: summed gradient at pos. Caller negates to get descent direction.
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

    # global slope
    grad += slope_grad(own_team, game_state)

    # own players — repel or sombrero depending on state
    for j, coord in enumerate(own_coords):
        if j == player_idx:
            continue
        if game_state == 'attack':
            grad += sombrero_grad(pos, coord, SOMBRERO_AMP, SOMBRERO_B)
        else:
            grad += gauss_grad(pos, coord, AMP_OWN_REPEL, SIGMA_PLAYER)

    # opponent players — each opponent's position, not the ball
    for coord in opp_coords:
        if game_state == 'defense':
            grad += sombrero_grad(pos, coord, SOMBRERO_AMP, SOMBRERO_B)
        elif game_state == 'neutral':
            grad += gauss_grad(pos, coord, AMP_OPP_NEUTRAL, SIGMA_PLAYER)
        else:  # attack
            grad += gauss_grad(pos, coord, AMP_OPP_ATTACK, SIGMA_PLAYER)

    # ball attraction — once, outside all loops
    if game_state == 'neutral':
        grad += gauss_grad(pos, ball_coords, AMP_BALL_NEUTRAL, SIGMA_BALL)
    elif game_state == 'attack':
        grad += gauss_grad(pos, ball_coords, AMP_BALL_ATTACK, SIGMA_BALL)
    elif game_state == 'defense':
        grad += gauss_grad(pos, ball_coords, AMP_BALL_DEFENSE, SIGMA_BALL)

    # boundary walls
    grad += boundary_grad(pos)

    return grad


# PRE:  own_coords holds 5 x 2D positions. carrier_idx = ball.player or -1.
# POST: player indices sorted by distance to ball, carrier pinned last.
def sort_players_by_distance(own_coords: np.ndarray,
                             ball_coords: np.ndarray,
                             carrier_idx: int = -1) -> np.ndarray:
    diffs     = own_coords - ball_coords
    distances = np.linalg.norm(diffs, axis=1)
    order     = np.argsort(distances)
    if carrier_idx >= 0:
        order = np.append(order[order != carrier_idx], carrier_idx)
    return order


# PRE:  pos and ball_coords are 2D points. grad is raw gradient at pos.
# POST: candidate position one full step in descent direction. No constraints.
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


# PRE:  min_d >= 0, max_d > min_d.
# POST: same direction from pos_from, distance clamped to [min_d, max_d].
def enforce_run_distance(pos_from: np.ndarray, pos_to: np.ndarray,
                         min_d: float, max_d: float) -> np.ndarray:
    diff = pos_to - pos_from
    d    = np.linalg.norm(diff)
    if d < EPS:
        diff, d = np.array([1.0, 0.0]), 1.0
    return pos_from + (diff / d) * np.clip(d, min_d, max_d)


# PRE:  Only call when ball.team is not None.
# POST: pushes pos to min_dist + buffer if too close to ball.
def enforce_ball_clearance(pos: np.ndarray, ball_coords: np.ndarray,
                           min_dist: float) -> np.ndarray:
    diff = pos - ball_coords
    d    = np.linalg.norm(diff)
    if d >= min_dist:
        return pos
    if d < EPS:
        diff, d = np.array([0.0, 1.0]), 1.0
    return ball_coords + (diff / d) * (min_dist + 0.05)


# PRE:  pos is a 2D point, possibly outside pitch.
# POST: clipped to pitch interior with 0.05m margin.
def clamp_to_pitch(pos: np.ndarray, margin: float = 0.05) -> np.ndarray:
    return np.clip(pos,
                   [-Pitch.X_BOUND + margin, -Pitch.Y_BOUND + margin],
                   [ Pitch.X_BOUND - margin,  Pitch.Y_BOUND - margin])


# ── Ball carrier ──────────────────────────────────────────────────────────────

# PRE:  own_coords should be new_coords (committed future positions of teammates).
# POST: target 2D point 3–20m from ball_coords.
#       Priority: shoot > pass forward > emergency clearance.
def ball_carrier_action(ball_coords: np.ndarray, own_coords: np.ndarray,
                        own_team: int) -> np.ndarray:
    sign   = 1.0 if own_team == 0 else -1.0
    goal_x = Pitch.X_BOUND * sign
    MIN_PASS = DistanceLimits.MIN_SHOOTING_DISTANCE + 0.05
    MAX_PASS = DistanceLimits.MAX_SHOOTING_DISTANCE - 0.05

    # Priority 1: shoot
    if abs(goal_x - ball_coords[0]) <= MAX_PASS:
        target  = np.array([goal_x * 1.1, ball_coords[1]])
        to_goal = target - ball_coords
        return ball_coords + (to_goal / np.linalg.norm(to_goal)) * MAX_PASS

    # Priority 2: pass to most forward teammate in range
    in_range = []
    for coord in own_coords:
        d = np.linalg.norm(coord - ball_coords)
        if MIN_PASS < d < MAX_PASS:
            in_range.append((coord[0] * sign, coord))
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

#temp

# ── Goalie constants ──────────────────────────────────────────────────────────
GOALIE_X       = 35.0   # distance from own goal to hold position [m]
GOALIE_X_RANGE = 10.0   # how far forward goalie ventures when ball is in opp half

# PRE:  ball_coords is current ball position. own_team is 0 or 1.
# POST: target 2D position for goalie — near own goal, tracking ball's y.
#       Advances slightly when ball is in opponent's half.
def goalie_target(ball_coords: np.ndarray, own_team: int) -> np.ndarray:
    sign = 1.0 if own_team == 0 else -1.0

    ball_in_own_half = (ball_coords[0] * sign) < 0
    if ball_in_own_half:
        target_x = -GOALIE_X * sign          # hold deep near own goal
    else:
        target_x = -(GOALIE_X - GOALIE_X_RANGE) * sign   # push up slightly

    target_y = np.clip(ball_coords[1], -Pitch.Y_BOUND + 2.0, Pitch.Y_BOUND - 2.0)
    return np.array([target_x, target_y])

def easy_strategy_with_goalie(strat_input: StrategyInput) -> StrategyOutput:
    # run easy_strategy normally first
    result = easy_strategy(strat_input)
    
    own_team    = strat_input.team
    own_coords  = strat_input.player_coords[own_team].copy()
    ball        = strat_input.ball
    ball_coords = ball.coords.copy()

    # overwrite goalie position(s) with positional target
    for goalie_i in [3, 4]:   # change to [4] for one goalie, [3, 4] for two
        target    = goalie_target(ball_coords, own_team)
        
        # second goalie holds slightly further back
        if goalie_i == 3:
            sign     = 1.0 if own_team == 0 else -1.0
            target   = target.copy()
            target[0] = np.clip(target[0], 
                                (-Pitch.X_BOUND + 5) * sign,
                                (-Pitch.X_BOUND + 20) * sign)

        candidate = enforce_run_distance(own_coords[goalie_i], target,
                                         DistanceLimits.MIN_RUNNING_DISTANCE + 0.05,
                                         DistanceLimits.MAX_RUNNING_DISTANCE - 0.05)
        if ball.team is not None:
            min_dist  = (DistanceLimits.MIN_OWN_BALL_DISTANCE
                         if ball.team == own_team
                         else DistanceLimits.MIN_OPP_BALL_DISTANCE)
            candidate = enforce_ball_clearance(candidate, ball_coords, min_dist)
        
        result.coords[goalie_i] = clamp_to_pitch(candidate)

    return result

def gradient_strategy_with_goalie(strat_input: StrategyInput) -> StrategyOutput:
    result      = gradient_strategy(strat_input)
    own_team    = strat_input.team
    own_coords  = strat_input.player_coords[own_team].copy()
    ball        = strat_input.ball
    ball_coords = ball.coords.copy()

    goalie_i  = 4
    target    = goalie_target(ball_coords, own_team)
    candidate = enforce_run_distance(own_coords[goalie_i], target,
                                     DistanceLimits.MIN_RUNNING_DISTANCE + 0.05,
                                     DistanceLimits.MAX_RUNNING_DISTANCE - 0.05)
    if ball.team is not None:
        min_dist  = (DistanceLimits.MIN_OWN_BALL_DISTANCE
                     if ball.team == own_team
                     else DistanceLimits.MIN_OPP_BALL_DISTANCE)
        candidate = enforce_ball_clearance(candidate, ball_coords, min_dist)
    result.coords[goalie_i] = clamp_to_pitch(candidate)
    return result


# ── Strategy assembly ─────────────────────────────────────────────────────────

# PRE:  valid StrategyInput. Must return within TIME_LIMIT = 0.01s.
# POST: StrategyOutput with 5 committed positions, all constraints satisfied.
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

    # ── Kickoff: ball at centre, send diagonally to avoid team 1 at (25,0) ──
    if (ball.team == own_team
            and np.linalg.norm(ball_coords) < 1.0
            and abs(own_coords[ball.player][0]) < 1.0):
        sign        = 1.0 if own_team == 0 else -1.0
        kickoff_dir = np.array([sign * 0.6, 1.0])
        kickoff_dir = kickoff_dir / np.linalg.norm(kickoff_dir)
        target      = ball_coords + kickoff_dir * (DistanceLimits.MAX_SHOOTING_DISTANCE - 0.05)
        target[1]   = np.clip(target[1], -Pitch.Y_BOUND + 0.05, Pitch.Y_BOUND - 0.05)
        new_coords[ball.player] = target
        # move other players toward the kickoff target so they're ready next turn
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
        return StrategyOutput(new_coords, 'attack')  # early return — skip main loop

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


if __name__ == '__main__':


    import time

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
    benchmark(gradient_strategy)


    from foosball import SessionState, easy_strategy

    combos = [
        ("gradient vs easy",               gradient_strategy,             easy_strategy),
        ("gradient vs easy+goalie",        gradient_strategy,             easy_strategy_with_goalie),
        ("gradient+goalie vs easy",        gradient_strategy_with_goalie, easy_strategy),
        ("gradient+goalie vs easy+goalie", gradient_strategy_with_goalie, easy_strategy_with_goalie),
    ]

    for name, s0, s1 in combos:
        state  = SessionState(kickoff_team=0)
        points = [0, 0]
        for turn in range(2000):
            winner = state.perform_iteration([s0, s1], seed=turn)
            if winner in (0, 1):
                points[winner] += 1
                state = SessionState(kickoff_team=1 - winner)
        print(f"{name}: {points[0]}-{points[1]}")

    # ── Animation — uses last combo's strategies ───────────────────────────
    s0, s1 = gradient_strategy_with_goalie, easy_strategy_with_goalie
    frames = []
    points = [0, 0]
    state  = SessionState(kickoff_team=0)

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
                f"GOAL — {'Blue' if winner == 0 else 'Red'} scores!  |  Blue {points[0]} – {points[1]} Red"
            ))
            state = SessionState(kickoff_team=1 - winner)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-50, 50)
    ax.set_ylim(-25, 25)
    ax.set_facecolor('#4a7c3f')
    ax.axvline(0,   color='white', linewidth=0.8, linestyle='--')
    ax.axvline(-50, color='white', linewidth=1.5)
    ax.axvline( 50, color='white', linewidth=1.5)

    scat0  = ax.scatter([], [], s=150, color='blue',  zorder=3, label='Team 0 (gradient+goalie)')
    scat1  = ax.scatter([], [], s=150, color='red',   zorder=3, label='Team 1 (easy+goalie)')
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
    ax_pause  = fig.add_axes([0.4,  0.01, 0.1, 0.05])
    ax_skip   = fig.add_axes([0.52, 0.01, 0.2, 0.05])
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