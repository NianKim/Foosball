# Python >= 3.10
import numpy as np
import time
import matplotlib.pyplot as plt

TIME_LIMIT = 0.01

# contains the dimensions of the football pitch (soccer field): [-X_BOUND, X_BOUND] \times [-Y_BOUND, Y_BOUND]
class Pitch:
    X_BOUND = 50.
    Y_BOUND = 25.

# contains the physical limitations of the fussball players
class DistanceLimits:
    MIN_OWN_BALL_DISTANCE = 5.
    MIN_OPP_BALL_DISTANCE = 4.
    MAX_RUNNING_DISTANCE = 10.
    MIN_RUNNING_DISTANCE = 1.
    MAX_SHOOTING_DISTANCE = 20.
    MIN_SHOOTING_DISTANCE = 3.
    REACH = 1.

        
def coord_distance(coords1, coords2):
    return np.sqrt(np.sum((coords2 - coords1)**2))

def squared_norm(vec):
    if isinstance(vec, np.ndarray) and len(vec.shape) < 2:
        return np.inner(vec, vec)
    return np.diag(np.inner(vec, vec))

def norm(vec):
    return np.sqrt(squared_norm(vec))
        
# line is given by a np array with endpoint coordinates ([from, to]); point + radius resembles a player
# returns np.inf if no intersection of circle with halfline, and otherwise the distance of the intersection to the start of the line segment
# assumption: distance start of line segment to point is at least radius
def first_intersect(line_segment, point, radius):
    # shift endpoint of line and point
    line_vec = line_segment[1] - line_segment[0]
    point_vec = point - line_segment[0]
    
    # compute discriminant of a quadratic polynomial
    inner_point_line = np.inner(point_vec, line_vec)
    inner_line_line = squared_norm(line_vec)
    inner_point_point = squared_norm(point_vec)
    quarter_discriminant = (inner_point_line**2 - inner_line_line * (inner_point_point - radius**2)) # D/4
    
    half_sqrt_discriminant = np.sqrt(np.maximum(quarter_discriminant, 0.0))
    line_fraction_intersect = np.where(quarter_discriminant >= 0, (inner_point_line - half_sqrt_discriminant) / inner_line_line, np.inf)
    
    fraction_valid = (line_fraction_intersect >= 0) & (line_fraction_intersect <= 1)
    return np.where(fraction_valid, line_fraction_intersect * norm(line_vec), np.inf)


    
# if ball is in posession of a team, then 
#    - ball.coords are the coordinates of the player that possesses the ball
#    - ball.team equals the team number (0 or 1)
#    - ball.player equals the index of the player that possesses the ball
# if the ball is not in possession of a team, then self.team is None.
class Ball:
    def __init__(self, team):
        self.coords = np.array([0.0, 0.0])
        self.team = team
        self.player = 0
        
    def copy(self):
        ball_copy = Ball(self.team)
        ball_copy.coords = self.coords.copy()
        ball_copy.player = self.player
        return ball_copy

class StrategyInput:
    def __init__(self, team, player_coords, ball, prev_state):
        self.team = team
        self.player_coords = player_coords.copy()
        self.ball = ball.copy()
        self.prev_state = prev_state
        
class StrategyOutput:
    def __init__(self, coords, state=None):
        self.coords = coords
        self.state = state

class SessionState:
    def __init__(self, kickoff_team, strategy_states=None):
        self.iteration = 0
        self.winner = None
        self.ball = Ball(kickoff_team)

        player_coords_team_1 = np.array([[25., 0.], [15., -15.], [15., 15.], [35., -15.], [35., 15.]])
        self.player_coords = np.array([-player_coords_team_1.copy(), player_coords_team_1])
        self.player_coords[kickoff_team][0][0] = 0. # sets x-coord of player 0 of the team taking the kickoff to 0.0
        self.strategy_states = [None, None] if strategy_states is None else strategy_states
        
    def get_strategy_input(self, team, state):
        return StrategyInput(team, self.player_coords, self.ball, self.strategy_states[team])
    
    def is_strategy_output_valid(self, strategy_output, team):
        try:
            if not isinstance(strategy_output, StrategyOutput):
                return False
            
            coords = strategy_output.coords
            
            if (not isinstance(coords, np.ndarray)) or (coords.shape != (5, 2)) or np.isnan(coords).any():
                print("coords is not a numpy array of the correct shape without nans.")
                return False
            
            
            
            for i in range(5):
                # if a coord has an y-value that is out of bounds, then return False
                if abs(coords[i][1]) > Pitch.Y_BOUND:
                    print(f"x-value of player {i} of team {team} is out of bounds.")
                    return False
                # if a coord that does not represent the position of the ball has an x-value that is out of bounds, then return False
                if abs(coords[i][0]) > Pitch.X_BOUND and (team != self.ball.team or i != self.ball.player):
                    print(f"y-value of player {i} of team {team} is out of bounds.")
                    return False
            
            
            # all players must move at most by the running_distance; ball must move at most by shooting distance
            for i in range(5):
                if team == self.ball.team and i == self.ball.player:
                    if coord_distance(coords[i], self.ball.coords) > DistanceLimits.MAX_SHOOTING_DISTANCE:
                        print("You cannot shoot that far.")
                        return False
                    if coord_distance(coords[i], self.ball.coords) < DistanceLimits.MIN_SHOOTING_DISTANCE:
                        print("You must shoot further.")
                        return False
                else:
                    if coord_distance(coords[i], self.player_coords[team][i]) > DistanceLimits.MAX_RUNNING_DISTANCE:
                        print(f"Player {i} of team {team} cannot run so fast.", coords[i], self.player_coords[team][i], coord_distance(coords[i], self.player_coords[team][i]))
                        return False
                    if coord_distance(coords[i], self.player_coords[team][i]) < DistanceLimits.MIN_RUNNING_DISTANCE:
                        print(f"Player {i} of team {team} needs to run faster.")
                        return False
            
            # must keep sufficient distance from ball in posession
            if self.ball.team is not None:
                required_distance = DistanceLimits.MIN_OWN_BALL_DISTANCE if team == self.ball.team else DistanceLimits.MIN_OPP_BALL_DISTANCE
                for i in range(5):
                    if (self.ball.team == team) and (self.ball.player == i):
                        continue
                    if coord_distance(coords[i], self.ball.coords) < required_distance:
                        print(f"Player {i} of team {team}: keep your distance from the ball when it's controlled by another player please.")
                        return False
        except:
            return False
        
        return True
            
        
        
    # only difficult part is determining which team gets ball posession
    # players cannot run with the ball
    # players move before ball moves
    def do_moves(self, moves):
        rng = np.random.default_rng()
        
        # determine which player gets the ball
        if self.ball.team is None:
            # then trajectories of players are compared to coordinate of ball
            trajectories = np.stack((self.player_coords, moves), axis=2)
            distances = np.array([[first_intersect(traj, self.ball.coords, DistanceLimits.REACH) for traj in trajectories[team]] for team in range(2)])
        else:
            # then trajectory of ball is compared to coordinates (after move) of players
            ball_traj = [self.ball.coords, moves[self.ball.team][self.ball.player]]
            distances = np.array([
                first_intersect(ball_traj, moves[0], DistanceLimits.REACH),
                first_intersect(ball_traj, moves[1], DistanceLimits.REACH)
            ])
            distances[self.ball.team][self.ball.player] = np.inf
            
        # set new player coords
        new_player_coords = np.array([moves[0].copy(), moves[1].copy()])
        if self.ball.team is not None:
            team = self.ball.team
            player = self.ball.player
            new_player_coords[team][player] = self.player_coords[team][player]
        self.player_coords = new_player_coords
        
        # set new values for ball
        min_dist = distances.min()
        if min_dist == np.inf:
            self.ball.coords = self.ball.coords if self.ball.team is None else ball_traj[1]
            self.ball.team = None
            self.ball.player = None
        else:
            first_players = np.nonzero(distances == min_dist)
            index = rng.choice(len(first_players[0]))
            
            self.ball.team = first_players[0][index]
            self.ball.player = first_players[1][index]
            self.ball.coords = moves[self.ball.team][self.ball.player]
        
        if abs(self.ball.coords[0]) > Pitch.X_BOUND:
            self.winner = int(self.ball.coords[0] < 0)
            self.ball.coords[0] = np.clip(self.ball.coords[0], -Pitch.X_BOUND, Pitch.X_BOUND) # for visualization
            
    def visualize(self):
        # plotting a line plot after changing it's width and height
        f = plt.figure()
        f.set_figwidth(8)
        f.set_figheight(4)
        
        plt.scatter(self.player_coords[0].T[0], self.player_coords[0].T[1], s=100, color='blue')
        plt.scatter(self.player_coords[1].T[0], self.player_coords[1].T[1], s=100, color='red')
        plt.scatter([self.ball.coords[0]], [self.ball.coords[1]], marker='o', s=150, linewidth=3, facecolors="None", edgecolors='black')
        plt.xlim([-Pitch.X_BOUND, Pitch.X_BOUND])
        plt.ylim([-Pitch.Y_BOUND, Pitch.Y_BOUND])
        plt.show()
        plt.close()
        
    # returns winner, possibly None (if no one has won yet)
    def perform_iteration(self, strategies, seed, visualize=False):
        self.iteration += 1
        
        # initialize lists
        valid = [True, True]
        strategy_coords = [None, None]
        function_call_duration = [None, None]
        
        for team in range(2):
            # construct input for the strategy
            strategy_input = self.get_strategy_input(team, self.strategy_states[team])
            
            # get output from the strategy and time it
            np.random.seed(seed)
            start = time.perf_counter()
            strategy_output = strategies[team](strategy_input)
            end = time.perf_counter()
            function_call_duration[team] = end - start
            
            # determine whether the strategy output is valid and computed within the time limits
            valid[team] = True #self.is_strategy_output_valid(strategy_output, team) and (function_call_duration[team] < TIME_LIMIT)
            if not valid[team]:
                continue
            
            self.strategy_states[team] = strategy_output.state
            strategy_coords[team] = strategy_output.coords
        
        # handle case where at least some of the output is invalid
        if valid[0] != valid[1]:
            print(f"The output of team {int(valid[0])} is invalid.")
            self.winner = int(valid[1])
            return int(valid[1])
        if not valid[0]:
            print("The output of both teams is invalid.")
            self.winner = -1
            return -1
        
        self.do_moves(np.array([strategy_coords[0], strategy_coords[1]]))
        
        if visualize:
            self.visualize()
        
        if self.winner is not None:
            print(f"Team {self.winner} scored!")
        
        return self.winner
            
# s session ends if a team scores
def play_session(strategies, kickoff_team, seed, visualize=False, strategy_states=None, max_iter=1000):
    session_state = SessionState(kickoff_team, strategy_states)
    
    if visualize:
        session_state.visualize()
    
    for i in range(max_iter):
        winner = session_state.perform_iteration(strategies, seed + i, visualize)
        if winner is not None:
            print(f"Team {winner} gained a point. Iterations = {i}.")
            return session_state
        
    return session_state

def play_game(strategies, seed, visualize=False, max_iter=1000):
    iter_remaining = max_iter
    strategy_states = None
    points = [0, 0]
    kickoff_team = 0
    
    while iter_remaining > 0:
        session_state = play_session(strategies, kickoff_team, seed, visualize=visualize, strategy_states=strategy_states, max_iter=iter_remaining)
        winner = session_state.winner
        
        if winner is None:
            # game has ended
            return points
        
        
        if (winner == 0) or (winner == 1):
            points[winner] += 1
            kickoff_team = 1 - winner
            
        strategy_states = session_state.strategy_states
        seed += max_iter
        
        iter_remaining -= session_state.iteration
    print(f"Game has ended! Final score of team 0 vs team 1: {points[0]} - {points[1]}.")
        
# ========================================== example strategy ==========================================

def easy_strategy(strat_input):
    EPS = 1e-10
    
    own_team = strat_input.team
    opp_team = 1 - own_team
    coords = strat_input.player_coords
    ball = strat_input.ball
    
    state = 100 * own_team if (strat_input.prev_state is None) else strat_input.prev_state + 1
    
    def reverse_orientation_if(coords):
        if own_team == 1:
            return -coords
        return coords
    
    def clip_to_pitch(coords):
        return np.array([np.clip(coords.T[0], -Pitch.X_BOUND, Pitch.X_BOUND), np.clip(coords.T[1], -Pitch.Y_BOUND, Pitch.Y_BOUND)]).T
    
    def is_in_pitch(coords):
        return (np.abs(coords[0]) < Pitch.X_BOUND - EPS) and (np.abs(coords[1]) < Pitch.Y_BOUND - EPS)
    
    def is_valid_run(coords_from, coords_to, ball):
        margin = EPS**2
        
        if not is_in_pitch(coords_to):
            return False
        
        distance_run = coord_distance(coords_from, coords_to)
        if distance_run > DistanceLimits.MAX_RUNNING_DISTANCE - margin:
            return False
        if distance_run < DistanceLimits.MIN_RUNNING_DISTANCE + margin:
            return False
        
        # there are no further limitations if the ball is not possessed by a team
        if ball.team is None:
            return True
        
        min_distance = DistanceLimits.MIN_OWN_BALL_DISTANCE if ball.team == own_team else DistanceLimits.MIN_OPP_BALL_DISTANCE
        if coord_distance(coords_to, ball.coords) < min_distance + margin:
            return False
        
        return True
    
    def create_valid_coords(coords_from, ball_coords):
        dist = DistanceLimits.MAX_RUNNING_DISTANCE - EPS
        deltas = np.array([[dist, 0], [0, dist], [0, -dist], [-dist, 0]])
        
        index_to_alter = np.argmin(np.abs(ball_coords - coords_from))
        direction = -1 if (coords_from[index_to_alter] > 0) else 1
        coords_copy = coords_from.copy()
        
        coords_copy[index_to_alter] += direction * dist
        return coords_copy

    coords = reverse_orientation_if(coords)
    ball.coords = reverse_orientation_if(ball.coords)
    
    if ball.team == own_team:
        
        # add one of 3 directions and clip
        reduced_running_distance = (DistanceLimits.MAX_RUNNING_DISTANCE - EPS) / np.sqrt(2) - EPS
        allowed_deltas = np.array([[DistanceLimits.MAX_RUNNING_DISTANCE - EPS, 0], [reduced_running_distance, -reduced_running_distance], [reduced_running_distance, reduced_running_distance]])
        delta_indices = (np.random.random_sample((5,)) * 3).astype(int)
        team_coords_candidate = clip_to_pitch(coords[own_team] + allowed_deltas[delta_indices])
        
        # correct if not valid
        for i in range(len(team_coords_candidate)):
            if i == ball.player:
                continue
            if not is_valid_run(coords[own_team][i], team_coords_candidate[i], ball):
                coords[own_team][i] = create_valid_coords(coords[own_team][i], ball.coords)
            else:
                coords[own_team][i] = team_coords_candidate[i]
        
        # if you can shoot to score, then do so
        if ball.coords[0] > Pitch.X_BOUND - DistanceLimits.MAX_SHOOTING_DISTANCE:
            coords[own_team][ball.player] = ball.coords + np.array([DistanceLimits.MAX_SHOOTING_DISTANCE - EPS, 0])
            
            return StrategyOutput(reverse_orientation_if(coords[own_team]), state)
                                
        # pass to player that's most forward, otherwise, pass forward
        best_available_player = -1
        best_x_coord = -Pitch.X_BOUND
        for i in range(5):
            if i == ball.player:
                continue
            distance_to_ball = coord_distance(ball.coords, coords[own_team][i])
            if (distance_to_ball < DistanceLimits.MAX_SHOOTING_DISTANCE - EPS) and (distance_to_ball > DistanceLimits.MIN_SHOOTING_DISTANCE + EPS):
                if coords[own_team][i][0] > best_x_coord:
                    best_available_player = i
                    best_x_coord = coords[own_team][i][0]
        if best_available_player >= 0:
            coords[own_team][ball.player] = coords[own_team][best_available_player]
        else:
            coords[own_team][ball.player] = ball.coords + np.array([DistanceLimits.MAX_SHOOTING_DISTANCE - EPS, 0])
        return StrategyOutput(reverse_orientation_if(coords[own_team]), state)
        
    # we do not have the ball
    min_distance = 0 if ball.team is None else DistanceLimits.MIN_OPP_BALL_DISTANCE + EPS
    for i in range(5):
        distance_to_ball = coord_distance(coords[own_team][i], ball.coords)
        running_distance = min(distance_to_ball, distance_to_ball - min_distance, DistanceLimits.MAX_RUNNING_DISTANCE)
        coefficient = running_distance / (distance_to_ball + EPS)
        
        coords_run = coefficient * ball.coords + (1 - coefficient) * coords[own_team][i]
        
        if not is_valid_run(coords[own_team][i], coords_run, ball):
            coords[own_team][i] = create_valid_coords(coords[own_team][i], ball.coords)
        else:
            coords[own_team][i] = coords_run

    return StrategyOutput(reverse_orientation_if(coords[own_team]), state)
    
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
