#    - ball.team equals the team number (0 or 1)
# if the ball is not in possession of a team, then self.team is None.
import numpy as np
from foosball import Ball
from foosball import Pitch

def determine_game_state(ball, own_team: int) -> str:
    if ball.team == own_team:
        return 'attack'
    elif ball.team is None:
        return 'neutral'
    else:
        return 'defense'
    

    #    - ball.player equals the index of the player that possesses the ball

'''
b = Ball(0)           # team 0 has the ball
assert determine_game_state(b, 0) == 'attack'
assert determine_game_state(b, 1) == 'defense'
b.team = None
assert determine_game_state(b, 0) == 'neutral'
print("step 1 ok")
'''
#should this be here or at the top of file? does this give performacnce benefits?
EPS = 1e-8

#PRE:   takes a position on gradient, the center, amplitude of gradient and curvature
#POST:  potential field based on location and fomula
def gauss_grad(pos: np.ndarray, center: np.ndarray, amplitude: float, sigma: float) -> np.ndarray:
    diff = pos - center                        # vector pointing away from center
    r2   = np.dot(diff, diff)                  # difference squared (scalar)
    V    = amplitude * np.exp(-r2 / (2.0 * sigma**2))
    return V * (-diff / sigma**2)              

'''
# Test 1: at the center, gradient must be exactly zero (peak of hill, flat)
g = gauss_grad(np.array([0.0, 0.0]), np.array([0.0, 0.0]), amplitude=1.0, sigma=5.0)
assert np.allclose(g, [0.0, 0.0]), f"expected zero at center, got {g}"

# Test 2: repulsive hill (amplitude > 0) — gradient should point TOWARD center
# so gradient descent moves AWAY. The returned vector points toward center.
g = gauss_grad(np.array([3.0, 0.0]), np.array([0.0, 0.0]), amplitude=1.0, sigma=5.0)
assert g[0] < 0, f"repulsive: gradient x should be negative (points left toward center), got {g}"

# Test 3: attractive well (amplitude < 0) — gradient points AWAY from center
g = gauss_grad(np.array([3.0, 0.0]), np.array([0.0, 0.0]), amplitude=-1.0, sigma=5.0)
assert g[0] > 0, f"attractive: gradient x should be positive (points right away from center), got {g}"

print("step 2 ok")
'''
#PRE:  Takes position, center of gradient, amplitude and a scalar b
#POST: Potential is V(r) = A * sin (b * r)/r, while b is a scalar
# => Gradient is  A *(b*cos(b*r)/r − sin(b*r)/r^2)

#ring at a different distance, solve b = π/(2·target_r).

def sombrero_grad(pos: np.ndarray, center: np.ndarray, amplitude: float, b: float) -> np.ndarray:
    diff = pos - center
    r = np.sqrt(np.dot(diff, diff)) #scalar distance
    if r < EPS:                     #dealing with singularity at zero with early zero
        return np.zeros(2)
    r_hat = diff/r                  #unit radial vector
    dV_dr  = amplitude * (b * np.cos(b * r) / r  -  np.sin(b * r) / r**2)
    return dV_dr * r_hat

'''
# Test 1: singularity guard — must not blow up or NaN at center
g = sombrero_grad(np.array([0.0, 0.0]), np.array([0.0, 0.0]), amplitude=1.5, b=0.22)
assert not np.any(np.isnan(g)), "NaN at center"
assert np.allclose(g, [0.0, 0.0])

# Test 2: gradient must be purely radial (no perpendicular component)
# Place pos along x-axis — gradient must have zero y component
g = sombrero_grad(np.array([5.0, 0.0]), np.array([0.0, 0.0]), amplitude=1.5, b=0.22)
assert abs(g[1]) < 1e-6, f"non-radial component: {g}"

# Test 3: visualise the ring structure — print V(r) at several distances
b = 0.22
print("r   →   V(r)   [should cross zero near r=π/b ≈ 14.3m]")
for r in [1, 3, 5, 7, 10, 14, 18]:
    V = 1.5 * np.sin(b * r) / r
    print(f"  r={r:4.1f}m   V={V:+.4f}")
print("step 3 ok")

'''

#PRE: takes who has the ball
#POST: V(x) =  sign * x (sign is dependant on who has posession)

SLOPE_WEIGHT = 0.3 #tweak this

def slope_grad(own_team: int) -> np.ndarray:
    sign = +1.0 if own_team == 1 else -1.0
    return np.array([sign * SLOPE_WEIGHT, 0.0])

#is this function dynmaically switching its sign for who has the ball in the current state? _> TODo
'''
g0 = slope_grad(0)
g1 = slope_grad(1)
assert g0[0] < 0, "team 0 slope should pull toward +x (gradient is negative x)"
assert g1[0] > 0, "team 1 slope should pull toward -x (gradient is positive x)"
assert g0[1] == 0.0 and g1[1] == 0.0, "slope has no y component"
print("step 4 ok")
'''

#will point HARD up the potential wall, and we want to do gradient descent (so negative of gradient)
#this is no longer a force/potential!! the sign is inverted 0v0
BOUNDARY_A = 80.0
BOUNDARY_W = 3.0

def boundary_grad(pos: np.ndarray) -> np.ndarray:
    from foosball import Pitch
    x, y = pos[0], pos[1]
    a_w  = BOUNDARY_A / BOUNDARY_W

    d_left   = x + Pitch.X_BOUND       # distance to left wall  (x=−50)
    d_right  = Pitch.X_BOUND - x       # distance to right wall (x=+50)
    d_bottom = y + Pitch.Y_BOUND       # distance to bottom wall
    d_top    = Pitch.Y_BOUND - y       # distance to top wall

    gx = -a_w * np.exp(-d_left  / BOUNDARY_W) \
         +a_w * np.exp(-d_right / BOUNDARY_W)
    gy = -a_w * np.exp(-d_bottom / BOUNDARY_W) \
         +a_w * np.exp(-d_top   / BOUNDARY_W)

    return np.array([gx, gy])

'''
# Test 1: pitch centre — walls are symmetric, gradient should be ~zero
g = boundary_grad(np.array([0.0, 0.0]))
assert np.allclose(g, [0.0, 0.0], atol=1e-6), f"centre should be near zero: {g}"

# Test 2: near left wall — should push strongly in +x direction
g = boundary_grad(np.array([-48.0, 0.0]))   # 2m from left wall
assert g[0] < -5.0, f"near left wall: expected strong +x push, got {g}"
assert abs(g[1]) < 1e-3, "should have no y component on x-axis"

# Test 3: near top wall — should push in -y direction
g = boundary_grad(np.array([0.0, 23.0]))    # 2m from top wall
assert g[1] > 5.0, f"near top wall: expected strong -y push, got {g}"

print("step 5 ok")
'''

'''
def slope_grad(own_team: int, game_state: str) -> np.ndarray:
    # If nobody has the ball, no slope pull
    if game_state == 'neutral':
        return np.zeros(2)
        
    # Team 1 attacks towards +x, Team 0 attacks towards -x (adjust based on your pitch layout)
    base_sign = +1.0 if own_team == 1 else -1.0
    
    # If defending, reverse the slope to pull them back
    if game_state == 'defense':
        base_sign *= -1.0
        
    return np.array([base_sign * SLOPE_WEIGHT, 0.0])
'''