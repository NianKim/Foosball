# Foosball Strategy — MindPhair 2026

This repository contains the autonomous agent logic for the Mathrix MindPhair 2026 Puzzle.
Our strategy uses **Artificial Potential Fields (APF)** to navigate a 5-player team across a 100×50m pitch in a turn-based 5v5 game.

**Result: 93% win rate vs. the provided easy_strategy baseline.**

---

## How it works

Every turn our strategy receives the current game state (coordinates of every player and the ball). By assigning a potential field to every object on the pitch and computing the superposition of all fields, we obtain a 3D topographic map of the field. Players follow the negative gradient (steepest descent) — they "roll downhill" toward attractive wells and are pushed away from repulsive hills.

Three game states switch which potential functions are active:

| State   | Own players | Opponents  | Ball                   | Global        |
|---------|-------------|------------|------------------------|---------------|
| Attack  | sombrero    | gauss +    | gauss − (weak)         | slope forward |
| Neutral | gauss +     | gauss +    | gauss − (strong)       | none          |
| Defense | gauss +     | sombrero   | gauss − (chase carrier)| slope backward|

**Call chain for non-carrier players:**
`propose_move → [free-ball intercept] → enforce_ball_clearance → enforce_run_distance → clamp_to_pitch`

---

## Potential functions

| Function | Description |
|---|---|
| `gauss_grad` | Bell-curve potential. amplitude > 0 = repulsive hill, < 0 = attractive well. |
| `sombrero_grad` | Ripple potential V(r) = A·sin(b·r)/r. Creates concentric attraction/repulsion rings. First attractive ring at r ≈ π/(2b). Used in attack (own players space into passing lanes) and defense (shadow opponents at intercept distance). |
| `slope_grad` | Constant field tilting the pitch forward in attack, backward in defense. |
| `boundary_grad` | Exponential wall repulsion keeping players inside the pitch. |
| `total_gradient` | Superposition of all above. Uses committed_moves (new_coords) for own team so later players see earlier players' decided positions. |

## Movement helpers

| Function | Description |
|---|---|
| `sort_players_by_distance` | Sorts players by distance to ball, closest first. Ball carrier pinned to back so teammates commit moves before the carrier decides where to pass. |
| `propose_move` | Steepest-descent step: always moves MAX_RUNNING_DISTANCE in the −gradient direction. Falls back toward ball at saddle points. |
| `enforce_run_distance` | Clamps move distance to [MIN, MAX] while preserving direction. |
| `enforce_ball_clearance` | Pushes player radially away from ball if within minimum clearance distance. |
| `clamp_to_pitch` | Clips to pitch interior with 0.05m margin. |
| `ball_carrier_action` | Discrete carrier logic: (1) shoot if goal in range, (2) pass to most forward teammate not behind own goal, (3) emergency clearance. Uses lateral penalty to prefer forward passes over sideways ones. |

## Special cases

**Kickoff:** Detected by ball at centre + carrier at centre. Passes backward-sideways to own winger. Alternates y-direction each kickoff using `prev_state` so the opponent cannot camp one side.

**Free-ball intercept:** If a player can reach the free ball in one step, they stop exactly at the ball instead of flying 10m past it.

---

## Tuning reference

| Observation | Fix |
|---|---|
| Players clump / form a line in attack | ↓ SOMBRERO_AMP |
| Players don't chase loose balls | ↑ \|AMP_BALL_NEUTRAL\| or ↓ SIGMA_BALL |
| Defenders retreat instead of pressing | ↑ \|AMP_BALL_DEFENSE\| or ↓ SLOPE_WEIGHT |
| Players stuck in corner | ↓ BOUNDARY_A or ↑ BOUNDARY_W |
| Attack stays in own half | ↑ SLOPE_WEIGHT |
| Passing always lateral | ↑ LAT_PASS_PENALTY |
| Own players spread too far apart | ↓ AMP_OWN_REPEL |

| Overall parameter performance | implemented optimizer against multiple benchmark strategies |

---

## Files

| File | Description |
|---|---|
| `gradient_strategy.py` | Submission file. Contains all strategy logic. Requires `foosball.py` for local testing. |
| `foosball.py` | Provided simulation environment — not modified. |

---

## Performance

| Matchup | Score | Win rate |
|---|---|---|
| gradient vs easy_strategy | 128-7 (2000 turns) | 94.9% | possession: 61.8% |
| gradient vs easy+2 goalies | 155–9 | 94.5% | possession: 57.4% |
| gradient vs gradient_goalie | 141-0 | 100.0% | possession: 57.1% |

Timing: mean 0.23ms / max 0.38ms — well within the 10ms limit.

---

## Collaborators

- @NianKim      (creator)
- @MFQX         (contributor)
- Marius Dragus (provided insight in to solving the problem)
