Foosball Strategy for MindPhair:
This repository contains the autonomous agent logic for the Mathrix MindPhair 2026 Puzzle. 
Our strategy uses Artificial Potential Fields (APF) to navigate a 5-player team across a 100x50m pitch in a turn-based 5v5 game.

How it works:
Every turn our strategy recieves the current game state (coordinates of every player and the ball).
By assigning a "potential field" to every object (see Strategy table) and making a superposition of all fields we get a 3D topographic map.
The movement logic and decision making process of every player is based on the 3D topographic map which players "roll" down hills and get pushed back by walls.

Strategy table:
Foosball AI: MindPhair 2026 Strategy
|               | Own Players   | Opp Players   | Ball          | Global       |
| ------------- | ------------- | ------------- | ------------- |------------- |
| Attack        | sombrero      | gauss+        | gauss-(weak)  |slope forward |
| Neutral       | gauss+        | gauss+        | gauss-(strong)|none          |
| Defense       | gauss+        | sombrero      | ------------- |slope backward|


Functions:
| Command | Description |
| --- | --- |
| boundary_grad | Boundary Repulsion: Exponential force "walls" that keep players inside the pitch. |
| gauss_grad | Gaussian Hills: Repulsive fields around players to avoid collisions. |
| slope_grad | Goal slopes: Will pull players towards own goal or enemy based on who has posession. |
| sombrero_grad | Oscillating "ripples" to simulate passing distances of players. |



Core Files:

gradient_strathegy.py: The main engine containing the potential field functions and game state logic. Will be merged with foosball.py before handin.

foosball.py: Contains the Ball and Pitch class definitions (Simulation environment).


TODOS:
- [x] **Thu, Apr 2: Steps 1-5 · Pure math layer** 
  - [x] `gauss_grad`
  - [x] `sombrero_grad`
  - [x] `slope_grad`
  - [x] `boundary_grad`

- [ ] **Fri, Apr 3: Steps 6-8 · Field composition** 
  - [ ] `total_gradient`: Sum up the math layers.
  - [ ] `sort_players_by_distance`: Logic to determine who is closest to the ball.
  - [ ] `propose_move`: Generate the raw movement vectors based on the combined gradients.

- [ ] **Sat, Apr 4: Steps 9-11 · Movement constraints** 
  - [ ] `enforce_run`: Cap maximum speed / stamina logic.
  - [ ] `enforce_clearance`: Prevent players from running out of bounds or into each other.
  - [ ] `ball_carrier_action`: Specific logic for the player who currently has the ball (e.g., passing vs. shooting).

- [ ] **Sun, Apr 5: Step 12 · Assemble gradient_strategy** *(integrate)*
  - [ ] Compose all previous steps into the final engine.
  - [ ] First full game run!
  - [ ] Test our AI against the baseline `easy_strategy`.

- [ ] **Mon, Apr 6: Step 13 · Visualize and tune** *(tune)*
  - [ ] Build `visualize_field` using matplotlib `quiver` plots to actually *see* the potential fields.
  - [ ] Tune `SIGMA` and `AMP` constants so movement feels natural (no jittering at walls).

- [ ] **Tue, Apr 7: Steps 14-15 · Benchmark and momentum** *(polish)*
  - [ ] Set up a timing harness (Ensure execution stays strictly under the 0.01s limit).
  - [ ] Run win-rate tests (play 100+ simulated games to get win %).
  - [ ] Factor in `velocity` from `prev_state` to add momentum to player movement.

- [ ] **Wed, Apr 8: 🚨 DEADLINE — Hand in**



Collaborators
We are a team of three working on the Mathrix challenge:

@NianKim 

@(maximilian tag)

@(vincent tag)
