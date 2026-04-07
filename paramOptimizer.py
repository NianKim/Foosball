import numpy as np
import io, contextlib
import gradient_strategy_mod as gs
import old_versions.gradient_strategy as old_grad
from foosball import SessionState, easy_strategy

# Optimizer against multiple oponent strategies

base = {
    'SIGMA_PLAYER': 8.0,
    'SIGMA_BALL': 14.0,
    'AMP_OWN_REPEL': 2.1,
    'AMP_OPP_ATTACK': 3.5,
    'AMP_OPP_NEUTRAL': 1.8,
    'AMP_BALL_ATTACK': -5.0,
    'AMP_BALL_NEUTRAL': -50.0,
    'AMP_BALL_DEFENSE': -40.0,
    'SOMBRERO_B': 0.2,
    'SOMBRERO_AMP': 4.0,
    'SLOPE_WEIGHT': 0.6,
    'BOUNDARY_A': 50.0,
    'BOUNDARY_W': 2.3,
    'LAT_PASS_PENALTY': 0.53,
}
param_names = list(base.keys())

OPPONENTS = [
    ("easy", easy_strategy),
    ("easy_goalie", old_grad.easy_strategy_with_goalie),
    ("gradient_goalie", old_grad.gradient_strategy_with_goalie),
]

def evaluate_strategy_against(opponent, seeds=(0, 5, 10), turns=1000):
    total = [0, 0]
    for seed in seeds:
        state = SessionState(kickoff_team=0)
        for t in range(turns):
            with contextlib.redirect_stdout(io.StringIO()):
                winner = state.perform_iteration([gs.gradient_strategy, opponent], seed=seed + t)
            if winner in (0, 1):
                total[winner] += 1
                state = SessionState(kickoff_team=1 - winner, strategy_states=state.strategy_states)

    score = 100.0 * total[0] / sum(total)
    return score, total

def eval_params(params, opponents=OPPONENTS):
    for name, value in params.items():
        setattr(gs, name, value)

    scores = []
    totals = []
    for name, opponent in opponents:
        score, total = evaluate_strategy_against(opponent)
        scores.append(score)
        totals.append((name, total))

    avg_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    robust_score = avg_score - 0.25 * std_score
    loss = 100.0 - avg_score
    return robust_score, loss, avg_score, std_score, scores, totals

def search_parameters(base, initial_step=0.05, min_step=1e-4):
    params = base.copy()
    objective, loss, avg_score, std_score, scores, totals = eval_params(params)
    print('start', f'objective={objective:.4f}', f'avg={avg_score:.4f}', f'std={std_score:.4f}', scores, totals)

    step = initial_step
    while step >= min_step:
        any_improvement = False

        for name in param_names:
            current = params[name]

            while True:
                best_val = current
                best_obj = objective
                best_avg = avg_score
                best_std = std_score

                for factor in (1.0 + step, 1.0 - step):
                    candidate = current * factor
                    params[name] = candidate
                    cand_obj, cand_loss, cand_avg, cand_std, cand_scores, cand_totals = eval_params(params)

                    print(f'    try {name}={candidate:.6f} -> objective={cand_obj:.4f} avg={cand_avg:.4f} std={cand_std:.4f} scores={cand_scores}')

                    if cand_obj > best_obj:
                        best_obj = cand_obj
                        best_avg = cand_avg
                        best_std = cand_std
                        best_val = candidate

                if best_obj > objective:
                    print(f'  accepted {name}: {current:.6f} -> {best_val:.6f} (objective {objective:.4f} -> {best_obj:.4f})')
                    current = best_val
                    params[name] = best_val
                    objective = best_obj
                    avg_score = best_avg
                    std_score = best_std
                    any_improvement = True
                else:
                    params[name] = current
                    break

        if not any_improvement:
            step /= 2.0
            print('no improvement in full pass, reduce step to', step)

    return params, objective, avg_score, std_score

best_params, best_objective, best_avg, best_std = search_parameters(base)
print('final objective', best_objective)
print('final avg', best_avg, 'std', best_std)
print('final params', best_params)


# Found parameters for overall performance against multiple opponents:

# Optimized parameters with paramOptimizer.py
# SIGMA_PLAYER     = 8.0    # spatial spread of player potentials [m]
# SIGMA_BALL       = 14.0   # spatial spread of ball potential [m]
# AMP_OWN_REPEL    = 2.1    # own-player repulsion (spread out)
# AMP_OPP_ATTACK   = 3.5    # opponent repulsion in attack (avoid blockers)
# AMP_OPP_NEUTRAL  = 1.8168046875    # opponent repulsion in neutral
# AMP_BALL_ATTACK  = -5.0   # ball attraction in attack (weak — carrier has it)
# AMP_BALL_NEUTRAL = -50.0  # ball attraction in neutral (strong — go get it)
# AMP_BALL_DEFENSE = -40.0  # ball attraction in defense (chase the carrier)
# SOMBRERO_B       = 0.2    # first attractive ring at r ≈ π/(2b) ≈ 7.8m
# SOMBRERO_AMP     = 4.025    # sombrero strength — shapes passing lanes in attack
# SLOPE_WEIGHT     = 0.6    # strength of global forward/backward tilt
# BOUNDARY_A       = 52.5   # wall repulsion amplitude
# BOUNDARY_W       = 2.3    # wall decay width [m] — kicks in within ~3m of edge
# LAT_PASS_PENALTY = 0.5565000000000001   # penalty per metre of lateral distance when choosing pass target
# === Evaluations ===
# Evaluating strategy against 3 opponents
# gradient vs easy: 132-11 | win_rate=92.3% | goal_diff=121 | avg_goal_span=13.9 | possession=61.8%
# gradient vs easy_goalie: 158-2 | win_rate=98.8% | goal_diff=156 | avg_goal_span=12.4 | possession=60.0%
# gradient vs gradient_goalie: 8-0 | win_rate=100.0% | goal_diff=8 | avg_goal_span=222.2 | possession=41.9%



############################### Optimizer against single opponent strategy ###########################################
# import numpy as np
# import io, contextlib
# import gradient_strategy3 as gs
# from foosball import SessionState, easy_strategy

# # Parameters found for this strategy
# base = {
#     'SIGMA_PLAYER': 8.0,
#     'SIGMA_BALL': 14.0,
#     'AMP_OWN_REPEL': 2.1,
#     'AMP_OPP_ATTACK': 3.5,
#     'AMP_OPP_NEUTRAL': 1.8,
#     'AMP_BALL_ATTACK': -5.0,
#     'AMP_BALL_NEUTRAL': -50.0,
#     'AMP_BALL_DEFENSE': -40.0,
#     'SOMBRERO_B': 0.2,
#     'SOMBRERO_AMP': 4.0,
#     'SLOPE_WEIGHT': 0.6,
#     'BOUNDARY_A': 50.0,
#     'BOUNDARY_W': 2.3,
#     'LAT_PASS_PENALTY': 0.53,
# }
# param_names = list(base.keys())

# def eval_params(params):
#     for name, value in params.items():
#         setattr(gs, name, value)

#     total = [0, 0]
#     for seed in [0, 5, 10]:
#         state = SessionState(kickoff_team=0)
#         for t in range(1000):
#             with contextlib.redirect_stdout(io.StringIO()):
#                 winner = state.perform_iteration([gs.gradient_strategy, easy_strategy], seed=seed + t)
#             if winner in (0, 1):
#                 total[winner] += 1
#                 state = SessionState(kickoff_team=1 - winner, strategy_states=state.strategy_states)

#     score = 100.0 * total[0] / sum(total)
#     loss = 100.0 - score
#     return score, loss, total

# def search_parameters(base, initial_step=0.05, min_step=1e-4):
#     params = base.copy()
#     score, loss, total = eval_params(params)
#     print('start', score, loss, total)

#     step = initial_step
#     while step >= min_step:
#         any_improvement = False

#         for name in param_names:
#             current = params[name]

#             while True:
#                 best_val = current
#                 best_score = score
#                 best_loss = loss

#                 for factor in (1.0 + step, 1.0 - step):
#                     candidate = current * factor
#                     params[name] = candidate
#                     cand_score, cand_loss, cand_total = eval_params(params)
#                     print(f'  try {name}={candidate:.6f} -> {cand_score:.4f}% loss={cand_loss:.4f}')

#                     if cand_score > best_score:
#                         best_score = cand_score
#                         best_loss = cand_loss
#                         best_val = candidate

#                 if best_score > score:
#                     print(f'accepted {name}: {current:.6f} -> {best_val:.6f} ({score:.4f}% -> {best_score:.4f}%)')
#                     current = best_val
#                     params[name] = best_val
#                     score = best_score
#                     loss = best_loss
#                     any_improvement = True
#                 else:
#                     params[name] = current
#                     break

#         if not any_improvement:
#             step /= 2.0
#             print('no improvement in full pass, reduce step to', step)

#     return params, score

# best_params, best_score = search_parameters(base)
# print('final', best_score, best_params)