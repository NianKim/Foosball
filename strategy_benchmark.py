# Python >= 3.10
import io
import time
import warnings
import contextlib
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

from foosball import SessionState, easy_strategy
import gradient_strategy_mod as gs
import Submission.gradient_strategy_final as gsf
import old_versions.gradient_strategy as old_grad

OPPONENTS = {
    "easy": easy_strategy,
    "easy_goalie": old_grad.easy_strategy_with_goalie,
    "gradient_goalie": old_grad.gradient_strategy_with_goalie,
}

PARAMETER_RANGES = {
    "AMP_BALL_NEUTRAL": (-70.0, -30.0),
    "AMP_BALL_DEFENSE": (-70.0, -20.0),
    "SLOPE_WEIGHT": (0.2, 1.2),
    "LAT_PASS_PENALTY": (0.2, 1.2),
    "SOMBRERO_AMP": (2.0, 6.0),
}


def evaluate_match(strategy, opponent, turns: int = 2000, seed: int = 0):
    state = SessionState(kickoff_team=0)
    points = [0, 0]
    possession_turns = 0
    goal_spans = []
    current_span = 0

    for t in range(turns):
        with contextlib.redirect_stdout(io.StringIO()):
            winner = state.perform_iteration([strategy, opponent], seed=seed + t)
        current_span += 1
        if state.ball.team == 0:
            possession_turns += 1

        if winner in (0, 1):
            points[winner] += 1
            goal_spans.append(current_span)
            current_span = 0
            state = SessionState(kickoff_team=1 - winner,
                                 strategy_states=state.strategy_states)

    if current_span > 0:
        goal_spans.append(current_span)

    total_goals = points[0] + points[1]
    win_rate = 100.0 * points[0] / max(total_goals, 1)
    avg_goal_span = float(np.mean(goal_spans)) if goal_spans else float(turns)
    possession_rate = 100.0 * possession_turns / turns

    return {
        "wins": points[0],
        "losses": points[1],
        "win_rate": win_rate,
        "goal_difference": points[0] - points[1],
        "avg_turns_per_goal": avg_goal_span,
        "possession_pct": possession_rate,
    }


def print_result(name: str, result: dict):
    print(f"{name}: {result['wins']}-{result['losses']} | "
          f"win_rate={result['win_rate']:.1f}% | "
          f"goal_diff={result['goal_difference']} | "
          f"avg_goal_span={result['avg_turns_per_goal']:.1f} | "
          f"possession={result['possession_pct']:.1f}%")


def evaluate_all(turns: int = 2000, seed: int = 0):
    print(f"Evaluating strategy against {len(OPPONENTS)} opponents")
    for name, opponent in OPPONENTS.items():
        result = evaluate_match(gs.gradient_strategy, opponent, turns=turns, seed=seed)
        print_result(f"gradient vs {name}", result)


def benchmark_strategy(strategy, n: int = 200):
    state = SessionState(kickoff_team=0)
    times = []
    for turn in range(n):
        inp = state.get_strategy_input(0, state.strategy_states[0])
        start = time.perf_counter()
        strategy(inp)
        times.append(time.perf_counter() - start)
        with contextlib.redirect_stdout(io.StringIO()):
            state.perform_iteration([strategy, easy_strategy], seed=turn)
    times = np.array(times)
    print(f"mean: {times.mean()*1000:.3f}ms  "
          f"max:  {times.max()*1000:.3f}ms  "
          f"limit: 10.000ms  "
          f"{'OK' if times.max() < 0.01 else '*** OVER LIMIT ***'}")


def main():
    print("=== Benchmark ===")
    benchmark_strategy(gs.gradient_strategy, n=200)
    print("\n=== Match evaluations ===")
    evaluate_all(turns=2000, seed=0)


if __name__ == '__main__':
    main()
