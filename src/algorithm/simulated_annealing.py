import math
import random
import time
from tetangga import generate_random_valid_neighbor

def simulated_annealing(initial_state, initial_temperature=1000.0, cooling_rate=0.95, min_temperature=0.01, max_iterations=1500):
    start_time = time.time()

    current_state = initial_state.copy()
    current_val = current_state.calculate_objective()

    best_state = current_state.copy()
    best_val = current_val

    history_objective = [current_val]
    history_acceptance_prob = [1.0]

    iterations = 0
    total_evaluations = 0
    stuck_count = 0
    t = float(initial_temperature)

    step = 0
    while t > min_temperature and step < max_iterations:
        step += 1

        next_state, trial_evals = generate_random_valid_neighbor(current_state)
        total_evaluations += trial_evals

        if next_state is None:
            stuck_count += 1
            history_acceptance_prob.append(0.0)
            t *= cooling_rate
            continue

        next_val = next_state.calculate_objective()
        delta_e = next_val - current_val

        if delta_e > 0:
            prob = 1.0
        else:
            try:
                prob = math.exp(delta_e / max(1e-9, t))
            except OverflowError:
                prob = 0.0

        history_acceptance_prob.append(round(prob, 4))

        if delta_e > 0 or random.random() < prob:
            current_state = next_state
            current_val = next_val
            iterations += 1
            history_objective.append(current_val)

            if current_val > best_val:
                best_val = current_val
                best_state = current_state.copy()
        else:
            stuck_count += 1

        t *= cooling_rate

    return {
        "initial_state": initial_state,
        "final_state": best_state,
        "best_objective": best_val,
        "history_objective": history_objective,
        "history_acceptance_prob": history_acceptance_prob,
        "stuck_count": stuck_count,
        "stuck_frequency": round(stuck_count / max(1, step), 4),
        "total_iterations": iterations,
        "total_evaluations": total_evaluations,
        "final_temperature": t,
        "execution_time": time.time() - start_time,
        "valid": best_state.is_valid()
    }
