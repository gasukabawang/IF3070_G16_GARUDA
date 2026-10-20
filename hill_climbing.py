import time
import random
from tetangga import generate_all_neighbors, initial_random_state

def steepest_ascent_hill_climbing(initial_state, max_iterations=500, max_no_improvement=50):
    start_time = time.time()
    current_state = initial_state.copy()
    current_val = current_state.calculate_objective()

    history_objective = [current_val]
    official_iterations = 0
    total_evaluations = 0

    while official_iterations < max_iterations:
        valid_neighbors, trial_evals = generate_all_neighbors(current_state)
        total_evaluations += trial_evals

        if not valid_neighbors:
            break

        best_val = float('-inf')
        best_candidates = []

        for n in valid_neighbors:
            val = n.calculate_objective()
            if val > best_val:
                best_val = val
                best_candidates = [n]
            elif val == best_val:
                best_candidates.append(n)

        if best_val > current_val:
            current_state = random.choice(best_candidates)
            current_val = best_val
            official_iterations += 1
            history_objective.append(current_val)
        else:
            break

    execution_time = time.time() - start_time
    return {
        "initial_state": initial_state,
        "final_state": current_state,
        "best_objective": current_val,
        "history_objective": history_objective,
        "total_iterations": official_iterations,
        "total_evaluations": total_evaluations,
        "execution_time": execution_time,
        "valid": current_state.is_valid()
    }