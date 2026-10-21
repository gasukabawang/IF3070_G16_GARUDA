import random
import time
from tetangga import generate_all_neighbors


def sideways_move_hill_climbing(initial_state, max_sideways_moves=50, max_iterations=1000):
    start_time = time.time()

    current_state = initial_state.copy()
    current_val = current_state.calculate_objective()

    history_objective = [current_val]
    total_iterations = 0
    total_evaluations = 0
    sideways_count = 0

    while total_iterations < max_iterations:
        neighbors_res = generate_all_neighbors(current_state)
        if isinstance(neighbors_res, tuple):
            neighbors, trial_evals = neighbors_res[0], neighbors_res[1]
            total_evaluations += trial_evals
        else:
            neighbors = neighbors_res

        if not neighbors:
            break

        best_val = float("-inf")
        best_candidates = []
        for neighbor in neighbors:
            val = neighbor.calculate_objective()
            if val > best_val:
                best_val = val
                best_candidates = [neighbor]
            elif val == best_val:
                best_candidates.append(neighbor)

        best_neighbor = random.choice(best_candidates)

        if best_val > current_val:
            sideways_count = 0
        elif best_val == current_val and sideways_count < max_sideways_moves:
            sideways_count += 1
        else:
            break

        current_state = best_neighbor
        current_val = best_val
        total_iterations += 1
        history_objective.append(current_val)

    execution_time = time.time() - start_time

    return {
        "initial_state": initial_state,
        "final_state": current_state,
        "best_objective": current_val,
        "history_objective": history_objective,
        "total_iterations": total_iterations,
        "total_evaluations": total_evaluations,
        "execution_time": execution_time,
        "valid": current_state.is_valid(),
        "sideways_moves_count": sideways_count,
    }
