import math
import time
import random
from tetangga import generate_random_valid_neighbor


def simulated_annealing(
    initial_state,
    initial_temperature=1000.0,
    cooling_rate=0.95,
    min_temperature=0.01,
    max_iterations=1500
):

    start_time = time.time()

    current_state = initial_state.copy()
    current_val = current_state.calculate_objective()

    best_state = current_state.copy()
    best_val = current_val

    history_objective = [current_val]
    history_acceptance_prob = []
    
    official_iterations = 0
    total_evaluations = 0
    stuck_count = 0

    T = float(initial_temperature)
    step = 0

    while T > min_temperature and step < max_iterations:
        step += 1

        
        next_state, trial_evals = generate_random_valid_neighbor(current_state)
        total_evaluations += trial_evals

        if next_state is None:
            stuck_count += 1
            T *= cooling_rate
            continue

        next_val = next_state.calculate_objective()
        delta_E = next_val - current_val

        if delta_E > 0:
            
            current_state = next_state
            current_val = next_val
            official_iterations += 1
            history_objective.append(current_val)
            history_acceptance_prob.append(1.0)

            if current_val > best_val:
                best_val = current_val
                best_state = current_state.copy()
        else:
            
            try:
                acceptance_prob = math.exp(delta_E / max(1e-9, T))
            except OverflowError:
                acceptance_prob = 0.0

            history_acceptance_prob.append(acceptance_prob)

            if random.random() < acceptance_prob:
                current_state = next_state
                current_val = next_val
                official_iterations += 1
                history_objective.append(current_val)
            else:
                
                stuck_count += 1

        
        T *= cooling_rate

    execution_time = time.time() - start_time

    return {
        "initial_state": initial_state,
        "final_state": best_state,
        "best_objective": best_val,
        "history_objective": history_objective,
        "history_acceptance_prob": history_acceptance_prob,
        "stuck_count": stuck_count,
        "total_iterations": official_iterations,
        "total_evaluations": total_evaluations,
        "final_temperature": T,
        "execution_time": execution_time,
        "valid": best_state.is_valid()
    }