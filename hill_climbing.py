import time
import random
from tetangga import generate_all_neighbors, initial_random_state

def steepest_ascent_hill_climbing(initial_state, max_iterations=500, max_no_improvement=50):
    start_time = time.time()
    
    current_state = initial_state.copy()
    current_val = current_state.calculate_objective()
    
    history_objective = [current_val]
    total_iterations = 0
    no_improvement_count = 0
    
    while total_iterations < max_iterations:
        neighbors = generate_all_neighbors(current_state)
        if not neighbors:
            break
            
        best_val = float('-inf')
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
            current_state = best_neighbor
            current_val = best_val
            total_iterations += 1
            history_objective.append(current_val)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement or best_val <= current_val:
                break
                
    execution_time = time.time() - start_time
    
    return {
        "initial_state": initial_state,
        "final_state": current_state,
        "best_objective": current_val,
        "history_objective": history_objective,
        "total_iterations": total_iterations,
        "execution_time": execution_time
    }