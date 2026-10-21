def stochastic_hill_climbing(initial_state, max_iterations=1000, max_attempts=50):
    start_time = time.time()
    current_state = initial_state.copy()
    current_val = current_state.calculate_objective()
    
    history_objective = [current_val]
    total_iterations = 0
    attempts_without_improvement = 0
    
    while total_iterations < max_iterations:
        neighbors = generate_all_neighbors(current_state)
        if not neighbors:
            break
            
        better_neighbors = []
        for neighbor in neighbors:
            val = neighbor.calculate_objective()
            if val > current_val:
                better_neighbors.append((neighbor, val))
                
        if better_neighbors:
            chosen_neighbor, chosen_val = random.choice(better_neighbors)
            current_state = chosen_neighbor
            current_val = chosen_val
            total_iterations += 1
            history_objective.append(current_val)
            attempts_without_improvement = 0
        else:
            attempts_without_improvement += 1
            if attempts_without_improvement >= max_attempts:
                break
                
    execution_time = time.time() - start_time
    
    return {"initial_state": initial_state, "final_state": current_state, "best_objective": current_val, 
            "history_objective": history_objective, "total_iterations": total_iterations,
            "execution_time": execution_time}

def random_restart_hill_climbing(initial_state=None, ship=None, vehicles_data=None, max_restarts=5, max_iterations_per_restart=500):
    start_time = time.time()
    best_overall_state = None
    best_overall_value = float('-inf')
    combined_history = []
    total_iterations_all = 0
    first_initial_state = initial_state.copy() if initial_state is not None else None
    
    
    if initial_state is not None:
        if ship is None:
            ship = initial_state.ship
        if vehicles_data is None:
            vehicles_data = [{"v_id": v.id,
                              "w": v.w,
                              "l": v.l,
                              "fee": v.shipping_fee,
                              "weight": v.weight,
                              "eta": v.eta}
                            for v in initial_state.vehicles]
            
    if ship is None or vehicles_data is None:
        raise ValueError("Parameter initial_state atau (ship, vehicles_data) harus disediakan.")
        
    for r in range(max_restarts):
        if r == 0 and initial_state is not None:
            restart_init_state = initial_state.copy()
        else:
            restart_init_state = initial_random_state(ship, vehicles_data)
            
        if first_initial_state is None:
            first_initial_state = restart_init_state.copy()
            
        result = steepest_ascent_hill_climbing(
            restart_init_state,
            max_iterations=max_iterations_per_restart
        )
        
        total_iterations_all += result["total_iterations"]
        combined_history.extend(result["history_objective"])
        
        if result["best_objective"] > best_overall_value:
            best_overall_value = result["best_objective"]
            best_overall_state = result["final_state"]
            
    execution_time = time.time() - start_time
    
    return {"initial_state": first_initial_state, "final_state": best_overall_state,
            "best_objective": best_overall_value, "history_objective": combined_history,
            "total_iterations": total_iterations_all, "execution_time": execution_time}