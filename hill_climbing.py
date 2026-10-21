import random
import time
from tetangga import initial_random_state, generate_all_neighbors

def stochastic_hill_climbing(initial_state, max_iterations=1000):
  start_time = time.time()
  current_state = initial_state.copy()
  current_val = current_state.calculate_objective()

  history_objective = [current_val]
  official_iterations = 0
  total_evaluations = 0

  for _ in range(max_iterations):
    valid_neighbors, trial_evals = generate_all_neighbors(current_state)
    total_evaluations += trial_evals

    if not valid_neighbors:
      break

    better_neighbors = []
    for n in valid_neighbors:
      val = n.calculate_objective()
      if val > current_val:
        better_neighbors.append((n, val))

    if not better_neighbors:
      break

    current_state, current_val = random.choice(better_neighbors)
    official_iterations += 1
    history_objective.append(current_val)

  return {"initial_state": initial_state,
          "final_state": current_state,
          "best_objective": current_val,
          "history_objective": history_objective,
          "total_iterations": official_iterations,
          "total_evaluations": total_evaluations,
          "execution_time": time.time() - start_time,
          "valid": current_state.is_valid()}


def random_restart_hill_climbing(initial_state=None, ship=None, vehicles_data=None,
                                 max_restarts=5, max_iterations_per_restart=1000):
  start_time = time.time()

  if initial_state is not None:
    ship = ship or initial_state.ship
    vehicles_data = vehicles_data or [{"v_id": v.id, "w": v.w, "l": v.l,
                                       "fee": v.shipping_fee, "weight": v.weight, "eta": v.eta}
        for v in initial_state.vehicles]

  if ship is None or vehicles_data is None:
    raise ValueError("Must provide either initial_state or (ship, vehicles_data).")

  best_overall_state = None
  best_overall_value = float("-inf")
  combined_history = []
  total_iterations_all = 0
  total_evaluations_all = 0
  first_initial_state = (initial_state.copy() if initial_state is not None else None)

  for r in range(max_restarts):
    if r == 0 and initial_state is not None:
      restart_init = initial_state.copy()
    else:
      restart_init = initial_random_state(ship, vehicles_data)

    if first_initial_state is None:
      first_initial_state = restart_init.copy()

    result = steepest_ascent_hill_climbing(
        restart_init, max_iterations=max_iterations_per_restart)

    total_iterations_all += result["total_iterations"]
    total_evaluations_all += result["total_evaluations"]
    combined_history.extend(result["history_objective"])

    if result["best_objective"] > best_overall_value:
      best_overall_value = result["best_objective"]
      best_overall_state = result["final_state"]

  return {"initial_state": first_initial_state,
          "final_state": best_overall_state,
          "best_objective": best_overall_value,
          "history_objective": combined_history,
          "total_iterations": total_iterations_all,
          "total_evaluations": total_evaluations_all,
          "restarts_count": max_restarts,
          "execution_time": time.time() - start_time,
          "valid": (best_overall_state.is_valid() if best_overall_state else True)}
import time
import random
from tetangga import generate_all_neighbors, initial_random_state

def steepest_ascent_hill_climbing(initial_state, max_iterations=1000):
    start_time = time.time()
    current_state = initial_state.copy()
    current_val = current_state.calculate_objective()

    history_objective = [current_val]
    iterations = 0
    total_evaluations = 0

    while iterations < max_iterations:
        valid_neighbors, trial_evals = generate_all_neighbors(current_state)
        total_evaluations += trial_evals

        if not valid_neighbors:
            break

        best_val = max(n.calculate_objective() for n in valid_neighbors)

        if best_val <= current_val:
            break

        best_candidates = [n for n in valid_neighbors if n.calculate_objective() == best_val]
        current_state = random.choice(best_candidates)
        current_val = best_val
        iterations += 1
        history_objective.append(current_val)

    return {
        "initial_state": initial_state,
        "final_state": current_state,
        "best_objective": current_val,
        "history_objective": history_objective,
        "total_iterations": iterations,
        "total_evaluations": total_evaluations,
        "execution_time": time.time() - start_time,
        "valid": current_state.is_valid()
    }

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
