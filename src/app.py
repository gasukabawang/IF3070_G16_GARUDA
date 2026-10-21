import os
import sys
import json
import time
import math
import random
from flask import Flask, render_template, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
ALGO_DIR = os.path.join(BASE_DIR, "algorithm")
if ALGO_DIR not in sys.path:
    sys.path.insert(0, ALGO_DIR)

from model import Ship, Vehicle
from tetangga import initial_random_state, generate_all_neighbors, generate_random_valid_neighbor
from algorithm.hill_climbing import (
    steepest_ascent_hill_climbing,
    stochastic_hill_climbing,
    sideways_move_hill_climbing,
    random_restart_hill_climbing
)
from algorithm.simulated_annealing import simulated_annealing
from algorithm.genetic_algorithm import (
    genetic_algorithm, crossover, mutate, select_parent_roulette
)

app = Flask(__name__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

default_ship = Ship(w=6, l=6, MaxCapacity=100)
default_vehicles_data = [
    {"v_id": "V1", "w": 2, "l": 2, "fee": 5000, "weight": 20, "eta": 1},
    {"v_id": "V2", "w": 2, "l": 3, "fee": 7000, "weight": 30, "eta": 1},
    {"v_id": "V3", "w": 1, "l": 4, "fee": 4000, "weight": 15, "eta": 2},
    {"v_id": "V4", "w": 2, "l": 2, "fee": 6000, "weight": 25, "eta": 3},
    {"v_id": "V5", "w": 3, "l": 1, "fee": 5500, "weight": 18, "eta": 2},
    {"v_id": "V6", "w": 1, "l": 2, "fee": 3000, "weight": 10, "eta": 4},
    {"v_id": "V7", "w": 2, "l": 1, "fee": 3500, "weight": 12, "eta": 5},
    {"v_id": "V8", "w": 3, "l": 2, "fee": 8000, "weight": 35, "eta": 3},
]
data_path = os.path.join(PROJECT_ROOT, "data", "vehicles.json")
if os.path.exists(data_path):
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            default_vehicles_data = json.load(f)
    except Exception:
        pass

current_state = None
current_iter = 0
total_evaluations = 0
history_objective = []
history_avg_objective = []
is_stuck = False
sideways_count = 0
restart_count = 0
best_overall_score = float("-inf")
current_objective_type = "fee"
sa_temperature = 1000.0
sa_history_prob = [1.0]
sa_stuck_count = 0
ga_population = []
step_session_start = None

def get_algo_slug(name):
    s = name.lower()
    for w in ["hill-climbing", "hill_climbing", "algorithm", "(", ")", "step", "move"]:
        s = s.replace(w, "")
    return "_".join(s.split()).strip("_") or "algo"

def get_next_run_number(slug):
    count = 1
    while True:
        candidates = [
            f"result_{slug}_{count}.json",
            f"result_{slug}_{count}.txt",
            f"run_{slug}_{count}.json",
            f"run_{slug}_{count}.txt",
            f"run_{slug}_run_{count}.json",
            f"run_{slug}_run_{count}.txt"
        ]
        if not any(os.path.exists(os.path.join(OUTPUT_DIR, f)) for f in candidates):
            return count
        count += 1

def save_run_result(algo_name, params_summary, metric, state, history, history_avg=None, objective_type="fee"):
    slug = get_algo_slug(algo_name)
    run_num = get_next_run_number(slug)
    base_name = f"result_{slug}_{run_num}"
    grid_matrix, _, _ = state.get_deck_grid()

    data = {
        "run_number": run_num,
        "algorithm": algo_name,
        "objective_type": objective_type,
        "parameters": params_summary,
        "execution_time_seconds": metric.get("execution_time", 0),
        "total_iterations": metric.get("iterations", 0),
        "total_evaluations": metric.get("evaluations", 0),
        "best_objective": metric.get("score", 0),
        "is_valid": metric.get("valid", state.is_valid()),
        "ship": {
            "w": default_ship.w, "l": default_ship.l, "max_capacity": default_ship.MaxCapacity,
            "loaded_weight": state.get_total_weight(), "total_fee": state.get_total_fee()
        },
        "vehicles": [v.to_dict() for v in state.vehicles],
        "history_objective": history,
        "history_avg_objective": history_avg or []
    }
    for k in ["stuck_count", "stuck_frequency", "history_acceptance_prob"]:
        if k in metric:
            data[k] = metric[k]

    with open(os.path.join(OUTPUT_DIR, f"{base_name}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, f"{base_name}.txt"), "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\nRUN #{run_num} - {algo_name.upper()}\n{'='*60}\n")
        f.write(f"Algorithm       : {algo_name}\nObjective Type  : {objective_type}\nParameters      : {params_summary}\n")
        f.write(f"Execution Time  : {metric.get('execution_time', 0):.4f}s\nIterations      : {metric.get('iterations', 0)}\n")
        f.write(f"Evaluations     : {metric.get('evaluations', 0)}\nObjective Score : {metric.get('score', 0)}\n")
        f.write(f"Deck Capacity   : {state.get_total_weight()} / {default_ship.MaxCapacity} Tons\nTotal Fee       : ${state.get_total_fee():,}\n")
        f.write(f"Valid Placement : {'YES' if metric.get('valid', state.is_valid()) else 'NO'}\n")
        if "stuck_count" in metric:
            f.write(f"Stuck Frequency : {metric['stuck_count']} times\n")
        f.write(f"{'-'*60}\nVEHICLES:\n")
        for v in state.vehicles:
            w_eff, l_eff = v.get_dimensions()
            st = "LOADED" if v.is_loaded else "PARKED"
            f.write(f"  [{st:6s}] {v.id:4s} | Pos: ({v.x},{v.y}) | Dim: {w_eff}x{l_eff} ({v.orientation:10s}) | Berat: {v.weight:2d}T | Fee: ${v.shipping_fee:5d} | ETA: {v.eta}\n")
        f.write(f"{'-'*60}\nDECK LAYOUT ({default_ship.w}x{default_ship.l}):\n")
        for r in range(default_ship.l):
            row_str = "  " + " ".join("[ . ]" if grid_matrix[r][c]["count"] == 0 else "[ ! ]" if grid_matrix[r][c]["count"] > 1 else f"[{grid_matrix[r][c]['vehicles'][0]:3s}]" for c in range(default_ship.w))
            f.write(row_str + "\n")
        f.write(f"{'='*60}\n")

    return f"{base_name}.json"

def init_session(obj_type="fee"):
    global current_state, current_iter, total_evaluations, history_objective, history_avg_objective
    global is_stuck, sideways_count, restart_count, best_overall_score
    global sa_temperature, sa_history_prob, sa_stuck_count, ga_population, current_objective_type, step_session_start

    current_objective_type = obj_type
    current_state = initial_random_state(default_ship, default_vehicles_data, objective_type=current_objective_type)
    current_iter = 0
    total_evaluations = 1
    score = current_state.calculate_objective()
    history_objective = [score]
    history_avg_objective = [score]
    is_stuck = False
    sideways_count = 0
    restart_count = 0
    best_overall_score = score
    sa_temperature = 1000.0
    sa_history_prob = [1.0]
    sa_stuck_count = 0
    ga_population = []
    step_session_start = None

def get_state_payload(message="", trial_clash=None):
    if current_state is None:
        init_session()
    grid_matrix, overlap_count, oob_count = current_state.get_deck_grid()
    score = current_state.calculate_objective()
    info = current_state.get_info()
    return {
        "grid": grid_matrix, "score": score, "total_fee": info["total_fee"], "total_weight": info["total_weight"],
        "MaxCapacity": info["MaxCapacity"], "valid": info["valid"], "overlap": overlap_count, "oob": oob_count,
        "iter": current_iter, "evaluations": total_evaluations, "is_stuck": is_stuck, "sideways_count": sideways_count,
        "restart_count": restart_count, "best_overall_score": best_overall_score if best_overall_score != float("-inf") else score,
        "sa_temperature": sa_temperature, "sa_history_prob": sa_history_prob, "sa_stuck_count": sa_stuck_count,
        "vehicles": info["vehicles"], "history": history_objective, "history_avg": history_avg_objective,
        "trial_clash": trial_clash, "ship_w": default_ship.w, "ship_l": default_ship.l,
        "objective_type": current_objective_type, "message": message
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    init_session(data.get("objective_type", current_objective_type))
    return jsonify(get_state_payload(message="Initial random state successfully generated, state is valid!"))

@app.route("/api/step", methods=["POST"])
def step():
    global current_state, current_iter, total_evaluations, history_objective, history_avg_objective
    global is_stuck, sideways_count, restart_count, best_overall_score
    global sa_temperature, sa_history_prob, sa_stuck_count, ga_population, current_objective_type, step_session_start

    if current_state is None:
        init_session()
    if step_session_start is None:
        step_session_start = time.time()

    data = request.get_json(silent=True) or {}
    algo = data.get("algorithm", "steepest")
    obj_type = data.get("objective_type", current_objective_type)
    if current_state.objective_type != obj_type:
        current_objective_type = obj_type
        current_state.objective_type = obj_type

    max_iter = int(data.get("max_iterations", 500))
    max_sideways = int(data.get("max_sideways_moves", 100))
    max_restarts = int(data.get("max_restarts", 5))
    cooling_rate = float(data.get("cooling_rate", 0.95))
    mutation_rate = float(data.get("mutation_rate", 0.15))
    pop_size = int(data.get("population_size", 20))
    max_gens = int(data.get("max_generations", 40))

    current_val = current_state.calculate_objective()
    msg, trial_clash = "", None

    if is_stuck and algo != "genetic_algorithm":
        return jsonify(get_state_payload(message=f"Search is terminated at Local Optimum ({current_val})."))

    if algo == "genetic_algorithm" and is_stuck:
        is_stuck, current_iter, ga_population = False, 0, []
        history_objective, history_avg_objective = [current_val], [current_val]

    if algo in ["steepest", "stochastic", "sideways", "restart"]:
        if current_iter >= max_iter and algo != "restart":
            is_stuck = True
            return jsonify(get_state_payload(message=f"Maximum iterations ({max_iter}) reached, search terminated."))

        valid_neighbors, trial_evals, clash_state = generate_all_neighbors(current_state, return_clash=True)
        total_evaluations += trial_evals
        if clash_state:
            c_grid, c_over, c_oob = clash_state.get_deck_grid()
            trial_clash = {"grid": c_grid, "overlap": c_over, "oob": c_oob}

        if algo == "steepest":
            if not valid_neighbors:
                is_stuck, msg = True, f"No valid neighbors generated. Trapped at Local Optimum ({current_val})."
            else:
                best_val = max(n.calculate_objective() for n in valid_neighbors)
                if best_val > current_val:
                    current_state = random.choice([n for n in valid_neighbors if n.calculate_objective() == best_val])
                    current_iter += 1
                    history_objective.append(best_val)
                    is_stuck = (current_iter >= max_iter)
                    msg = f"Step {current_iter}: Moved to better neighbor ({best_val}, +{best_val - current_val})."
                else:
                    is_stuck, msg = True, f"Trapped at Local Optimum (Objective: {current_val}). There are no better neighbor."

        elif algo == "stochastic":
            better = [(n, val) for n in valid_neighbors if (val := n.calculate_objective()) > current_val]
            if better:
                current_state, chosen_val = random.choice(better)
                current_iter += 1
                history_objective.append(chosen_val)
                is_stuck = (current_iter >= max_iter)
                msg = f"Step {current_iter}: Selected improving neighbor ({chosen_val}, +{chosen_val - current_val})."
            else:
                is_stuck, msg = True, f"Trapped at Local Optimum (Objective: {current_val}). There are no better neighbor."

        elif algo == "sideways":
            if not valid_neighbors:
                is_stuck, msg = True, f"No valid neighbors generated. Search halted at {current_val}."
            else:
                best_val = max(n.calculate_objective() for n in valid_neighbors)
                best_neighbor = random.choice([n for n in valid_neighbors if n.calculate_objective() == best_val])
                if best_val > current_val:
                    current_state, sideways_count = best_neighbor, 0
                    current_iter += 1
                    history_objective.append(best_val)
                    is_stuck = (current_iter >= max_iter)
                    msg = f"Step {current_iter}: Improved objective to {best_val} (+{best_val - current_val})."
                elif best_val == current_val and sideways_count < max_sideways:
                    current_state = best_neighbor
                    sideways_count += 1
                    current_iter += 1
                    history_objective.append(best_val)
                    is_stuck = (current_iter >= max_iter or sideways_count >= max_sideways)
                    msg = f"Step {current_iter}: Sideways move #{sideways_count}/{max_sideways} (Objective: {best_val})."
                else:
                    is_stuck = True
                    msg = f"Max sideways moves reached ({sideways_count}/{max_sideways})." if sideways_count >= max_sideways else f"Local Optimum reached at {current_val}."

        elif algo == "restart":
            if is_stuck or not valid_neighbors:
                if restart_count < max_restarts:
                    restart_count += 1
                    current_state = initial_random_state(default_ship, default_vehicles_data, objective_type=current_objective_type)
                    score = current_state.calculate_objective()
                    total_evaluations += 1
                    current_iter += 1
                    history_objective.append(score)
                    is_stuck = False
                    best_overall_score = max(best_overall_score, score)
                    msg = f"Restart #{restart_count}/{max_restarts}: Generated new random valid state ({score})."
                else:
                    is_stuck, msg = True, f"All {max_restarts} restarts completed. Best Overall Score: {best_overall_score}."
            else:
                best_val = max(n.calculate_objective() for n in valid_neighbors)
                if best_val > current_val:
                    current_state = random.choice([n for n in valid_neighbors if n.calculate_objective() == best_val])
                    current_iter += 1
                    history_objective.append(best_val)
                    best_overall_score = max(best_overall_score, best_val)
                    msg = f"Restart #{restart_count} (Step {current_iter}): Objective climbed to {best_val}."
                else:
                    best_overall_score = max(best_overall_score, current_val)
                    if restart_count < max_restarts:
                        restart_count += 1
                        current_state = initial_random_state(default_ship, default_vehicles_data, objective_type=current_objective_type)
                        score = current_state.calculate_objective()
                        total_evaluations += 1
                        current_iter += 1
                        history_objective.append(score)
                        is_stuck = False
                        best_overall_score = max(best_overall_score, score)
                        msg = f"Restart #{restart_count}/{max_restarts}: Local peak reached. Restarting ({score})."
                    else:
                        is_stuck, msg = True, f"Restart #{restart_count} finished. Completed all {max_restarts} restarts. Global Best: {best_overall_score}."

    elif algo == "simulated_annealing":
        if current_iter >= max_iter or sa_temperature <= 0.1:
            is_stuck, msg = True, f"Simulated Annealing terminated (T={sa_temperature:.2f}, Iter {current_iter}). Best Score: {best_overall_score}."
        else:
            next_state, trial_evals, clash_state = generate_random_valid_neighbor(current_state, return_clash=True)
            total_evaluations += trial_evals
            if clash_state:
                c_grid, c_over, c_oob = clash_state.get_deck_grid()
                trial_clash = {"grid": c_grid, "overlap": c_over, "oob": c_oob}

            if next_state is None:
                sa_stuck_count += 1
                sa_history_prob.append(0.0)
                is_stuck, msg = True, f"Step {current_iter}: No valid neighbor found. Search terminated."
            else:
                next_val = next_state.calculate_objective()
                delta_E = next_val - current_val
                if delta_E > 0:
                    prob = 1.0
                    current_state, current_iter = next_state, current_iter + 1
                    history_objective.append(next_val)
                    best_overall_score = max(best_overall_score, next_val)
                    msg = f"Step {current_iter} (T={sa_temperature:.2f}): Better move accepted ({next_val}, delta_E={delta_E})."
                else:
                    prob = math.exp(delta_E / max(1e-9, sa_temperature)) if sa_temperature > 0 else 0
                    current_iter += 1
                    if random.random() < prob:
                        current_state = next_state
                        history_objective.append(next_val)
                        msg = f"Step {current_iter} (T={sa_temperature:.2f}): Accepted downhill move (P={prob:.3f}, delta_E={delta_E})."
                    else:
                        sa_stuck_count += 1
                        msg = f"Step {current_iter} (T={sa_temperature:.2f}): Rejected downhill move (P={prob:.3f}, delta_E={delta_E}). Stuck in local optima."

                sa_history_prob.append(round(prob, 4))
                sa_temperature = max(0.01, sa_temperature * cooling_rate)
                is_stuck = (sa_temperature <= 0.1 or current_iter >= max_iter)

    elif algo == "genetic_algorithm":
        if current_iter >= max_gens:
            is_stuck, msg = True, f"Genetic Algorithm terminated after reaching {max_gens} generations. Best Fitness: {best_overall_score}."
        else:
            if not ga_population or len(ga_population) != pop_size:
                ga_population = [initial_random_state(default_ship, default_vehicles_data, objective_type=current_objective_type) for _ in range(pop_size)]
                ga_population[0] = current_state.copy()

            fitness_scores = [ind.calculate_objective() for ind in ga_population]
            total_evaluations += len(ga_population)
            elite = max(ga_population, key=lambda ind: ind.calculate_objective()).copy()
            new_pop, sample_clash_ga = [elite], None

            while len(new_pop) < pop_size:
                p1, p2 = select_parent_roulette(ga_population, fitness_scores), select_parent_roulette(ga_population, fitness_scores)
                child, unrepaired = crossover(p1, p2, return_unrepaired=True)
                if unrepaired and sample_clash_ga is None:
                    sample_clash_ga = unrepaired
                new_pop.append(mutate(child, mutation_rate=mutation_rate))

            if sample_clash_ga:
                c_grid, c_over, c_oob = sample_clash_ga.get_deck_grid()
                trial_clash = {"grid": c_grid, "overlap": c_over, "oob": c_oob}

            ga_population = new_pop
            current_iter += 1
            current_state = elite
            best_val = elite.calculate_objective()
            history_objective.append(best_val)
            best_overall_score = max(best_overall_score, best_val)
            avg_fit = sum(fitness_scores) / len(fitness_scores)
            history_avg_objective.append(round(avg_fit, 2))
            is_stuck = (current_iter >= max_gens)
            msg = f"Generation {current_iter}/{max_gens}: Best Fitness = {best_val}, Population Avg = {avg_fit:.1f}."

    saved_output_file = None
    if is_stuck:
        client_elapsed = float(data.get("execution_time", 0.0))
        exec_time = client_elapsed if client_elapsed > 0 else round(max(0.0001, time.time() - (step_session_start or time.time())), 4)
        algo_meta = {
            "steepest": ("Steepest Ascent Hill-Climbing", f"Max Iter: {max_iter}"),
            "stochastic": ("Stochastic Hill-Climbing", f"Max Iter: {max_iter}"),
            "sideways": ("Hill-Climbing with Sideways Move", f"Max Sideways: {max_sideways}, Max Iter: {max_iter}"),
            "restart": ("Random Restart Hill-Climbing", f"Restarts: {max_restarts}, Max Iter/Run: {max_iter}"),
            "simulated_annealing": ("Simulated Annealing", f"T0: {sa_temperature:.1f}, Cooling: {cooling_rate}, Max Iter: {max_iter}"),
            "genetic_algorithm": ("Genetic Algorithm", f"Pop: {pop_size}, Gens: {max_gens}, Mut: {mutation_rate}")
        }
        display_name, params_str = algo_meta.get(algo, (algo, ""))
        step_metric = {
            "execution_time": exec_time, "iterations": current_iter, "evaluations": total_evaluations,
            "score": current_state.calculate_objective(), "valid": current_state.is_valid()
        }
        if algo == "simulated_annealing":
            step_metric["stuck_count"] = sa_stuck_count
            step_metric["stuck_frequency"] = round(sa_stuck_count / max(1, current_iter), 4)
            step_metric["history_acceptance_prob"] = sa_history_prob

        saved_output_file = save_run_result(
            algo_name=display_name, params_summary=params_str, metric=step_metric, state=current_state,
            history=history_objective, history_avg=history_avg_objective, objective_type=current_objective_type
        )
        msg += f" [Saved to {saved_output_file}]"

    payload = get_state_payload(message=msg, trial_clash=trial_clash)
    if saved_output_file:
        payload["saved_output_file"] = saved_output_file
    return jsonify(payload)

@app.route("/api/run", methods=["POST"])
def run_full():
    global current_state, current_iter, total_evaluations, history_objective, history_avg_objective
    global is_stuck, sideways_count, restart_count, best_overall_score
    global sa_history_prob, sa_stuck_count, current_objective_type

    data = request.get_json(silent=True) or {}
    algo = data.get("algorithm", "steepest")
    current_objective_type = data.get("objective_type", current_objective_type)

    if algo != "genetic_algorithm" and is_stuck:
        return jsonify({
            "error": "Algorithm is terminated / reach Local Optimum",
            "blocked": True, "is_stuck": True
        }), 400

    max_iter = int(data.get("max_iterations", 500))
    max_sideways = int(data.get("max_sideways_moves", 100))
    max_restarts = int(data.get("max_restarts", 5))
    initial_temp = float(data.get("initial_temperature", 1000.0))
    cooling_rate = float(data.get("cooling_rate", 0.95))
    pop_size = int(data.get("population_size", 20))
    max_gens = int(data.get("max_generations", 40))
    mutation_rate = float(data.get("mutation_rate", 0.15))

    if current_state is None:
        init_session(current_objective_type)
    else:
        current_state.objective_type = current_objective_type

    start_state = current_state.copy()
    algo_dispatch = {
        "steepest": ("Steepest Ascent Hill-Climbing", f"Max Iter: {max_iter}", lambda: steepest_ascent_hill_climbing(start_state, max_iterations=max_iter)),
        "stochastic": ("Stochastic Hill-Climbing", f"Max Iter: {max_iter}", lambda: stochastic_hill_climbing(start_state, max_iterations=max_iter)),
        "sideways": ("Hill-Climbing with Sideways Move", f"Max Sideways: {max_sideways}, Max Iter: {max_iter}", lambda: sideways_move_hill_climbing(start_state, max_sideways_moves=max_sideways, max_iterations=max_iter)),
        "restart": ("Random Restart Hill-Climbing", f"Restarts: {max_restarts}, Max Iter/Run: {max_iter}", lambda: random_restart_hill_climbing(initial_state=start_state, ship=default_ship, vehicles_data=default_vehicles_data, max_restarts=max_restarts, max_iterations_per_restart=max_iter)),
        "simulated_annealing": ("Simulated Annealing", f"T0: {initial_temp}, Cooling: {cooling_rate}, Max Iter: {max_iter}", lambda: simulated_annealing(start_state, initial_temperature=initial_temp, cooling_rate=cooling_rate, max_iterations=max_iter)),
        "genetic_algorithm": ("Genetic Algorithm", f"Pop: {pop_size}, Gens: {max_gens}, Mut: {mutation_rate}", lambda: genetic_algorithm(default_ship, default_vehicles_data, population_size=pop_size, max_generations=max_gens, mutation_rate=mutation_rate, objective_type=current_objective_type))
    }

    if algo not in algo_dispatch:
        return jsonify({"error": "Unknown algorithm"}), 400

    algo_display_name, params_summary, runner_fn = algo_dispatch[algo]
    result = runner_fn()

    current_state = result["final_state"]
    current_iter = result["total_iterations"]
    total_evaluations = result["total_evaluations"]
    history_objective = result["history_objective"]
    history_avg_objective = result.get("history_avg_objective", [])
    best_overall_score = result["best_objective"]
    is_stuck = True

    run_metric = {
        "algorithm": algo_display_name, "parameters": params_summary, "objective_type": current_objective_type,
        "score": result["best_objective"], "iterations": result["total_iterations"], "evaluations": result["total_evaluations"],
        "execution_time": round(result["execution_time"], 4), "valid": result["valid"],
        "history_best": result.get("history_best_objective", result["history_objective"]),
        "history_avg": result.get("history_avg_objective", [])
    }

    if algo == "simulated_annealing":
        sa_stuck_count = result.get("stuck_count", 0)
        sa_history_prob = result.get("history_acceptance_prob", [])
        run_metric["stuck_count"] = sa_stuck_count
        run_metric["stuck_frequency"] = result.get("stuck_frequency", 0.0)
        run_metric["history_acceptance_prob"] = sa_history_prob

    saved_file = save_run_result(
        algo_name=algo_display_name, params_summary=params_summary, metric=run_metric, state=current_state,
        history=history_objective, history_avg=history_avg_objective, objective_type=current_objective_type
    )

    response_payload = get_state_payload(
        message=f"{algo_display_name} executed in {result['execution_time']:.4f}s with score {result['best_objective']}. Result saved at {saved_file}."
    )
    response_payload["run_metric"] = run_metric
    response_payload["saved_output_file"] = saved_file
    return jsonify(response_payload)

@app.route("/api/clear_output", methods=["POST"])
def clear_output():
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f != ".gitkeep":
                try:
                    os.remove(os.path.join(OUTPUT_DIR, f))
                except OSError:
                    pass
    return jsonify({"status": "cleared"})

@app.route("/api/output_files", methods=["GET"])
def list_output_files():
    files = []
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith((".json", ".txt")) and not f.startswith("."):
                f_path = os.path.join(OUTPUT_DIR, f)
                files.append({"name": f, "size": os.path.getsize(f_path), "created": os.path.getmtime(f_path)})
        files.sort(key=lambda x: x["created"], reverse=True)
    return jsonify({"files": files})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
