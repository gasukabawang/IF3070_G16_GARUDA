import time
import math
import random
from flask import Flask, render_template, jsonify, request
from model import Ship, Vehicle
from tetangga import initial_random_state, generate_all_neighbors
from hill_climbing import (stochastic_hill_climbing, random_restart_hill_climbing)

app = Flask(__name__)

default_ship = Ship(w=6, l=6, MaxCapacity=100)

default_vehicles_data = [
    {"v_id": "V1", "w": 2, "l": 2, "fee": 5000, "weight": 20, "eta": 1},
    {"v_id": "V2", "w": 2, "l": 3, "fee": 7000, "weight": 30, "eta": 1},
    {"v_id": "V3", "w": 1, "l": 4, "fee": 4000, "weight": 15, "eta": 2},
    {"v_id": "V4", "w": 2, "l": 2, "fee": 6000, "weight": 25, "eta": 3},
    {"v_id": "V5", "w": 3, "l": 1, "fee": 5500, "weight": 18, "eta": 2},
    {"v_id": "V6", "w": 1, "l": 2, "fee": 3000, "weight": 10, "eta": 4},
    {"v_id": "V7", "w": 2, "l": 1, "fee": 3500, "weight": 12, "eta": 5},
    {"v_id": "V8", "w": 3, "l": 2, "fee": 8000, "weight": 35, "eta": 3}]

current_state = None
current_iter = 0
total_evaluations = 0
history_objective = []
history_avg_objective = []
is_stuck = False
sideways_count = 0
restart_count = 0
best_overall_score = float('-inf')

def init_session():
    global current_state, current_iter, total_evaluations, history_objective, history_avg_objective
    global is_stuck, sideways_count, restart_count, best_overall_score

    current_state = initial_random_state(default_ship, default_vehicles_data)
    current_iter = 0
    total_evaluations = 1
    score = current_state.calculate_objective()
    history_objective = [score]
    history_avg_objective = [score]
    is_stuck = False
    sideways_count = 0
    restart_count = 0
    best_overall_score = score

def get_state_payload(message="", trial_clash=None):
    global current_state, current_iter, total_evaluations, history_objective, history_avg_objective
    global is_stuck, sideways_count, restart_count, best_overall_score

    if current_state is None:
        init_session()

    grid_matrix, overlap_count, oob_count = current_state.get_deck_grid()
    score = current_state.calculate_objective()
    info = current_state.get_info()

    return {
        "grid": grid_matrix,
        "score": score,
        "total_fee": info["total_fee"],
        "total_weight": info["total_weight"],
        "MaxCapacity": info["MaxCapacity"],
        "valid": info["valid"],
        "overlap": overlap_count,
        "oob": oob_count,
        "iter": current_iter,
        "evaluations": total_evaluations,
        "is_stuck": is_stuck,
        "sideways_count": sideways_count,
        "restart_count": restart_count,
        "best_overall_score": best_overall_score if best_overall_score != float('-inf') else score,
        "vehicles": info["vehicles"],
        "history": history_objective,
        "history_avg": history_avg_objective,
        "trial_clash": trial_clash,
        "ship_w": default_ship.w,
        "ship_l": default_ship.l,
        "message": message}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/reset", methods=["POST"])
def reset():
    init_session()
    return jsonify(get_state_payload(message="Initial random state successfully generated. State is 100% valid."))

@app.route("/api/step", methods=["POST"])
def step():
    global current_state, current_iter, total_evaluations, history_objective
    global is_stuck, sideways_count, restart_count, best_overall_score

    if current_state is None:
        init_session()

    data = request.get_json() or {}
    algo = data.get("algorithm", "stochastic")
    max_iterations = int(data.get("max_iterations", 500))
    max_restarts = int(data.get("max_restarts", 5))

    current_val = current_state.calculate_objective()
    msg = ""
    trial_clash = None

    if is_stuck and algo != "restart":
        return jsonify(get_state_payload(message=f"Algorithm already terminated at state with objective {current_val}."))

    if algo == "stochastic":
        if current_iter >= max_iterations:
            is_stuck = True
            msg = f"Maximum iterations ({max_iterations}) reached. Search terminated."
        else:
            valid_neighbors, trial_evals, clash_state = generate_all_neighbors(current_state, return_clash=True)
            total_evaluations += trial_evals

            if clash_state:
                c_grid, c_over, c_oob = clash_state.get_deck_grid()
                trial_clash = {"grid": c_grid, "overlap": c_over, "oob": c_oob}

            better_neighbors = [(n, n.calculate_objective()) for n in valid_neighbors if n.calculate_objective() > current_val]

            if better_neighbors:
                chosen_neighbor, chosen_val = random.choice(better_neighbors)
                current_state = chosen_neighbor
                current_iter += 1
                history_objective.append(chosen_val)
                if current_iter >= max_iterations:
                    is_stuck = True
                    msg = f"Step {current_iter}: Reached max iterations ({max_iterations}). Final Score: {chosen_val}."
                else:
                    is_stuck = False
                    msg = f"Step {current_iter}: Stochastically selected improving neighbor ({chosen_val}, +{chosen_val - current_val})."
            else:
                is_stuck = True
                msg = f"Trapped at Local Optimum (Objective: {current_val}). No strictly better neighbor."

    elif algo == "restart":
        if is_stuck:
            if restart_count < max_restarts:
                restart_count += 1
                current_state = initial_random_state(default_ship, default_vehicles_data)
                score = current_state.calculate_objective()
                total_evaluations += 1
                current_iter += 1
                history_objective.append(score)
                is_stuck = False
                if score > best_overall_score:
                    best_overall_score = score
                msg = f"Restart #{restart_count}/{max_restarts}: Generated new random valid initial state (Objective: {score})."
            else:
                is_stuck = True
                msg = f"All {max_restarts} restarts completed. Best Overall Score: {best_overall_score}."
        else:
            valid_neighbors, trial_evals, clash_state = generate_all_neighbors(current_state, return_clash=True)
            total_evaluations += trial_evals

            if clash_state:
                c_grid, c_over, c_oob = clash_state.get_deck_grid()
                trial_clash = {"grid": c_grid, "overlap": c_over, "oob": c_oob}

            best_val = float('-inf')
            best_candidates = []
            for n in valid_neighbors:
                val = n.calculate_objective()
                if val > best_val:
                    best_val = val
                    best_candidates = [n]
                elif val == best_val:
                    best_candidates.append(n)

            if best_candidates and best_val > current_val:
                current_state = random.choice(best_candidates)
                current_iter += 1
                history_objective.append(best_val)
                is_stuck = False
                if best_val > best_overall_score:
                    best_overall_score = best_val
                msg = f"Restart #{restart_count} (Step {current_iter}): Objective climbed to {best_val}."
            else:
                if current_val > best_overall_score:
                    best_overall_score = current_val

                if restart_count < max_restarts:
                    restart_count += 1
                    current_state = initial_random_state(default_ship, default_vehicles_data)
                    score = current_state.calculate_objective()
                    total_evaluations += 1
                    current_iter += 1
                    history_objective.append(score)
                    is_stuck = False
                    if score > best_overall_score:
                        best_overall_score = score
                    msg = f"Restart #{restart_count}/{max_restarts}: Local peak reached. Restarting with new random state (Objective: {score})."
                else:
                    is_stuck = True
                    msg = f"Restart #{restart_count} finished. Completed all {max_restarts} restarts. Global Best: {best_overall_score}."

    return jsonify(get_state_payload(message=msg, trial_clash=trial_clash))

@app.route("/api/run", methods=["POST"])
def run_full():
    global current_state, current_iter, total_evaluations, history_objective
    global is_stuck, sideways_count, restart_count, best_overall_score

    data = request.get_json() or {}
    algo = data.get("algorithm", "stochastic")

    max_iterations = int(data.get("max_iterations", 500))
    max_restarts = int(data.get("max_restarts", 5))

    if current_state is None:
        init_session()
    start_state = current_state.copy()

    start_exec = time.time()
    algo_display_name = ""
    params_summary = ""

    if algo == "stochastic":
        algo_display_name = "Stochastic Hill-Climbing"
        params_summary = f"Max Iter: {max_iterations}"
        result = stochastic_hill_climbing(start_state, max_iterations=max_iterations)

    elif algo == "restart":
        algo_display_name = "Random Restart Hill-Climbing"
        params_summary = f"Restarts: {max_restarts}, Max Iter/Run: {max_iterations}"
        result = random_restart_hill_climbing(
            initial_state=start_state,
            ship=default_ship,
            vehicles_data=default_vehicles_data,
            max_restarts=max_restarts,
            max_iterations_per_restart=max_iterations)
    else:
        return jsonify({"error": "Unknown algorithm"}), 400

    current_state = result["final_state"]
    current_iter = result["total_iterations"]
    total_evaluations = result["total_evaluations"]
    history_objective = result["history_objective"]
    history_avg_objective = result.get("history_avg_objective", [])
    best_overall_score = result["best_objective"]
    is_stuck = True

    response_payload = get_state_payload(
        message=f"{algo_display_name} execution completed in {result['execution_time']:.4f}s with score {result['best_objective']}.")
    response_payload["run_metric"] = {
        "algorithm": algo_display_name,
        "parameters": params_summary,
        "score": result["best_objective"],
        "iterations": result["total_iterations"],
        "evaluations": result["total_evaluations"],
        "execution_time": round(result["execution_time"], 4),
        "valid": result["valid"],
        "history_best": result.get("history_best_objective", result["history_objective"]),
        "history_avg": result.get("history_avg_objective", [])}

    return jsonify(response_payload)

if __name__ == "__main__":
    app.run(debug=True, port=5000)