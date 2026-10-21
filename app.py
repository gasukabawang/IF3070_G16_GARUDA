import math
import random
import time
from flask import Flask, jsonify, render_template, request
from model import Ship
from tetangga import generate_all_neighbors, generate_random_valid_neighbor, initial_random_state
from hill_climbing import sideways_move_hill_climbing
from simulated_annealing import simulated_annealing

app = Flask(__name__)

# Ship configuration & vehicle dataset
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

# Session state
current_state = None
current_iter = 0
total_evaluations = 0
history_objective = []
is_stuck = False
sideways_count = 0
best_overall_score = float("-inf")
sa_temperature = 1000.0


def init_session():
    global current_state, current_iter, total_evaluations, history_objective
    global is_stuck, sideways_count, best_overall_score, sa_temperature

    current_state = initial_random_state(default_ship, default_vehicles_data)
    current_iter = 0
    total_evaluations = 1
    score = current_state.calculate_objective()
    history_objective = [score]
    is_stuck = False
    sideways_count = 0
    best_overall_score = score
    sa_temperature = 1000.0


def get_state_payload(message="", trial_clash=None):
    global current_state, current_iter, total_evaluations, history_objective
    global is_stuck, sideways_count, best_overall_score, sa_temperature

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
        "best_overall_score": best_overall_score if best_overall_score != float("-inf") else score,
        "sa_temperature": sa_temperature,
        "vehicles": info["vehicles"],
        "history": history_objective,
        "trial_clash": trial_clash,
        "ship_w": default_ship.w,
        "ship_l": default_ship.l,
        "message": message,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reset", methods=["POST"])
def reset():
    init_session()
    return jsonify(get_state_payload(message="Initial random state generated."))


@app.route("/api/step", methods=["POST"])
def step():
    global current_state, current_iter, total_evaluations, history_objective
    global is_stuck, sideways_count, best_overall_score, sa_temperature

    if current_state is None:
        init_session()

    data = request.get_json() or {}
    algo = data.get("algorithm", "sideways")
    max_iterations = int(data.get("max_iterations", 500))
    max_sideways = int(data.get("max_sideways_moves", 100))
    cooling_rate = float(data.get("cooling_rate", 0.95))

    current_val = current_state.calculate_objective()
    msg = ""
    trial_clash = None

    if is_stuck:
        return jsonify(get_state_payload(message=f"Search halted at score {current_val}."))

    if algo == "sideways":
        if current_iter >= max_iterations:
            is_stuck = True
            msg = f"Reached max iterations ({max_iterations}). Final score: {current_val}."
        else:
            valid_neighbors, trial_evals, clash_state = generate_all_neighbors(current_state, return_clash=True)
            total_evaluations += trial_evals

            if clash_state:
                c_grid, c_over, c_oob = clash_state.get_deck_grid()
                trial_clash = {"grid": c_grid, "overlap": c_over, "oob": c_oob}

            if not valid_neighbors:
                is_stuck = True
                msg = f"No valid neighbors found. Search halted at {current_val}."
            else:
                best_val = float("-inf")
                best_candidates = []
                for n in valid_neighbors:
                    val = n.calculate_objective()
                    if val > best_val:
                        best_val = val
                        best_candidates = [n]
                    elif val == best_val:
                        best_candidates.append(n)

                best_neighbor = random.choice(best_candidates)

                if best_val > current_val:
                    current_state = best_neighbor
                    sideways_count = 0
                    current_iter += 1
                    history_objective.append(best_val)
                    is_stuck = current_iter >= max_iterations
                    msg = f"Step {current_iter}: Improved score to {best_val} (+{best_val - current_val})."
                elif best_val == current_val and sideways_count < max_sideways:
                    current_state = best_neighbor
                    sideways_count += 1
                    current_iter += 1
                    history_objective.append(best_val)
                    is_stuck = current_iter >= max_iterations or sideways_count >= max_sideways
                    msg = f"Step {current_iter}: Sideways move #{sideways_count}/{max_sideways} (Score: {best_val})."
                else:
                    is_stuck = True
                    msg = f"Local optimum reached at score {current_val}."

    elif algo == "simulated_annealing":
        if current_iter >= max_iterations or sa_temperature <= 0.1:
            is_stuck = True
            msg = f"Simulated Annealing terminated (T={sa_temperature:.2f}, Iter {current_iter}). Best: {best_overall_score}."
        else:
            next_state, trial_evals, clash_state = generate_random_valid_neighbor(current_state, return_clash=True)
            total_evaluations += trial_evals

            if clash_state:
                c_grid, c_over, c_oob = clash_state.get_deck_grid()
                trial_clash = {"grid": c_grid, "overlap": c_over, "oob": c_oob}

            if next_state is None:
                is_stuck = True
                msg = f"Step {current_iter}: No valid neighbor found. Search stopped."
            else:
                next_val = next_state.calculate_objective()
                delta_e = next_val - current_val

                prob = 1.0 if delta_e > 0 else (math.exp(delta_e / max(1e-9, sa_temperature)) if sa_temperature > 0 else 0)
                accepted = delta_e > 0 or random.random() < prob

                current_iter += 1
                if accepted:
                    current_state = next_state
                    history_objective.append(next_val)
                    if next_val > best_overall_score:
                        best_overall_score = next_val
                    msg = f"Step {current_iter} (T={sa_temperature:.2f}): Accepted move to {next_val} (delta={delta_e:+d})."
                else:
                    msg = f"Step {current_iter} (T={sa_temperature:.2f}): Rejected move (P={prob:.3f}, delta={delta_e:+d})."

                sa_temperature = max(0.01, sa_temperature * cooling_rate)
                is_stuck = sa_temperature <= 0.1 or current_iter >= max_iterations
                if is_stuck:
                    msg += f" [Done: T={sa_temperature:.2f}]"
    else:
        return jsonify({"error": "Unknown algorithm"}), 400

    return jsonify(get_state_payload(message=msg, trial_clash=trial_clash))


@app.route("/api/run", methods=["POST"])
def run_full():
    global current_state, current_iter, total_evaluations, history_objective
    global is_stuck, sideways_count, best_overall_score

    data = request.get_json() or {}
    algo = data.get("algorithm", "sideways")

    max_iterations = int(data.get("max_iterations", 500))
    max_sideways = int(data.get("max_sideways_moves", 100))
    initial_temp = float(data.get("initial_temperature", 1000.0))
    cooling_rate = float(data.get("cooling_rate", 0.95))

    if current_state is None:
        init_session()
    start_state = current_state.copy()

    if algo == "sideways":
        algo_name = "Hill-Climbing with Sideways Move"
        params_summary = f"Max Sideways: {max_sideways}, Max Iter: {max_iterations}"
        result = sideways_move_hill_climbing(start_state, max_sideways_moves=max_sideways, max_iterations=max_iterations)
    elif algo == "simulated_annealing":
        algo_name = "Simulated Annealing"
        params_summary = f"T0: {initial_temp}, Cooling: {cooling_rate}, Max Iter: {max_iterations}"
        result = simulated_annealing(
            start_state,
            initial_temperature=initial_temp,
            cooling_rate=cooling_rate,
            max_iterations=max_iterations,
        )
    else:
        return jsonify({"error": "Unknown algorithm"}), 400

    current_state = result["final_state"]
    current_iter = result["total_iterations"]
    total_evaluations = result.get("total_evaluations", 0)
    history_objective = result["history_objective"]
    best_overall_score = result["best_objective"]
    is_stuck = True

    payload = get_state_payload(
        message=f"{algo_name} finished in {result['execution_time']:.4f}s with score {result['best_objective']}."
    )
    payload["run_metric"] = {
        "algorithm": algo_name,
        "parameters": params_summary,
        "score": result["best_objective"],
        "iterations": result["total_iterations"],
        "evaluations": result.get("total_evaluations", 0),
        "execution_time": round(result["execution_time"], 4),
        "valid": result.get("valid", current_state.is_valid() if current_state else True),
        "history_best": result.get("history_best_objective", result["history_objective"]),
    }

    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True, port=5000)