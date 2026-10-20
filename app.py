import time
import math
import random
from flask import Flask, render_template, jsonify, request
from model import Ship, Vehicle
from tetangga import initial_random_state, generate_all_neighbors, generate_random_valid_neighbor
from hill_climbing import steepest_ascent_hill_climbing
from genetic_algorithm import genetic_algorithm, crossover, mutate, select_parent_roulette

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
    {"v_id": "V8", "w": 3, "l": 2, "fee": 8000, "weight": 35, "eta": 3},
]

current_state = None
current_iter = 0
total_evaluations = 0
history_objective = []
history_avg_objective = []
is_stuck = False
sideways_count = 0
restart_count = 0
best_overall_score = float('-inf')

sa_temperature = 1000.0

ga_population = []

def init_session():
    global current_state, current_iter, total_evaluations, history_objective, history_avg_objective
    global is_stuck, sideways_count, restart_count, best_overall_score
    global sa_temperature, ga_population

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
    sa_temperature = 1000.0
    ga_population = []

def get_state_payload(message="", trial_clash=None):
    global current_state, current_iter, total_evaluations, history_objective, history_avg_objective
    global is_stuck, sideways_count, restart_count, best_overall_score, sa_temperature

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
        "sa_temperature": sa_temperature,
        "vehicles": info["vehicles"],
        "history": history_objective,
        "history_avg": history_avg_objective,
        "trial_clash": trial_clash,
        "ship_w": default_ship.w,
        "ship_l": default_ship.l,
        "message": message
    }

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
    global sa_temperature, ga_population

    if current_state is None:
        init_session()

    data = request.get_json() or {}
    algo = data.get("algorithm", "steepest")
    max_iterations = int(data.get("max_iterations", 500))
    mutation_rate = float(data.get("mutation_rate", 0.15))
    population_size = int(data.get("population_size", 20))
    max_generations = int(data.get("max_generations", 40))

    current_val = current_state.calculate_objective()
    msg = ""
    trial_clash = None

    if is_stuck:
        return jsonify(get_state_payload(message=f"Algorithm already terminated at state with objective {current_val}."))

    if algo == "steepest":
        if current_iter >= max_iterations:
            is_stuck = True
            msg = f"Maximum iterations ({max_iterations}) reached. Search terminated."
        else:
            valid_neighbors, trial_evals, clash_state = generate_all_neighbors(current_state, return_clash=True)
            total_evaluations += trial_evals

            if clash_state:
                c_grid, c_over, c_oob = clash_state.get_deck_grid()
                trial_clash = {"grid": c_grid, "overlap": c_over, "oob": c_oob}

            if not valid_neighbors:
                is_stuck = True
                msg = f"No valid neighbors generated. Trapped at local optimum ({current_val})."
            else:
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
                    current_iter += 1
                    history_objective.append(best_val)
                    if current_iter >= max_iterations:
                        is_stuck = True
                        msg = f"Step {current_iter}: Reached max iterations ({max_iterations}). Final Score: {best_val}."
                    else:
                        is_stuck = False
                        msg = f"Step {current_iter}: Moved to strictly better neighbor ({best_val}, +{best_val - current_val})."
                else:
                    is_stuck = True
                    msg = f"Trapped at Local Optimum / Peak (Objective: {current_val}). No strictly better neighbor."

    elif algo == "genetic_algorithm":
        if current_iter >= max_generations:
            is_stuck = True
            msg = f"Genetic Algorithm terminated after reaching {max_generations} generations. Best Fitness: {best_overall_score}."
        else:
            if not ga_population or len(ga_population) != population_size:
                ga_population = [initial_random_state(default_ship, default_vehicles_data) for _ in range(population_size)]
                ga_population[0] = current_state.copy()

            fitness_scores = [ind.calculate_objective() for ind in ga_population]
            total_evaluations += len(ga_population)

            gen_best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            elite = ga_population[gen_best_idx].copy()

            new_pop = [elite]
            sample_clash_ga = None

            while len(new_pop) < population_size:
                p1 = select_parent_roulette(ga_population, fitness_scores)
                p2 = select_parent_roulette(ga_population, fitness_scores)
                child, unrepaired = crossover(p1, p2, return_unrepaired=True)
                if unrepaired and sample_clash_ga is None:
                    sample_clash_ga = unrepaired
                child = mutate(child, mutation_rate=mutation_rate)
                new_pop.append(child)

            if sample_clash_ga:
                c_grid, c_over, c_oob = sample_clash_ga.get_deck_grid()
                trial_clash = {"grid": c_grid, "overlap": c_over, "oob": c_oob}

            ga_population = new_pop
            current_iter += 1
            current_state = elite
            best_val = elite.calculate_objective()
            history_objective.append(best_val)
            if best_val > best_overall_score:
                best_overall_score = best_val

            avg_fit = sum(fitness_scores) / len(fitness_scores)
            history_avg_objective.append(round(avg_fit, 2))

            if current_iter >= max_generations:
                is_stuck = True
                msg = f"Generation {current_iter}/{max_generations} [TERMINATED]: Best Fitness = {best_val}, Population Avg = {avg_fit:.1f}."
            else:
                is_stuck = False
                msg = f"Generation {current_iter}/{max_generations}: Best Fitness = {best_val}, Population Avg = {avg_fit:.1f}."

    return jsonify(get_state_payload(message=msg, trial_clash=trial_clash))

@app.route("/api/run", methods=["POST"])
def run_full():
    global current_state, current_iter, total_evaluations, history_objective
    global is_stuck, sideways_count, restart_count, best_overall_score

    data = request.get_json() or {}
    algo = data.get("algorithm", "steepest")

    max_iterations = int(data.get("max_iterations", 500))
    pop_size = int(data.get("population_size", 20))
    max_gens = int(data.get("max_generations", 40))
    mutation_rate = float(data.get("mutation_rate", 0.15))

    if current_state is None:
        init_session()
    start_state = current_state.copy()

    algo_display_name = ""
    params_summary = ""

    if algo == "steepest":
        algo_display_name = "Steepest Ascent Hill-Climbing"
        params_summary = f"Max Iter: {max_iterations}"
        result = steepest_ascent_hill_climbing(start_state, max_iterations=max_iterations)

    elif algo == "genetic_algorithm":
        algo_display_name = "Genetic Algorithm"
        params_summary = f"Pop: {pop_size}, Gens: {max_gens}, Mut: {mutation_rate}"
        result = genetic_algorithm(
            default_ship,
            default_vehicles_data,
            population_size=pop_size,
            max_generations=max_gens,
            mutation_rate=mutation_rate
        )
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
        message=f"{algo_display_name} execution completed in {result['execution_time']:.4f}s with score {result['best_objective']}."
    )
    response_payload["run_metric"] = {
        "algorithm": algo_display_name,
        "parameters": params_summary,
        "score": result["best_objective"],
        "iterations": result["total_iterations"],
        "evaluations": result["total_evaluations"],
        "execution_time": round(result["execution_time"], 4),
        "valid": result["valid"],
        "history_best": result.get("history_best_objective", result["history_objective"]),
        "history_avg": result.get("history_avg_objective", [])
    }
    return jsonify(response_payload)

if __name__ == "__main__":
    app.run(debug=True, port=5000)