import time
import math
import random
from flask import Flask, render_template, jsonify, request
from model import Ship, Vehicle
from tetangga import initial_random_state, generate_all_neighbors, generate_random_valid_neighbor
from hill_climbing import (
    steepest_ascent_hill_climbing,
    stochastic_hill_climbing,
    sideways_move_hill_climbing,
    random_restart_hill_climbing)
from simulated_annealing import simulated_annealing
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
    return jsonify(get_state_payload(message="Initial random state successfully generated, state is valid!"))

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
    max_sideways = int(data.get("max_sideways_moves", 100))
    max_restarts = int(data.get("max_restarts", 5))
    cooling_rate = float(data.get("cooling_rate", 0.95))
    mutation_rate = float(data.get("mutation_rate", 0.15))
    population_size = int(data.get("population_size", 20))
    max_generations = int(data.get("max_generations", 40))

    current_val = current_state.calculate_objective()
    msg = ""
    trial_clash = None

    if is_stuck and algo != "restart":
        return jsonify(get_state_payload(message=f"Algorithm terminated at state with objective {current_val}."))

    if algo in ["steepest", "stochastic", "sideways", "restart"]:
        if current_iter >= max_iterations and algo != "restart":
            is_stuck = True
            msg = f"Maximum iterations ({max_iterations}) reached, thus search terminated."
            return jsonify(get_state_payload(message=msg))

        valid_neighbors, trial_evals, clash_state = generate_all_neighbors(current_state, return_clash=True)
        total_evaluations += trial_evals

        if clash_state:
            c_grid, c_over, c_oob = clash_state.get_deck_grid()
            trial_clash = {"grid": c_grid, "overlap": c_over, "oob": c_oob}

        if algo == "steepest":
            if not valid_neighbors:
                is_stuck = True
                msg = f"No valid neighbors generated. Trapped at local optimum ({current_val})."
            else:
                best_val = max(n.calculate_objective() for n in valid_neighbors)
                if best_val > current_val:
                    best_candidates = [n for n in valid_neighbors if n.calculate_objective() == best_val]
                    current_state = random.choice(best_candidates)
                    current_iter += 1
                    history_objective.append(best_val)
                    is_stuck = (current_iter >= max_iterations)
                    msg = f"Step {current_iter}: Moved to strictly better neighbor ({best_val}, +{best_val - current_val})."
                else:
                    is_stuck = True
                    msg = f"Trapped at Local Optimum (Objective: {current_val}). There is no better neighbor."

        elif algo == "stochastic":
            better_neighbors = [(n, val) for n in valid_neighbors if (val := n.calculate_objective()) > current_val]
            if better_neighbors:
                current_state, chosen_val = random.choice(better_neighbors)
                current_iter += 1
                history_objective.append(chosen_val)
                is_stuck = (current_iter >= max_iterations)
                msg = f"Step {current_iter}: Stochastically selected improving neighbor ({chosen_val}, +{chosen_val - current_val})."
            else:
                is_stuck = True
                msg = f"Trapped at Local Optimum (Objective: {current_val}). There is no better neighbor."

        elif algo == "sideways":
            if not valid_neighbors:
                is_stuck = True
                msg = f"No valid neighbors generated. Search halted at {current_val}."
            else:
                best_val = max(n.calculate_objective() for n in valid_neighbors)
                best_candidates = [n for n in valid_neighbors if n.calculate_objective() == best_val]
                best_neighbor = random.choice(best_candidates)

                if best_val > current_val:
                    current_state = best_neighbor
                    sideways_count = 0
                    current_iter += 1
                    history_objective.append(best_val)
                    is_stuck = (current_iter >= max_iterations)
                    msg = f"Step {current_iter}: Improved objective to {best_val} (+{best_val - current_val})."
                elif best_val == current_val and sideways_count < max_sideways:
                    current_state = best_neighbor
                    sideways_count += 1
                    current_iter += 1
                    history_objective.append(best_val)
                    is_stuck = (current_iter >= max_iterations or sideways_count >= max_sideways)
                    msg = f"Step {current_iter}: Sideways move #{sideways_count}/{max_sideways} (Objective: {best_val})."
                else:
                    is_stuck = True
                    msg = f"Max sideways moves limit reached ({sideways_count}/{max_sideways}). Search halted." if sideways_count >= max_sideways else f"Local optimum reached at {current_val}."

        elif algo == "restart":
            if is_stuck or not valid_neighbors:
                if restart_count < max_restarts:
                    restart_count += 1
                    current_state = initial_random_state(default_ship, default_vehicles_data)
                    score = current_state.calculate_objective()
                    total_evaluations += 1
                    current_iter += 1
                    history_objective.append(score)
                    is_stuck = False
                    best_overall_score = max(best_overall_score, score)
                    msg = f"Restart #{restart_count}/{max_restarts}: Generated new random valid initial state ({score})."
                else:
                    is_stuck = True
                    msg = f"All {max_restarts} restarts completed. Best Overall Score: {best_overall_score}."
            else:
                best_val = max(n.calculate_objective() for n in valid_neighbors)
                if best_val > current_val:
                    best_candidates = [n for n in valid_neighbors if n.calculate_objective() == best_val]
                    current_state = random.choice(best_candidates)
                    current_iter += 1
                    history_objective.append(best_val)
                    best_overall_score = max(best_overall_score, best_val)
                    msg = f"Restart #{restart_count} (Step {current_iter}): Objective climbed to {best_val}."
                else:
                    best_overall_score = max(best_overall_score, current_val)
                    if restart_count < max_restarts:
                        restart_count += 1
                        current_state = initial_random_state(default_ship, default_vehicles_data)
                        score = current_state.calculate_objective()
                        total_evaluations += 1
                        current_iter += 1
                        history_objective.append(score)
                        is_stuck = False
                        best_overall_score = max(best_overall_score, score)
                        msg = f"Restart #{restart_count}/{max_restarts}: Local peak reached. Restarting ({score})."
                    else:
                        is_stuck = True
                        msg = f"Restart #{restart_count} finished. Completed all {max_restarts} restarts. Global Best: {best_overall_score}."

    elif algo == "simulated_annealing":
        if current_iter >= max_iterations or sa_temperature <= 0.1:
            is_stuck = True
            msg = f"Simulated Annealing terminated (T={sa_temperature:.2f}, Iter {current_iter}). Best Score: {best_overall_score}."
        else:
            next_state, trial_evals, clash_state = generate_random_valid_neighbor(current_state, return_clash=True)
            total_evaluations += trial_evals

            if clash_state:
                c_grid, c_over, c_oob = clash_state.get_deck_grid()
                trial_clash = {"grid": c_grid, "overlap": c_over, "oob": c_oob}

            if next_state is None:
                is_stuck = True
                msg = f"Step {current_iter}: No valid neighbor found. Search terminated."
            else:
                next_val = next_state.calculate_objective()
                delta_E = next_val - current_val

                if delta_E > 0:
                    current_state = next_state
                    current_iter += 1
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
                        msg = f"Step {current_iter} (T={sa_temperature:.2f}): Rejected downhill move (P={prob:.3f}, delta_E={delta_E})."

                sa_temperature = max(0.01, sa_temperature * cooling_rate)
                is_stuck = (sa_temperature <= 0.1 or current_iter >= max_iterations)

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

            elite = max(ga_population, key=lambda ind: ind.calculate_objective()).copy()
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
            best_overall_score = max(best_overall_score, best_val)

            avg_fit = sum(fitness_scores) / len(fitness_scores)
            history_avg_objective.append(round(avg_fit, 2))

            is_stuck = (current_iter >= max_generations)
            msg = f"Generation {current_iter}/{max_generations}: Best Fitness = {best_val}, Population Avg = {avg_fit:.1f}."
    return jsonify(get_state_payload(message=msg, trial_clash=trial_clash))

@app.route("/api/run", methods=["POST"])
def run_full():
    global current_state, current_iter, total_evaluations, history_objective
    global is_stuck, sideways_count, restart_count, best_overall_score

    data = request.get_json() or {}
    algo = data.get("algorithm", "steepest")

    max_iterations = int(data.get("max_iterations", 500))
    max_sideways = int(data.get("max_sideways_moves", 100))
    max_restarts = int(data.get("max_restarts", 5))
    initial_temp = float(data.get("initial_temperature", 1000.0))
    cooling_rate = float(data.get("cooling_rate", 0.95))
    pop_size = int(data.get("population_size", 20))
    max_gens = int(data.get("max_generations", 40))
    mutation_rate = float(data.get("mutation_rate", 0.15))

    if current_state is None:
        init_session()
    start_state = current_state.copy()

    algo_dispatch = {
        "steepest": (
            "Steepest Ascent Hill-Climbing",
            f"Max Iter: {max_iterations}",
            lambda: steepest_ascent_hill_climbing(start_state, max_iterations=max_iterations)
        ),
        "stochastic": (
            "Stochastic Hill-Climbing",
            f"Max Iter: {max_iterations}",
            lambda: stochastic_hill_climbing(start_state, max_iterations=max_iterations)
        ),
        "sideways": (
            "Hill-Climbing with Sideways Move",
            f"Max Sideways: {max_sideways}, Max Iter: {max_iterations}",
            lambda: sideways_move_hill_climbing(start_state, max_sideways_moves=max_sideways, max_iterations=max_iterations)
        ),
        "restart": (
            "Random Restart Hill-Climbing",
            f"Restarts: {max_restarts}, Max Iter/Run: {max_iterations}",
            lambda: random_restart_hill_climbing(
                initial_state=start_state,
                ship=default_ship,
                vehicles_data=default_vehicles_data,
                max_restarts=max_restarts,
                max_iterations_per_restart=max_iterations
            )
        ),
        "simulated_annealing": (
            "Simulated Annealing",
            f"T0: {initial_temp}, Cooling: {cooling_rate}, Max Iter: {max_iterations}",
            lambda: simulated_annealing(
                start_state,
                initial_temperature=initial_temp,
                cooling_rate=cooling_rate,
                max_iterations=max_iterations
            )
        ),
        "genetic_algorithm": (
            "Genetic Algorithm",
            f"Pop: {pop_size}, Gens: {max_gens}, Mut: {mutation_rate}",
            lambda: genetic_algorithm(
                default_ship,
                default_vehicles_data,
                population_size=pop_size,
                max_generations=max_gens,
                mutation_rate=mutation_rate
            )
        )
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