import time
import random
from model import State, Vehicle
from tetangga import initial_random_state

def crossover(parent1, parent2, return_unrepaired=False):
    n = len(parent1.vehicles)
    if n <= 1:
        child = parent1.copy()
        if return_unrepaired:
            return child, None
        return child

    cut_point = random.randint(1, n - 1)
    child_vehicles = []

    for i in range(n):
        if i < cut_point:
            child_vehicles.append(parent1.vehicles[i].copy())
        else:
            child_vehicles.append(parent2.vehicles[i].copy())

    child_state = State(parent1.ship, child_vehicles)
    unrepaired = child_state.copy() if return_unrepaired and not child_state.is_valid() else None
    child_state.repair_to_valid()
    
    if return_unrepaired:
        return child_state, unrepaired
    return child_state

def mutate(state, mutation_rate=0.15):
    if random.random() > mutation_rate or not state.vehicles:
        return state

    idx = random.randint(0, len(state.vehicles) - 1)
    target = state.vehicles[idx]

    if target.is_loaded:
        action = random.choice(["move", "rotate", "unload"])
        if action == "move":
            target.x = max(0, min(state.ship.w - 1, target.x + random.choice([-1, 1])))
            target.y = max(0, min(state.ship.l - 1, target.y + random.choice([-1, 1])))
        elif action == "rotate":
            target.orientation = "Vertical" if target.orientation == "Horizontal" else "Horizontal"
        elif action == "unload":
            target.is_loaded = False
    else:
        target.is_loaded = True
        target.orientation = random.choice(["Horizontal", "Vertical"])
        w_eff, l_eff = target.get_dimensions()
        max_x = max(0, state.ship.w - w_eff)
        max_y = max(0, state.ship.l - l_eff)
        target.x = random.randint(0, max_x)
        target.y = random.randint(0, max_y)

    state.repair_to_valid()
    return state

def select_parent_roulette(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    if total_fitness <= 0:
        return random.choice(population)

    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitness_scores):
        current += fitness
        if current >= pick:
            return individual
    return population[-1]

def genetic_algorithm(
    ship,
    vehicles_data,
    population_size=20,
    max_generations=50,
    mutation_rate=0.15,
    initial_population=None
):
    start_time = time.time()

    if initial_population is not None and len(initial_population) >= population_size:
        population = [s.copy() for s in initial_population[:population_size]]
    else:
        population = [initial_random_state(ship, vehicles_data) for _ in range(population_size)]

    fitness_scores = [ind.calculate_objective() for ind in population]
    total_evaluations = len(population)

    best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
    best_overall_state = population[best_idx].copy()
    best_overall_fitness = fitness_scores[best_idx]
    first_initial_state = best_overall_state.copy()

    history_best_objective = [best_overall_fitness]
    history_avg_objective = [sum(fitness_scores) / len(fitness_scores)]

    for gen in range(1, max_generations + 1):
        new_population = []

        new_population.append(best_overall_state.copy())

        while len(new_population) < population_size:
            parent1 = select_parent_roulette(population, fitness_scores)
            parent2 = select_parent_roulette(population, fitness_scores)

            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate=mutation_rate)
            new_population.append(child)

        population = new_population
        fitness_scores = [ind.calculate_objective() for ind in population]
        total_evaluations += (len(population) - 1)

        gen_best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_avg_fitness = sum(fitness_scores) / len(fitness_scores)

        if gen_best_fitness > best_overall_fitness:
            best_overall_fitness = gen_best_fitness
            best_overall_state = population[gen_best_idx].copy()

        history_best_objective.append(gen_best_fitness)
        history_avg_objective.append(gen_avg_fitness)

    execution_time = time.time() - start_time

    return {
        "initial_state": first_initial_state,
        "final_state": best_overall_state,
        "best_objective": best_overall_fitness,
        "history_objective": history_best_objective,
        "history_best_objective": history_best_objective,
        "history_avg_objective": history_avg_objective,
        "total_generations": max_generations,
        "total_iterations": max_generations,
        "total_evaluations": total_evaluations,
        "execution_time": execution_time,
        "valid": best_overall_state.is_valid()
    }