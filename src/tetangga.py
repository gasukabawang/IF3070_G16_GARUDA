import random
from model import State, Vehicle

def initial_random_state(ship, vehicles_data, objective_type="fee"):
    vehicles = [Vehicle(**data) for data in vehicles_data]
    indices = list(range(len(vehicles)))
    random.shuffle(indices)

    for i in indices:
        v = vehicles[i]
        if random.random() < 0.7:
            v.orientation = random.choice(["Horizontal", "Vertical"])
            w_eff, l_eff = v.get_dimensions()
            max_x = max(0, ship.w - w_eff)
            max_y = max(0, ship.l - l_eff)
            v.x = random.randint(0, max_x)
            v.y = random.randint(0, max_y)
            v.is_loaded = True
        else:
            v.is_loaded = False

    state = State(ship, vehicles, objective_type=objective_type)
    state.repair_to_valid()
    return state

def generate_all_neighbors(state, return_clash=False):
    valid_neighbors = []
    trial_evaluations = 0
    sample_clash_state = None

    for i, v in enumerate(state.vehicles):
        if v.is_loaded:
            trial_evaluations += 1
            s_rot = state.copy()
            target_rot = s_rot.vehicles[i]
            target_rot.orientation = "Vertical" if target_rot.orientation == "Horizontal" else "Horizontal"
            if s_rot.is_valid():
                valid_neighbors.append(s_rot)
            elif sample_clash_state is None:
                sample_clash_state = s_rot

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                trial_evaluations += 1
                s_move = state.copy()
                target_m = s_move.vehicles[i]
                target_m.x += dx
                target_m.y += dy
                if s_move.is_valid():
                    valid_neighbors.append(s_move)
                elif sample_clash_state is None:
                    sample_clash_state = s_move

            trial_evaluations += 1
            s_unload = state.copy()
            s_unload.vehicles[i].is_loaded = False
            if s_unload.is_valid():
                valid_neighbors.append(s_unload)

        else:
            sample_positions = [
                (0, 0),
                (state.ship.w // 2, 0),
                (0, state.ship.l // 2),
                (state.ship.w // 2, state.ship.l // 2),
                (max(0, state.ship.w - v.w), max(0, state.ship.l - v.l))
            ]
            for orient in ["Horizontal", "Vertical"]:
                for px, py in sample_positions:
                    trial_evaluations += 1
                    s_load = state.copy()
                    t_load = s_load.vehicles[i]
                    t_load.is_loaded = True
                    t_load.orientation = orient
                    t_load.x = px
                    t_load.y = py
                    if s_load.is_valid():
                        valid_neighbors.append(s_load)
                    elif sample_clash_state is None:
                        sample_clash_state = s_load

    loaded_indices = [idx for idx, veh in enumerate(state.vehicles) if veh.is_loaded]
    for idx_a in range(len(loaded_indices)):
        for idx_b in range(idx_a + 1, len(loaded_indices)):
            trial_evaluations += 1
            s_swap = state.copy()
            va = s_swap.vehicles[loaded_indices[idx_a]]
            vb = s_swap.vehicles[loaded_indices[idx_b]]
            va.x, vb.x = vb.x, va.x
            va.y, vb.y = vb.y, va.y
            if s_swap.is_valid():
                valid_neighbors.append(s_swap)
            elif sample_clash_state is None:
                sample_clash_state = s_swap

    if return_clash:
        return valid_neighbors, trial_evaluations, sample_clash_state
    return valid_neighbors, trial_evaluations

def generate_random_valid_neighbor(state, max_trials=100, return_clash=False):
    trials = 0
    sample_clash = None
    if not state.vehicles:
        return (None, 0, None) if return_clash else (None, 0)

    while trials < max_trials:
        trials += 1
        new_state = state.copy()
        idx = random.randint(0, len(new_state.vehicles) - 1)
        target = new_state.vehicles[idx]

        if target.is_loaded:
            action = random.choice(["move", "rotate", "unload"])
            if action == "move":
                target.x += random.choice([-1, 1, 0])
                target.y += random.choice([-1, 1, 0])
            elif action == "rotate":
                target.orientation = "Vertical" if target.orientation == "Horizontal" else "Horizontal"
            elif action == "unload":
                target.is_loaded = False
        else:
            target.is_loaded = True
            target.orientation = random.choice(["Horizontal", "Vertical"])
            w_eff, l_eff = target.get_dimensions()
            max_x = max(0, new_state.ship.w - w_eff)
            max_y = max(0, new_state.ship.l - l_eff)
            target.x = random.randint(0, max_x)
            target.y = random.randint(0, max_y)

        if new_state.is_valid():
            return (new_state, trials, sample_clash) if return_clash else (new_state, trials)
        elif sample_clash is None:
            sample_clash = new_state

    valid_neighbors, all_trials, det_clash = generate_all_neighbors(state, return_clash=True)
    trials += all_trials
    chosen_clash = sample_clash or det_clash
    if valid_neighbors:
        chosen = random.choice(valid_neighbors)
        return (chosen, trials, chosen_clash) if return_clash else (chosen, trials)

    return (None, trials, chosen_clash) if return_clash else (None, trials)
