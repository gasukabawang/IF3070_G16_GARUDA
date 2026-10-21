from flask import Flask, render_template, jsonify, request
from model import Ship, Vehicle
from tetangga import initial_random_state, generate_random_valid_neighbor, generate_all_neighbors
from hill_climbing import (stochastic_hill_climbing, random_restart_hill_climbing)
import copy

app = Flask(__name__)

kapal = Ship(6, 6, 100)
kendaraan_data = [{"v_id": "V1", "w": 2, "l": 2, "fee": 5000000, "weight": 20, "eta": 1},
                  {"v_id": "V2", "w": 2, "l": 3, "fee": 7500000, "weight": 30, "eta": 1},
                  {"v_id": "V3", "w": 1, "l": 4, "fee": 4000000, "weight": 15, "eta": 2},
                  {"v_id": "V4", "w": 2, "l": 2, "fee": 6000000, "weight": 25, "eta": 3},
                  {"v_id": "V5", "w": 3, "l": 2, "fee": 8000000, "weight": 35, "eta": 1},
                  {"v_id": "V6", "w": 1, "l": 2, "fee": 3000000, "weight": 10, "eta": 2}]

current_state = None
current_iter = 0

def serialize_state(state):
    if state is None:
        return None
        
    overlap, oob, grid = state.get_grid_and_violations()
    info = state.get_info()
    score = state.calculate_objective()
    
    vehicles_info = []
    for v in state.vehicles:
        w_eff, l_eff = v.get_dimensions()
        vehicles_info.append({"id": v.id,
                              "w": v.w,
                              "l": v.l,
                              "w_eff": w_eff,
                              "l_eff": l_eff,
                              "shipping_fee": v.shipping_fee,
                              "weight": v.weight,
                              "eta": v.eta,
                              "is_loaded": v.is_loaded,
                              "x": v.x,
                              "y": v.y,
                              "orientation": v.orientation})
        
    return {"score": score,
            "grid": grid.tolist(),
            "valid": info["valid"],
            "overlap": int(overlap),
            "oob": int(oob),
            "total_weight": info["berat_total"],
            "total_fee": info["total_fee"],
            "vehicles": vehicles_info}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    global current_state
    if current_state is None:
        current_state = initial_random_state(kapal, kendaraan_data)
        
    return jsonify({"ship": {"w": kapal.w,
                             "l": kapal.l,
                             "max_capacity": kapal.max_capacity},
                    "vehicles_data": kendaraan_data,
                    "initial_state": serialize_state(current_state)})

@app.route("/api/reset", methods=["POST"])
@app.route("/reset", methods=["POST"])
def reset():
    global current_state, current_iter
    current_state = initial_random_state(kapal, kendaraan_data)
    current_iter = 0
    return jsonify({"message": "Reset berhasil",
                    "initial_state": serialize_state(current_state),
                    "ship": {"w": kapal.w,
                             "l": kapal.l,
                             "max_capacity": kapal.max_capacity}})

@app.route("/run-search", methods=["POST"])
@app.route("/api/run-search", methods=["POST"])
def run_search():
    global current_state
    payload = request.get_json() or {}
    
    algo = payload.get("algorithm", "stochastic")
    max_iterations = int(payload.get("max_iterations", 500))
    max_restarts = int(payload.get("max_restarts", 5))
    
    if current_state is None:
        current_state = initial_random_state(kapal, kendaraan_data)
        
    init_state_copy = current_state.copy()
    
    if algo == "stochastic":
        result = stochastic_hill_climbing(init_state_copy, max_iterations=max_iterations)
    elif algo == "random_restart":
        result = random_restart_hill_climbing(initial_state=init_state_copy,
                                              max_restarts=max_restarts,
                                              max_iterations_per_restart=max_iterations)
    else:
        return jsonify({"error": f"Algoritma tidak dikenali: {algo}"}), 400
        
    current_state = result["final_state"]
    
    return jsonify({"algorithm": algo,
                    "best_objective": result["best_objective"],
                    "history_objective": result["history_objective"],
                    "total_iterations": result["total_iterations"],
                    "execution_time": round(result["execution_time"], 4),
                    "initial_state": serialize_state(result["initial_state"]),
                    "final_state": serialize_state(result["final_state"]),
                    "ship": {"w": kapal.w,
                             "l": kapal.l,
                             "max_capacity": kapal.max_capacity}})

if __name__ == "__main__":
    app.run(debug=True)