from flask import Flask, render_template, jsonify, request
from model import Ship, Vehicle
from tetangga import initial_random_state
from hill_climbing import sideways_move_hill_climbing
from simulated_annealing import simulated_annealing

app = Flask(__name__)

kapal = Ship(6, 6, 100)
kendaraan_data = [
    {"v_id": "V1", "w": 2, "l": 2, "fee": 5000000, "weight": 20, "eta": 1},
    {"v_id": "V2", "w": 2, "l": 3, "fee": 7500000, "weight": 30, "eta": 1},
    {"v_id": "V3", "w": 1, "l": 4, "fee": 4000000, "weight": 15, "eta": 2},
    {"v_id": "V4", "w": 2, "l": 2, "fee": 6000000, "weight": 25, "eta": 3},
    {"v_id": "V5", "w": 3, "l": 2, "fee": 8000000, "weight": 35, "eta": 1},
    {"v_id": "V6", "w": 1, "l": 2, "fee": 3000000, "weight": 10, "eta": 2}
]

current_state = None
current_iter = 0


def serialize_state(state):
    """Mengubah objek State menjadi dictionary terstruktur untuk JSON response"""
    if state is None:
        return None

    grid_matrix, overlap_count, oob_count = state.get_deck_grid()
    info = state.get_info()
    score = state.calculate_objective()

    vehicles_info = [v.to_dict() for v in state.vehicles]

    return {
        "score": score,
        "grid": grid_matrix,
        "valid": info["valid"],
        "overlap": int(overlap_count),
        "oob": int(oob_count),
        "total_weight": info["total_weight"],
        "total_fee": info["total_fee"],
        "vehicles": vehicles_info
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/config", methods=["GET"])
def get_config():
    """Mengambil konfigurasi spesifikasi kapal & daftar kendaraan"""
    global current_state
    if current_state is None:
        current_state = initial_random_state(kapal, kendaraan_data)
        
    return jsonify({
        "ship": {
            "w": kapal.w,
            "l": kapal.l,
            "max_capacity": kapal.max_capacity
        },
        "vehicles_data": kendaraan_data,
        "initial_state": serialize_state(current_state)
    })


@app.route("/api/reset", methods=["POST"])
@app.route("/reset", methods=["POST"])
def reset():
    """Mereset posisi awal kendaraan secara acak"""
    global current_state, current_iter
    current_state = initial_random_state(kapal, kendaraan_data)
    current_iter = 0
    return jsonify({
        "message": "Reset berhasil",
        "initial_state": serialize_state(current_state),
        "ship": {
            "w": kapal.w,
            "l": kapal.l,
            "max_capacity": kapal.max_capacity
        }
    })


@app.route("/run-search", methods=["POST"])
@app.route("/api/run-search", methods=["POST"])
def run_search():
    """Menjalankan algoritma Orang 2: Hill Climbing Sideways Move atau Simulated Annealing"""
    global current_state
    payload = request.get_json() or {}
    
    algo = payload.get("algorithm", "sideways")
    
    if current_state is None:
        current_state = initial_random_state(kapal, kendaraan_data)
        
    init_state_copy = current_state.copy()
    
    if algo == "sideways":
        max_iterations = int(payload.get("max_iterations", 1000))
        max_sideways_moves = int(payload.get("max_sideways_moves", 100))
        result = sideways_move_hill_climbing(
            init_state_copy,
            max_sideways_moves=max_sideways_moves,
            max_iterations=max_iterations
        )
    elif algo in ["simulated_annealing", "sa"]:
        initial_temperature = float(payload.get("initial_temperature", 1000.0))
        cooling_rate = float(payload.get("cooling_rate", 0.95))
        min_temperature = float(payload.get("min_temperature", 0.01))
        max_iterations = int(payload.get("max_iterations", 1500))
        result = simulated_annealing(
            init_state_copy,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            min_temperature=min_temperature,
            max_iterations=max_iterations
        )
    else:
        return jsonify({"error": f"Algoritma tidak dikenali: {algo}"}), 400
        
    current_state = result["final_state"]
    
    response_data = {
        "algorithm": algo,
        "best_objective": result["best_objective"],
        "history_objective": result["history_objective"],
        "total_iterations": result["total_iterations"],
        "total_evaluations": result.get("total_evaluations", 0),
        "execution_time": round(result["execution_time"], 4),
        "initial_state": serialize_state(result["initial_state"]),
        "final_state": serialize_state(result["final_state"]),
        "ship": {
            "w": kapal.w,
            "l": kapal.l,
            "max_capacity": kapal.max_capacity
        }
    }

    # Parameter output khusus masing-masing algoritma
    if "sideways_moves_count" in result:
        response_data["sideways_moves_count"] = result["sideways_moves_count"]
    if "history_acceptance_prob" in result:
        response_data["history_acceptance_prob"] = result["history_acceptance_prob"]
    if "stuck_count" in result:
        response_data["stuck_count"] = result["stuck_count"]
    if "final_temperature" in result:
        response_data["final_temperature"] = result["final_temperature"]
    
    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)
