"""
Modul Hill-Climbing untuk Penyelesaian Stacking Geladak Kapal Kargo 2D
IF3070 Dasar Inteligensi Artifisial - Kelompok G16_GARUDA

Mengimplementasikan 4 Varian Hill-Climbing:
1. Steepest Ascent Hill-Climbing
2. Stochastic Hill-Climbing
3. Hill-Climbing with Sideways Move
4. Random Restart Hill-Climbing
"""

import time
import random
from tetangga import generate_all_neighbors, initial_random_state


def steepest_ascent_hill_climbing(initial_state, max_iterations=1000, max_no_improvement=50):
    """
    Steepest Ascent Hill-Climbing.
    
    Cara Kerja:
    1. Evaluasi semua tetangga yang dibangkitkan dari current_state.
    2. Cari tetangga dengan nilai objective TERBAIK.
    3. Pindah hanya jika nilai tetangga terbaik MURNI LEBIH TINGGI (strictly better) dari current_state.
    4. Berhenti jika tidak ada tetangga yang lebih baik (local optima / plateau),
       atau mencapai max_iterations, atau mencapai max_no_improvement.
    
    Returns:
        dict: Hasil eksekusi algoritma dalam format standar.
    """
    start_time = time.time()
    
    current_state = initial_state.copy()
    current_val = current_state.calculate_objective()
    
    history_objective = [current_val]
    total_iterations = 0
    no_improvement_count = 0
    
    while total_iterations < max_iterations:
        neighbors = generate_all_neighbors(current_state)
        if not neighbors:
            break
            
        best_val = float('-inf')
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
            current_state = best_neighbor
            current_val = best_val
            total_iterations += 1
            history_objective.append(current_val)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
            if no_improvement_count >= max_no_improvement or best_val <= current_val:
                break
                
    execution_time = time.time() - start_time
    
    return {
        "initial_state": initial_state,
        "final_state": current_state,
        "best_objective": current_val,
        "history_objective": history_objective,
        "total_iterations": total_iterations,
        "execution_time": execution_time
    }


def stochastic_hill_climbing(initial_state, max_iterations=1000, max_attempts=50):
    """
    Stochastic Hill-Climbing.
    
    Cara Kerja:
    1. Filter seluruh tetangga yang memiliki nilai objective LEBIH TINGGI dari current_state.
    2. Jika himpunan tetangga yang lebih baik tidak kosong, pilih 1 tetangga SECARA ACAK.
    3. Berhenti jika tidak ditemukan tetangga yang lebih baik setelah max_attempts percobaan,
       atau himpunan tetangga lebih baik kosong, atau mencapai max_iterations.
    
    Returns:
        dict: Hasil eksekusi algoritma dalam format standar.
    """
    start_time = time.time()
    
    current_state = initial_state.copy()
    current_val = current_state.calculate_objective()
    
    history_objective = [current_val]
    total_iterations = 0
    attempts_without_improvement = 0
    
    while total_iterations < max_iterations:
        neighbors = generate_all_neighbors(current_state)
        if not neighbors:
            break
            
        better_neighbors = []
        for neighbor in neighbors:
            val = neighbor.calculate_objective()
            if val > current_val:
                better_neighbors.append((neighbor, val))
                
        if better_neighbors:
            
            chosen_neighbor, chosen_val = random.choice(better_neighbors)
            current_state = chosen_neighbor
            current_val = chosen_val
            total_iterations += 1
            history_objective.append(current_val)
            attempts_without_improvement = 0
        else:
            attempts_without_improvement += 1
            if attempts_without_improvement >= max_attempts:
                break
                
    execution_time = time.time() - start_time
    
    return {
        "initial_state": initial_state,
        "final_state": current_state,
        "best_objective": current_val,
        "history_objective": history_objective,
        "total_iterations": total_iterations,
        "execution_time": execution_time
    }


def sideways_move_hill_climbing(initial_state, max_sideways_moves=50, max_iterations=1000):
    """
    Hill-Climbing with Sideways Move.
    
    Cara Kerja:
    1. Evaluasi seluruh tetangga dan cari best_neighbor.
    2. Jika VALUE(best_neighbor) > VALUE(current_state):
       - Pindah ke best_neighbor dan reset sideways_count = 0.
    3. Jika VALUE(best_neighbor) == VALUE(current_state) DAN sideways_count < max_sideways_moves:
       - Pindah ke best_neighbor dan sideways_count += 1.
    4. Berhenti jika VALUE(best_neighbor) < VALUE(current_state) ATAU sideways_count >= max_sideways_moves
       ATAU mencapai max_iterations.
    
    Returns:
        dict: Hasil eksekusi algoritma dalam format standar.
    """
    start_time = time.time()
    
    current_state = initial_state.copy()
    current_val = current_state.calculate_objective()
    
    history_objective = [current_val]
    total_iterations = 0
    sideways_count = 0
    
    while total_iterations < max_iterations:
        neighbors = generate_all_neighbors(current_state)
        if not neighbors:
            break
            
        best_val = float('-inf')
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
            
            current_state = best_neighbor
            current_val = best_val
            sideways_count = 0
            total_iterations += 1
            history_objective.append(current_val)
        elif best_val == current_val and sideways_count < max_sideways_moves:
            
            current_state = best_neighbor
            current_val = best_val
            sideways_count += 1
            total_iterations += 1
            history_objective.append(current_val)
        else:
            
            break
            
    execution_time = time.time() - start_time
    
    return {
        "initial_state": initial_state,
        "final_state": current_state,
        "best_objective": current_val,
        "history_objective": history_objective,
        "total_iterations": total_iterations,
        "execution_time": execution_time
    }


def random_restart_hill_climbing(initial_state=None, ship=None, vehicles_data=None, max_restarts=5, max_iterations_per_restart=500):
    """
    Random Restart Hill-Climbing.
    
    Cara Kerja:
    1. Loop sebanyak max_restarts kali.
    2. Setiap restart membangkitkan State Awal acak baru.
    3. Jalankan Steepest Ascent Hill-Climbing dari state awal tersebut.
    4. Simpan dan perbarui State Terbaik (best_overall_state) dan nilai terbaiknya.
    5. Kembalikan State Terbaik beserta seluruh riwayat pencarian.
    
    Returns:
        dict: Hasil eksekusi algoritma dalam format standar.
    """
    start_time = time.time()
    
    best_overall_state = None
    best_overall_value = float('-inf')
    combined_history = []
    total_iterations_all = 0
    first_initial_state = initial_state.copy() if initial_state is not None else None
    
    
    if initial_state is not None:
        if ship is None:
            ship = initial_state.ship
        if vehicles_data is None:
            vehicles_data = [
                {
                    "v_id": v.id,
                    "w": v.w,
                    "l": v.l,
                    "fee": v.shipping_fee,
                    "weight": v.weight,
                    "eta": v.eta
                }
                for v in initial_state.vehicles
            ]
            
    if ship is None or vehicles_data is None:
        raise ValueError("Parameter initial_state atau (ship, vehicles_data) harus disediakan.")
        
    for r in range(max_restarts):
        
        if r == 0 and initial_state is not None:
            restart_init_state = initial_state.copy()
        else:
            restart_init_state = initial_random_state(ship, vehicles_data)
            
        if first_initial_state is None:
            first_initial_state = restart_init_state.copy()
            
        result = steepest_ascent_hill_climbing(
            restart_init_state,
            max_iterations=max_iterations_per_restart
        )
        
        total_iterations_all += result["total_iterations"]
        combined_history.extend(result["history_objective"])
        
        if result["best_objective"] > best_overall_value:
            best_overall_value = result["best_objective"]
            best_overall_state = result["final_state"]
            
    execution_time = time.time() - start_time
    
    return {
        "initial_state": first_initial_state,
        "final_state": best_overall_state,
        "best_objective": best_overall_value,
        "history_objective": combined_history,
        "total_iterations": total_iterations_all,
        "execution_time": execution_time
    }