import copy

class Vehicle:
    def __init__(self, v_id=None, w=0, l=0, fee=0, weight=0, eta=1, vehicle_id=None, shipping_fee=None):
        self.id = vehicle_id if vehicle_id is not None else v_id
        self.w = int(w)
        self.l = int(l)
        self.shipping_fee = int(shipping_fee if shipping_fee is not None else fee)
        self.weight = int(weight)
        self.eta = int(eta) if eta is not None else 1
        self.is_loaded = False
        self.x = 0
        self.y = 0
        self.orientation = "Horizontal"

    def get_dimensions(self):
        if self.orientation == "Horizontal":
            return self.w, self.l
        return self.l, self.w

    def copy(self):
        v = Vehicle(
            v_id=self.id,
            w=self.w,
            l=self.l,
            fee=self.shipping_fee,
            weight=self.weight,
            eta=self.eta
        )
        v.is_loaded = self.is_loaded
        v.x = self.x
        v.y = self.y
        v.orientation = self.orientation
        return v

    def to_dict(self):
        w_eff, l_eff = self.get_dimensions()
        return {
            "id": self.id,
            "w": self.w,
            "l": self.l,
            "w_eff": w_eff,
            "l_eff": l_eff,
            "shipping_fee": self.shipping_fee,
            "weight": self.weight,
            "eta": int(self.eta),
            "is_loaded": self.is_loaded,
            "x": self.x,
            "y": self.y,
            "orientation": self.orientation
        }

class Ship:
    def __init__(self, w, l, MaxCapacity=100):
        self.w = int(w)
        self.l = int(l)
        self.MaxCapacity = int(MaxCapacity)
        self.max_capacity = int(MaxCapacity)

class State:
    def __init__(self, ship, vehicles, objective_type="fee"):
        self.ship = ship
        self.vehicles = vehicles
        self.objective_type = objective_type

    def copy(self):
        return State(self.ship, [v.copy() for v in self.vehicles], objective_type=self.objective_type)

    def get_total_weight(self):
        return sum(v.weight for v in self.vehicles if v.is_loaded)

    def get_total_fee(self):
        return sum(v.shipping_fee for v in self.vehicles if v.is_loaded)

    def is_valid(self):
        if self.get_total_weight() > self.ship.MaxCapacity:
            return False

        grid = [[0 for _ in range(self.ship.w)] for _ in range(self.ship.l)]

        for v in self.vehicles:
            if not v.is_loaded:
                continue

            w_eff, l_eff = v.get_dimensions()
            if v.x < 0 or v.y < 0 or (v.x + w_eff) > self.ship.w or (v.y + l_eff) > self.ship.l:
                return False

            for dy in range(l_eff):
                for dx in range(w_eff):
                    py = v.y + dy
                    px = v.x + dx
                    if grid[py][px] != 0:
                        return False
                    grid[py][px] = 1
        return True

    def repair_to_valid(self):
        current_weight = 0
        grid = [[None for _ in range(self.ship.w)] for _ in range(self.ship.l)]

        for v in self.vehicles:
            if not v.is_loaded:
                continue

            w_eff, l_eff = v.get_dimensions()

            if v.x < 0 or v.y < 0 or (v.x + w_eff) > self.ship.w or (v.y + l_eff) > self.ship.l:
                v.is_loaded = False
                continue

            if current_weight + v.weight > self.ship.MaxCapacity:
                v.is_loaded = False
                continue

            overlap = False
            for dy in range(l_eff):
                for dx in range(w_eff):
                    if grid[v.y + dy][v.x + dx] is not None:
                        overlap = True
                        break
                if overlap:
                    break

            if overlap:
                v.is_loaded = False
                continue

            for dy in range(l_eff):
                for dx in range(w_eff):
                    grid[v.y + dy][v.x + dx] = v.id
            current_weight += v.weight
        return self

    def calculate_objective(self, obj_type=None):
        if not self.is_valid():
            return -1000000

        mode = obj_type or getattr(self, "objective_type", "fee")

        if mode == "eta_priority":
            score = 0
            for v in self.vehicles:
                if v.is_loaded:
                    eta_val = max(1, int(getattr(v, "eta", 1) or 1))
                    score += (v.shipping_fee / eta_val)
            return round(score)

        elif mode == "space_efficiency":
            total_fee = self.get_total_fee()
            deck_area = self.ship.w * self.ship.l
            loaded_area = sum(v.w * v.l for v in self.vehicles if v.is_loaded)
            loaded_weight = self.get_total_weight()

            space_utilization = (loaded_area / deck_area) if deck_area > 0 else 0
            weight_utilization = (loaded_weight / self.ship.MaxCapacity) if self.ship.MaxCapacity > 0 else 0
            bonus = (space_utilization * 5000) + (weight_utilization * 5000)
            return round(total_fee + bonus)

        return self.get_total_fee()

    def get_deck_grid(self):
        grid_matrix = [[{"count": 0, "vehicles": []} for _ in range(self.ship.w)] for _ in range(self.ship.l)]
        overlap_count = 0
        oob_count = 0

        for v in self.vehicles:
            if not v.is_loaded:
                continue

            w_eff, l_eff = v.get_dimensions()
            for dy in range(l_eff):
                for dx in range(w_eff):
                    px = v.x + dx
                    py = v.y + dy
                    if 0 <= px < self.ship.w and 0 <= py < self.ship.l:
                        grid_matrix[py][px]["count"] += 1
                        grid_matrix[py][px]["vehicles"].append(v.id)
                    else:
                        oob_count += 1

        for r in range(self.ship.l):
            for c in range(self.ship.w):
                cnt = grid_matrix[r][c]["count"]
                if cnt > 1:
                    overlap_count += (cnt - 1)

        return grid_matrix, overlap_count, oob_count

    def get_info(self):
        grid_matrix, overlap_count, oob_count = self.get_deck_grid()
        tot_weight = self.get_total_weight()
        valid = (overlap_count == 0) and (oob_count == 0) and (tot_weight <= self.ship.MaxCapacity)
        return {
            "valid": valid,
            "total_fee": self.get_total_fee(),
            "total_weight": tot_weight,
            "MaxCapacity": self.ship.MaxCapacity,
            "overlap": overlap_count,
            "oob": oob_count,
            "ship_w": self.ship.w,
            "ship_l": self.ship.l,
            "objective_type": getattr(self, "objective_type", "fee"),
            "vehicles": [v.to_dict() for v in self.vehicles]
        }
