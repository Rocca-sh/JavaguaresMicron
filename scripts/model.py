
from pulp import *

def build_model(products, weeks, demand, initial_inventory, sst, max_ramp_up=560):
    # Crear el modelo
    model = LpProblem("Wafer_Production_Optimization", LpMinimize)

    # Variables de decisión
    x = LpVariable.dicts("Production", (products, weeks), lowBound=0, cat='Integer')
    s = LpVariable.dicts("Inventory", (products, weeks), lowBound=0)

    # Función objetivo (puede ajustarse con costos reales)
    model += lpSum(x[i][t] + 0.1 * s[i][t] for i in products for t in weeks)

    # Balance de inventario
    for i in products:
        for t in weeks:
            if t == weeks[0]:
                model += s[i][t] == initial_inventory[i] + x[i][t] - demand[(i, t)]
            else:
                model += s[i][t] == s[i][t-1] + x[i][t] - demand[(i, t)]

    # Inventario mínimo (SST)
    for i in products:
        for t in weeks:
            model += s[i][t] >= sst[i]

    # Ramp-up (producción no puede aumentar más de max_ramp_up)
    for i in products:
        for t_idx in range(1, len(weeks)):
            t, t_prev = weeks[t_idx], weeks[t_idx - 1]
            model += x[i][t] - x[i][t_prev] <= max_ramp_up

    # Producción total mínima semanal
    for t in weeks:
        model += lpSum(x[i][t] for i in products) >= 350

    return model, x, s
