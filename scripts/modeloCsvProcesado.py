from pulp import *
import pandas as pd

def build_model_only_capacity(products, weeks, df_capacity):
    # Crear el modelo
    model = LpProblem("Wafer_Production_Only_Capacity", LpMinimize)

    # Variables de decisión: solo la producción es relevante aquí
    x = LpVariable.dicts("Production", (products, weeks), lowBound=0, cat='Integer')
    model += lpSum(x[i][t] for i in products for t in weeks)

    capacity_dict = {}
    for index, row in df_capacity.iterrows():
        product_id = row['Product_ID']
        week_original = row['Week_Original']
        capacity = row['Capacity_Value']
        try:
            week_index = weeks.index(week_original) + 1  # Si tus semanas son 1-based
            if product_id != 'Total':
                capacity_dict[(product_id, week_index)] = capacity
        except ValueError:
            print(f"Advertencia: No se encontró la semana '{week_original}' en la lista de semanas.")

    for i in products:
        for t in weeks:
            if (i, t) in capacity_dict:
                model += x[i][t] <= capacity_dict[(i, t)]

    return model, x

# Ejemplo de cómo podrías usar la función:
if __name__ == '__main__':
    # Datos de ejemplo (reemplaza con tus datos reales)
    products = ['21A', '22B', '23C']
    weeks_original = ['WW_09', 'WW_10.1', 'WW_11'] # Ejemplo de tus Week_Original
    weeks_model = [1, 2, 3] # Ejemplo de cómo podrías indexar las semanas para el modelo
    capacity_df_data = {
        'Product_ID': ['21A', '21A', '22B', '22B', '23C', '23C'],
        'Week_Original': ['WW_09', 'WW_10.1', 'WW_09', 'WW_10.1', 'WW_09', 'WW_10.1'],
        'Capacity_Value': [600.0, 620.0, 450.0, 480.0, 700.0, 720.0]
    }
    df_long_capacity = pd.DataFrame(capacity_df_data)

    # Crear el modelo considerando solo la capacidad
    model, production_vars = build_model_only_capacity(
        products, weeks_model, df_long_capacity
    )

    # Resolver el modelo (incluso con una función objetivo simplificada)
    model.solve()

    print("Status:", LpStatus[model.status])

    # Imprimir la producción "óptima" (limitada solo por la capacidad)
    for v in model.variables():
        if "Production" in v.name and v.varValue > 0:
            print(f"{v.name} = {v.varValue}")

    # Imprimir el valor de la función objetivo (que en este caso es solo la producción total)
    print("Total Production = ", value(model.objective))