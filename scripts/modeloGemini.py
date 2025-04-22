# -*- coding: utf-8 -*-
"""
Implementación en PuLP de la Formulación Matemática Revisada
Rol 3: Desarrollador del Modelo de Optimización
"""

import pulp
import pandas as pd

# --------------------------------------------------------------------------
# 1. Cargar y Preparar Datos (Función Placeholder - Requiere output de Rol 1)
# --------------------------------------------------------------------------
def cargar_y_preparar_datos(ruta_datos="."):
    """
    Carga y prepara todos los parámetros necesarios desde los archivos CSV.
    ESTA FUNCIÓN DEBE SER COMPLETADA POR MIEMBRO 1 o en colaboración.
    Devuelve un diccionario con todos los parámetros listos para usar.
    """
    print("Cargando y preparando datos (PLACEHOLDER)...")

    # --- Datos de ejemplo ---
    # Conjuntos
    productos = ['21A', '22B', '23C']
    semanas = list(range(1, 53)) # Ajustar según el horizonte real
    T_max = semanas[-1]
    # Identificar semanas congeladas (ejemplo: las primeras 26)
    semanas_congeladas = list(range(1, 27)) # Placeholder
    # Identificar semanas de fin de trimestre (ejemplo Q1=13, Q2=26, ...)
    # ¡Cuidado con Q496 que tiene 14 semanas! Ajustar según calendario real.
    semanas_fin_trimestre = [13, 26, 39, 52] # Placeholder - Ajustar!

    # Parámetros (usar DataFrames cargados por Rol 1 y convertirlos a diccionarios)
    # Ejemplo de estructura diccionario (clave: (producto, semana) o (semana) o (producto))
    demanda_D = {(p, t): 1.0e8 for p in productos for t in semanas} # Bytes (Placeholder)
    inventario_S0 = {p: 1.5e8 for p in productos} # Bytes (Placeholder)
    yield_Y = {(p, t): 0.90 for p in productos for t in semanas} # % (Placeholder)
    densidad_Dens = {'21A': 94500, '22B': 69300, '23C': 66850} # bytes/wafer
    cap_total_CapT = {t: 12000 for t in semanas} # wafers (Placeholder)
    cap_prod_CapP = {(p, t): 4500 for p in productos for t in semanas} # wafers (Placeholder)
    sst_SST = {(p, t): 0.5e8 for p in productos for t in semanas} # Bytes (Placeholder)
    prod_min_PMin = 350 # wafers/semana
    ramp_up_max_RampUpM = 560 # wafers/semana
    # Plan fijo para semanas congeladas (debe venir de algun archivo o definicion)
    plan_fijo_x_bar = {(p, t): 100 for p in productos for t in semanas_congeladas} # wafers (Placeholder)
    inv_min_exc_q_InvMinEQ = 70e6 # Bytes
    inv_max_exc_q_InvMaxEQ = 140e6 # Bytes

    # Penalizaciones por demanda insatisfecha (¡IMPORTANTE DEFINIR!)
    # Deben ser números grandes, con M_21A > M_22B > M_23C
    penalizacion_M = {'21A': 10000, '22B': 100, '23C': 1} # Placeholder MUY IMPORTANTE AJUSTAR

    # Empaquetar todo en un diccionario
    parametros = {
        "productos": productos,
        "semanas": semanas,
        "T_max": T_max,
        "semanas_congeladas": semanas_congeladas,
        "semanas_fin_trimestre": semanas_fin_trimestre,
        "D": demanda_D,
        "S0": inventario_S0,
        "Y": yield_Y,
        "Dens": densidad_Dens,
        "CapT": cap_total_CapT,
        "CapP": cap_prod_CapP,
        "SST": sst_SST,
        "PMin_t": {t: prod_min_PMin for t in semanas}, # Convertido a dict por semana
        "RampUpM": ramp_up_max_RampUpM,
        "x_bar": plan_fijo_x_bar,
        "InvMinEQ": inv_min_exc_q_InvMinEQ,
        "InvMaxEQ": inv_max_exc_q_InvMaxEQ,
        "M": penalizacion_M
    }
    print("Datos (placeholders) listos.")
    return parametros

# --------------------------------------------------------------------------
# 2. Construir el Modelo de Optimización con PuLP
# --------------------------------------------------------------------------
def construir_modelo_pulp(params):
    """
    Construye el modelo LP en PuLP usando los parámetros proporcionados.
    """
    print("Construyendo el modelo PuLP...")

    # Extraer conjuntos y parámetros
    I = params["productos"]
    T = params["semanas"]
    T_cong = params["semanas_congeladas"]
    T_fin_Q = params["semanas_fin_trimestre"]
    D = params["D"]
    S0 = params["S0"]
    Y = params["Y"]
    Dens = params["Dens"]
    CapT = params["CapT"]
    CapP = params["CapP"]
    SST = params["SST"]
    PMin_t = params["PMin_t"]
    RampUpM = params["RampUpM"]
    x_bar = params["x_bar"]
    InvMinEQ = params["InvMinEQ"]
    InvMaxEQ = params["InvMaxEQ"]
    M = params["M"]

    # Crear el modelo
    model = pulp.LpProblem("Optimizacion_Produccion_Wafer_Revisado", pulp.LpMinimize)

    # --- Variables de Decisión ---
    x = pulp.LpVariable.dicts("Wafers_Producidos", (I, T), lowBound=0, cat='Integer')
    z = pulp.LpVariable.dicts("Aux_Multiplo_5", (I, T), lowBound=0, cat='Integer')
    s = pulp.LpVariable.dicts("Inventario_Final_Bytes", (I, T), lowBound=0, cat='Continuous')
    s_excess = pulp.LpVariable.dicts("Inventario_Exceso_Bytes", (I, T), lowBound=0, cat='Continuous')
    u = pulp.LpVariable.dicts("Demanda_Insatisfecha_Bytes", (I, T), lowBound=0, cat='Continuous')

    # --- Función Objetivo ---
    objetivo = pulp.lpSum(s_excess[i][t] for i in I for t in T) + \
               pulp.lpSum(M[i] * u[i][t] for i in I for t in T)
    model += objetivo, "Minimizar_Exceso_Inv_y_Penalizacion_Demanda"

    # --- Restricciones ---
    print("Añadiendo restricciones...")
    for t in T:
        # 4. Capacidad Total Semanal
        model += pulp.lpSum(x[i][t] for i in I) <= CapT[t], f"Capacidad_Total_Sem_{t}"

        # 6. Producción Mínima Total Semanal
        model += pulp.lpSum(x[i][t] for i in I) >= PMin_t[t], f"Produccion_Min_Sem_{t}"

        # 7. Ramp-Up Máximo
        if t > T[0]:
            model += pulp.lpSum(x[i][t] for i in I) - pulp.lpSum(x[i][t-1] for i in I) <= RampUpM, f"Ramp_Up_Sem_{t}"

        # 10. Rango de Inventario Excedente Trimestral (si la semana es fin de trimestre)
        if t in T_fin_Q:
            inventario_excedente_total_fin_trim = pulp.lpSum(s_excess[i][t] for i in I)
            model += inventario_excedente_total_fin_trim >= InvMinEQ, f"Inv_Exceso_Min_Fin_Trim_{t}"
            model += inventario_excedente_total_fin_trim <= InvMaxEQ, f"Inv_Exceso_Max_Fin_Trim_{t}"

        for i in I:
            # 1. Balance de Inventario
            produccion_bytes = x[i][t] * Dens[i] * Y[i, t]
            demanda_efectiva = D[i, t] - u[i][t] # Demanda que sí se satisface
            if t == T[0]:
                inventario_anterior = S0[i]
            else:
                inventario_anterior = s[i][t-1]
            model += s[i][t] == inventario_anterior + produccion_bytes - demanda_efectiva, f"Balance_Inv_{i}_Sem_{t}"

            # 2. Demanda Insatisfecha <= Demanda
            model += u[i][t] <= D[i, t], f"Limite_Demanda_Insatisfecha_{i}_Sem_{t}"

            # 3. Cálculo Inventario Excedente
            model += s_excess[i][t] >= s[i][t] - SST[i, t], f"Calc_Exceso_Inv_{i}_Sem_{t}"

            # 5. Capacidad por Producto Semanal
            model += x[i][t] <= CapP[i, t], f"Capacidad_Prod_{i}_Sem_{t}"

            # 9. Producción en Múltiplos de 5
            model += x[i][t] == 5 * z[i][t], f"Multiplo_5_{i}_Sem_{t}"

            # 8. Producción Congelada
            if t in T_cong:
                # Asume que x_bar tiene la entrada para esta i, t
                if (i, t) in x_bar:
                    model += x[i][t] == x_bar[i, t], f"Congelado_{i}_Sem_{t}"
                # else: # Opcional: ¿Qué hacer si no hay plan fijo definido? Error, o no restringir?
                #    print(f"Advertencia: No se encontró plan fijo para {i}, {t} en periodo congelado.")


    print("Modelo construido.")
    return model, x, s, s_excess, u

# --------------------------------------------------------------------------
# 3. Resolver y Mostrar Resultados (Función Placeholder)
# --------------------------------------------------------------------------
def resolver_y_mostrar_resultados(model, x, s, s_excess, u, params):
    """
    Resuelve el modelo y muestra/guarda los resultados principales.
    """
    print("Resolviendo el modelo...")
    # Podrías añadir opciones de solver aquí: model.solve(pulp.CPLEX_CMD(msg=1))
    model.solve()
    status = pulp.LpStatus[model.status]
    print(f"Status: {status}")

    if model.status == pulp.LpStatusOptimal or model.status == pulp.LpStatusFeasible:
        print(f"Valor Objetivo = {pulp.value(model.objective):,.2f}")

        # Extraer resultados a DataFrames (ejemplo)
        resultados = []
        for t in params["semanas"]:
            for i in params["productos"]:
                resultados.append({
                    'Semana': t,
                    'Producto': i,
                    'Wafers_Prod': x[i][t].varValue,
                    'Inventario_Fin_Bytes': s[i][t].varValue,
                    'Inv_Exceso_Bytes': s_excess[i][t].varValue,
                    'Demanda_Insat_Bytes': u[i][t].varValue,
                    'SST_Bytes': params["SST"][i,t],
                    'Demanda_Bytes': params["D"][i,t]
                })
        df_resultados = pd.DataFrame(resultados)

        print("\nResultados Principales (Primeras filas):")
        print(df_resultados.head())

        # Aquí Miembro 4 podría tomar df_resultados para formatear y exportar
        # df_resultados.to_csv("resultados_optimizacion.csv", index=False)

        print("\nValidación Rápida (Ejemplos):")
        # Suma Producción Semanal vs Min/Max Capacidad
        df_prod_sem = df_resultados.groupby('Semana')['Wafers_Prod'].sum()
        print("Producción Semanal Total (vs Min 350):")
        print(df_prod_sem.head())
        # Inventario Excedente Final de Trimestre (requiere identificar semanas correctas)
        print(f"Semanas fin trimestre consideradas: {params['semanas_fin_trimestre']}")
        df_fin_trim = df_resultados[df_resultados['Semana'].isin(params['semanas_fin_trimestre'])]
        inv_exc_fin_trim = df_fin_trim.groupby('Semana')['Inv_Exceso_Bytes'].sum()
        print("Inventario Excedente Total al Final del Trimestre (vs 70M-140M):")
        print(inv_exc_fin_trim)


    else:
        print("No se encontró una solución óptima o factible.")
        df_resultados = None

    return status, df_resultados

if __name__ == "__main__":
    parametros = cargar_y_preparar_datos() # Idealmente, recibe la ruta a los datos de Miembro 1

    if parametros:
        modelo, var_x, var_s, var_s_excess, var_u = construir_modelo_pulp(parametros)


        status_final, df_final = resolver_y_mostrar_resultados(modelo, var_x, var_s, var_s_excess, var_u, parametros)

        print(f"\nEjecución completada con status: {status_final}")