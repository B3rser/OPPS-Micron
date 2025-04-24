import pulp as lp
import pandas as pd
import numpy as np
from itertools import cycle
import re
import math


# Cargar el archivo Excel una sola vez
excel_file = pd.ExcelFile('./Hackaton DB Final 04.21.xlsx')

# Leer Supply_Demand
df = excel_file.parse("Supply_Demand")

# Filtrar columnas con formato Qx YY
quarter_cols = [col for col in df.columns if re.match(r'^Q[1-4] \d{2}$', str(col))]

# Diccionario para Supply_Demand
data = {}

for col in quarter_cols:
    data[col] = {
        "21A": {
            "YS": df.loc[0, col],
            "SST": df.loc[1, col],
            "SSTW": df.loc[2, col],
            "ED": df.loc[3, col],
            "TP": df.loc[4, col],
            "ESST": df.loc[5, col]
        },
        "22B": {
            "YS": df.loc[6, col],
            "SST": df.loc[7, col],
            "SSTW": df.loc[8, col],
            "ED": df.loc[9, col],
            "TP": df.loc[10, col],
            "ESST": df.loc[11, col]
        },
        "23C": {
            "YS": df.loc[12, col],
            "SST": df.loc[13, col],
            "SSTW": df.loc[14, col],
            "ED": df.loc[15, col],
            "TP": df.loc[16, col],
            "ESST": df.loc[17, col]
        }
    }

# Obtener Density per Wafer
df_density = excel_file.parse("Density per Wafer")
density_wafer = {
    "21A": df_density["21A"].iloc[0],
    "22B": df_density["22B"].iloc[0],
    "23C": df_density["23C"].iloc[0]
}

# Leer Boundary Conditions
dfbound_raw = excel_file.parse("Boundary Conditions", header=None)
#dfbound_raw = dfbound_raw.iloc[:, 1:]  # quitar columna A

# Crear MultiIndex
quarters = dfbound_raw.iloc[0]
weeks = dfbound_raw.iloc[1]
multi_index = pd.MultiIndex.from_arrays([quarters, weeks], names=["Quarter", "Week"])

dfbound = dfbound_raw.iloc[2:].reset_index(drop=True)
dfbound.columns = multi_index

# Diccionario para Boundary Conditions
boundaryConditions = {}
for quarter in dfbound.columns.levels[0]:
    cols_q = dfbound.loc[:, quarter]

    boundaryConditions[quarter] = {
        "21A": {
            "Available": cols_q.iloc[3].tolist(),
            "Scheduled": cols_q.iloc[4].tolist(),
            "OverUnder": cols_q.iloc[5].tolist()
        },
        "22B": {
            "Available": cols_q.iloc[6].tolist(),
            "Scheduled": cols_q.iloc[7].tolist(),
            "OverUnder": cols_q.iloc[8].tolist()
        },
        "23C": {
            "Available": cols_q.iloc[9].tolist(),
            "Scheduled": cols_q.iloc[10].tolist(),
            "OverUnder": cols_q.iloc[11].tolist()
        }
    }

# Leer Wafer Plan
dfwafer_raw = excel_file.parse("Wafer Plan", header=None)
dfwafer_raw = dfwafer_raw.iloc[:, 1:]  # quitar columna A

quarters = dfwafer_raw.iloc[0]
weeks = dfwafer_raw.iloc[1]
#multi_index = pd.MultiIndex.from_arrays([quarters, weeks], names=["Quarter", "Week"])
print(multi_index)



quarters_unique = quarters.drop_duplicates()
#Hacer el modelo en plup
# Definir el problema
model = lp.LpProblem("Wafer_Production_Model", lp.LpMinimize)

weeks_per_quarter = {
    quarter: dfbound[quarter].columns.tolist()
    for quarter in quarters
}

# Definir variables de decisión

# Variables semanales de producción: X21Aij, X22Bij, X23Cij
X21A = lp.LpVariable.dicts("X21A", ((q, w) for q in quarters_unique for w in weeks_per_quarter[q]), lowBound=0, cat='Integer')
X22B = lp.LpVariable.dicts("X22B", ((q, w) for q in quarters_unique for w in weeks_per_quarter[q]), lowBound=0, cat='Integer')
X23C = lp.LpVariable.dicts("X23C", ((q, w) for q in quarters_unique for w in weeks_per_quarter[q]), lowBound=0, cat='Integer')

# Variables trimestrales de producción: X21Ai, X22Bi, X23Ci
X21A_q = lp.LpVariable.dicts("X21A_q", quarters, lowBound=0, cat='Integer')
X22B_q = lp.LpVariable.dicts("X22B_q", quarters, lowBound=0, cat='Integer')
X23C_q = lp.LpVariable.dicts("X23C_q", quarters, lowBound=0, cat='Integer')

# Variables de Yielded Supply (inventario) por quarter: YS21A, YS22B, YS23C
YS21A = lp.LpVariable.dicts("YS21A", quarters, lowBound=0, cat='Integer')
YS22B = lp.LpVariable.dicts("YS22B", quarters, lowBound=0, cat='Integer')
YS23C = lp.LpVariable.dicts("YS23C", quarters, lowBound=0, cat='Integer')

# Función objetivo: Minimizar el Yielded Supply total
model += lp.lpSum([YS21A[q] + YS22B[q] + YS23C[q] for q in quarters_unique]), "Minimize_Total_YS"
#print(model)
# -------------
# Restricciones 
# -------------

# -------------
#Minimo de produccion x week 
# -------------

names_set = set()

for q in quarters_unique:
    for w in weeks_per_quarter[q]:
        constraint_name = f"Min_Production_Week_{q}_{w}"

        if constraint_name not in names_set:
            model += (
                X21A[(q, w)] + X22B[(q, w)] + X23C[(q, w)] >= 350,
                constraint_name
            )
            names_set.add(constraint_name)
# -------------
#Maximo de produccion x week
# -------------

# Set para verificar nombres únicos
constraint_names = set()

for quarter in quarters_unique:
    weeks = weeks_per_quarter[quarter]
    week_names = dfbound[quarter].columns.tolist()
    week_index_map = {week: idx for idx, week in enumerate(week_names)}
    
    for week in weeks:
        idx = week_index_map[week]
        available_21A = boundaryConditions[quarter]["21A"]["Available"][idx]
        available_22B = boundaryConditions[quarter]["22B"]["Available"][idx]
        available_23C = boundaryConditions[quarter]["23C"]["Available"][idx]

        # Función para generar un nombre único
        def unique_name(base):
            if base not in constraint_names:
                constraint_names.add(base)
                return base
            i = 1
            while f"{base}_{i}" in constraint_names:
                i += 1
            new_name = f"{base}_{i}"
            constraint_names.add(new_name)
            return new_name

        # Agregar restricciones con nombres únicos
        model += X21A[(quarter, week)] <= available_21A, unique_name(f"MaxProd_21A_{quarter}_{week}")
        model += X22B[(quarter, week)] <= available_22B, unique_name(f"MaxProd_22B_{quarter}_{week}")
        model += X23C[(quarter, week)] <= available_23C, unique_name(f"MaxProd_23C_{quarter}_{week}")

# -------------
#Inventario minimo y maximo en bytes por wafer por quarter
# -------------

# Para evitar duplicados en nombres de restricciones
inv_names_set = set()

def safe_number(x):
    try:
        return 0 if (x is None or isinstance(x, str) or math.isnan(float(x))) else float(x)
    except:
        return 0

# Función para obtener el quarter anterior
def get_previous_quarter(current_quarter):
    quarter_number = int(current_quarter[1])
    year_suffix = int(current_quarter[3:])
    if quarter_number == 1:
        return f"Q4 {year_suffix - 1:02d}"
    else:
        return f"Q{quarter_number - 1} {year_suffix:02d}"

# Restricciones de inventario mínimo y máximo con nombres únicos
for quarter in quarters_unique:
    # Producto 21A
    TP_prev_21A = safe_number(data.get(get_previous_quarter(quarter), {}).get("21A", {}).get("TP", 0))
    ED_21A = safe_number(data[quarter]["21A"].get("ED", 0))
    SST_21A = safe_number(data[quarter]["21A"].get("SST", 0))
    model += (YS21A[quarter] * 94500 + TP_prev_21A - ED_21A - SST_21A >= 70_000_000), f"InvMin_21A_{quarter}"
    model += (YS21A[quarter] * 94500 + TP_prev_21A - ED_21A - SST_21A <= 140_000_000), f"InvMax_21A_{quarter}"

    # Producto 22B
    TP_prev_22B = safe_number(data.get(get_previous_quarter(quarter), {}).get("22B", {}).get("TP", 0))
    ED_22B = safe_number(data[quarter]["22B"].get("ED", 0))
    SST_22B = safe_number(data[quarter]["22B"].get("SST", 0))
    model += (YS22B[quarter] * 69300 + TP_prev_22B - ED_22B - SST_22B >= 70_000_000), f"InvMin_22B_{quarter}"
    model += (YS22B[quarter] * 69300 + TP_prev_22B - ED_22B - SST_22B <= 140_000_000), f"InvMax_22B_{quarter}"

    # Producto 23C
    TP_prev_23C = safe_number(data.get(get_previous_quarter(quarter), {}).get("23C", {}).get("TP", 0))
    ED_23C = safe_number(data[quarter]["23C"].get("ED", 0))
    SST_23C = safe_number(data[quarter]["23C"].get("SST", 0))
    model += (YS23C[quarter] * 66850 + TP_prev_23C - ED_23C - SST_23C >= 70_000_000), f"InvMin_23C_{quarter}"
    model += (YS23C[quarter] * 66850 + TP_prev_23C - ED_23C - SST_23C <= 140_000_000), f"InvMax_23C_{quarter}"
# -------------
#Ramp Up (no mayor a 500)
# -------------
# Conjunto para verificar nombres duplicados
names_set = set()
# Conjunto para evitar duplicación de nombres de restricciones
names_set = set()

# Función para extraer el número de semana como entero
def get_week_number(week_str):
    match = re.search(r'\d+', week_str)
    return int(match.group()) if match else None

for q in quarters_unique:
    # Ordenar las semanas numéricamente
    weeks = sorted(weeks_per_quarter[q], key=lambda w: get_week_number(w))
    for i in range(1, len(weeks)):  # Empezamos desde la segunda semana
        current_week = weeks[i]
        prev_week = weeks[i - 1]

        # Nombres únicos
        ramp_up_name_21A = f"RampUp_X21A_{q}_{current_week}"
        ramp_up_name_22B = f"RampUp_X22B_{q}_{current_week}"
        ramp_up_name_23C = f"RampUp_X23C_{q}_{current_week}"

        # Solo agregar si el nombre no ha sido usado
        if ramp_up_name_21A not in names_set:
            names_set.add(ramp_up_name_21A)
            model += (X21A[(q, current_week)] - X21A[(q, prev_week)] <= 560), ramp_up_name_21A

        if ramp_up_name_22B not in names_set:
            names_set.add(ramp_up_name_22B)
            model += (X22B[(q, current_week)] - X22B[(q, prev_week)] <= 560), ramp_up_name_22B

        if ramp_up_name_23C not in names_set:
            names_set.add(ramp_up_name_23C)
            model += (X23C[(q, current_week)] - X23C[(q, prev_week)] <= 560), ramp_up_name_23C

# -------------
# Todas la producciones deben de ser modulos de 5
# -------------
for q in quarters_unique:
    for w in weeks_per_quarter[q]:
        # Variables auxiliares para el múltiplo
        k21A = lp.LpVariable(f"K21A_{q}_{w}", cat="Integer")
        k22B = lp.LpVariable(f"K22B_{q}_{w}", cat="Integer")
        k23C = lp.LpVariable(f"K23C_{q}_{w}", cat="Integer")

        # Restricciones para forzar múltiplos de 5
        model += X21A[(q, w)] == 5 * k21A, f"Modulo5_X21A_{q}_{w}"
        model += X22B[(q, w)] == 5 * k22B, f"Modulo5_X22B_{q}_{w}"
        model += X23C[(q, w)] == 5 * k23C, f"Modulo5_X23C_{q}_{w}"

# -------------

# -------------


# -------------
#Ejecucion del modelo
# -------------
# Resolver el modelo
model.solve()

# Verificar el estado de la solución
print("Estado del modelo:", lp.LpStatus[model.status])

# Imprimir el valor de la función objetivo
print("Valor óptimo (Total Yielded Supply):", lp.value(model.objective))

# Si quieres imprimir los valores de todas las variables:
#for v in model.variables():
#    if v.varValue is not None and v.varValue != 0:
#        print(v.name, "=", v.varValue)
# -------------
#Almacenar los resultados en excel 
# -------------

# Limpiar strings
dfbound_raw.iloc[:, 0] = dfbound_raw.iloc[:, 0].astype(str).str.strip()
dfbound_raw.iloc[:, 1] = dfbound_raw.iloc[:, 1].astype(str).str.strip()

# Buscar filas de Scheduled Capacity para cada producto
def get_index(product_id, attribute):
    match = dfbound_raw[
        (dfbound_raw.iloc[:, 0] == product_id) & 
        (dfbound_raw.iloc[:, 1] == attribute)
    ]
    if not match.empty:
        return match.index[0]
    else:
        raise ValueError(f"No se encontró la fila para {product_id} - {attribute}")

index_21A = get_index('21A', 'Scheduled Capacity')
index_22B = get_index('22B', 'Scheduled Capacity')
index_23C = get_index('23C', 'Scheduled Capacity')

# Mapear a columnas por Quarter y Semana

column_map = {
    (quarter, week): col_idx
    for col_idx, (quarter, week) in enumerate(multi_index, start=0)  # empieza desde la col C (índice 2)
}
#print(column_map)
# Función auxiliar para asignar valores de variables a su fila correspondiente
def actualizar_scheduled(index_fila, variable_dict, nombre_var):
    if index_fila is None:
        print(f"❌ Se omite la escritura de {nombre_var} por falta de fila.")
        return

    total_escritos = 0
    total_fallidos = 0

    for (quarter, week), var in variable_dict.items():
        key = (str(quarter).strip(), str(week).strip())
        col_idx = column_map.get(key)

        if col_idx is None:
            print(f"⚠️ No se encontró la columna para ({quarter}, {week}) en {nombre_var} — Clave ausente en column_map")
            total_fallidos += 1
            continue

        if var.varValue is None:
            print(f"🕳️ La variable {nombre_var}[{quarter}, {week}] no tiene valor asignado (varValue = None)")
            total_fallidos += 1
            continue

        try:
            dfbound_raw.iat[index_fila, col_idx] = round(var.varValue, 2)
            total_escritos += 1
        except Exception as e:
            print(f"❌ Error al escribir en celda fila {index_fila}, columna {col_idx} para {nombre_var}[{quarter}, {week}]: {e}")
            total_fallidos += 1

    print(f"✅ Finalizó {nombre_var}: {total_escritos} valores escritos, {total_fallidos} con problemas.")

# Calcular Over/Under y Totales
index_total_available = get_index("Total", "Available Capacity")
index_total_scheduled = get_index("Total", "Scheduled Capacity")
index_total_over = get_index("Total", "Over/Under Capacity")

index_available = {
    "21A": get_index("21A", "Available Capacity"),
    "22B": get_index("22B", "Available Capacity"),
    "23C": get_index("23C", "Available Capacity")
}
index_scheduled = {
    "21A": get_index("21A", "Scheduled Capacity"),
    "22B": get_index("22B", "Scheduled Capacity"),
    "23C": get_index("23C", "Scheduled Capacity")
}
index_over = {
    "21A": get_index("21A", "Over/Under Capacity"),
    "22B": get_index("22B", "Over/Under Capacity"),
    "23C": get_index("23C", "Over/Under Capacity")
}

# Calcular todos los valores
for (quarter, week), _ in X21A.items():
    key = (str(quarter).strip(), str(week).strip())
    col_idx = column_map.get(key)
    if col_idx is None:
        continue

    total_available = 0
    total_scheduled = 0

    for prod in ["21A", "22B", "23C"]:
        av = dfbound_raw.iat[index_available[prod], col_idx]
        sc = dfbound_raw.iat[index_scheduled[prod], col_idx]
        if pd.notna(av) and pd.notna(sc):
            dfbound_raw.iat[index_over[prod], col_idx] = round(sc - av, 2)
            total_available += av
            total_scheduled += sc

    dfbound_raw.iat[index_total_available, col_idx] = round(total_available, 2)
    dfbound_raw.iat[index_total_scheduled, col_idx] = round(total_scheduled, 2)
    dfbound_raw.iat[index_total_over, col_idx] = round(total_scheduled - total_available, 2)


# Actualizar valores
actualizar_scheduled(index_21A, X21A, "X21A")
actualizar_scheduled(index_22B, X22B, "X22B")
actualizar_scheduled(index_23C, X23C, "X23C")
# Guardar el mismo archivo sobrescribiéndolo (¡ciérralo antes de correr esto!)
with pd.ExcelWriter('./Hackaton DB Final 04.21.xlsx', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    dfbound_raw.to_excel(writer, sheet_name="Boundary Conditions", index=False, header=None)
# -------------
# -------------
# Crear una lista de diccionarios con los resultados
results = []

for v in model.variables():
    if v.varValue is not None and v.varValue != 0:
        results.append({
            "Variable": v.name,
            "Valor": v.varValue
        })

# Convertir a DataFrame
#df_variables  = pd.DataFrame(results)
# 📏 Restricciones
#constraints = []
#for name, constraint in model.constraints.items():
#    constraints.append({
#        "Nombre": name,
#        "Restricción": str(constraint)
#    })
#df_constraints = pd.DataFrame(constraints)
# Guardar en Excel
#output_path = "resultados_modelo.xlsx"
#with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
#    df_variables.to_excel(writer, sheet_name="Variables", index=False)
#    df_constraints.to_excel(writer, sheet_name="Restricciones", index=False)

#print(f" Resultados guardados en: {output_path}")

# -------------
# -------------


#print(model.objective)
#for name, constraint in model.constraints.items():
#    print(f"{name}: {constraint}")

