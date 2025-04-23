import pulp as lp
import pandas as pd
import numpy as np
from itertools import cycle
import re

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
dfbound_raw = dfbound_raw.iloc[:, 1:]  # quitar columna A

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
multi_index = pd.MultiIndex.from_arrays([quarters, weeks], names=["Quarter", "Week"])

dfwafer = dfwafer_raw.iloc[2:].reset_index(drop=True)
dfwafer.columns = multi_index

waferPlan = {
    "21A": dfwafer.iloc[0].tolist(),
    "22B": dfwafer.iloc[1].tolist(),
    "23C": dfwafer.iloc[2].tolist()
}

#Hacer el modelo en plup
# Definir el problema
model = lp.LpProblem("Wafer_Production_Model", lp.LpMinimize)

weeks_per_quarter = {
    quarter: dfbound[quarter].columns.tolist()
    for quarter in quarters
}
# Definir variables de decisión

# Variables semanales de producción: X21Aij, X22Bij, X23Cij
X21A = lp.LpVariable.dicts("X21A", ((q, w) for q in quarters for w in weeks_per_quarter[q]), lowBound=0, cat='Continuous')
X22B = lp.LpVariable.dicts("X22B", ((q, w) for q in quarters for w in weeks_per_quarter[q]), lowBound=0, cat='Continuous')
X23C = lp.LpVariable.dicts("X23C", ((q, w) for q in quarters for w in weeks_per_quarter[q]), lowBound=0, cat='Continuous')

# Variables trimestrales de producción: X21Ai, X22Bi, X23Ci
X21A_q = lp.LpVariable.dicts("X21A_q", quarters, lowBound=0, cat='Integer')
X22B_q = lp.LpVariable.dicts("X22B_q", quarters, lowBound=0, cat='Integer')
X23C_q = lp.LpVariable.dicts("X23C_q", quarters, lowBound=0, cat='Integer')

# Variables de Yielded Supply (inventario) por quarter: YS21A, YS22B, YS23C
YS21A = lp.LpVariable.dicts("YS21A", quarters, lowBound=0, cat='Integer')
YS22B = lp.LpVariable.dicts("YS22B", quarters, lowBound=0, cat='Integer')
YS23C = lp.LpVariable.dicts("YS23C", quarters, lowBound=0, cat='Integer')

# Función objetivo: Minimizar el Yielded Supply total
model += lp.lpSum([YS21A[q] + YS22B[q] + YS23C[q] for q in quarters]), "Minimize_Total_YS"
#print(model)
#Restricciones 
#Minimo de produccion x week 
names_set = set()

for q in quarters:
    for w in weeks_per_quarter[q]:
        constraint_name = f"Min_Production_Week_{q}_{w}"
        if constraint_name in names_set:
            print("⚠️ Nombre duplicado:", constraint_name)
        names_set.add(constraint_name)

        
print(model)
