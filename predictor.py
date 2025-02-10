import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("resultados_procesados.csv")

# Extraer las últimas tres columnas utilizando indexación .iloc
ultimas_tres = df.iloc[:, -3:]

# Extraer cada columna en un vector aparte
vector_area = ultimas_tres.iloc[:, 0].values
vector_filled_volume = ultimas_tres.iloc[:, 1].values
vector_num_atm = ultimas_tres.iloc[:, 2].values

print("Vector de 'area':", vector_area)
print("Vector de 'filled_volume':", vector_filled_volume)
print("Vector de 'num_atm':", vector_num_atm)
