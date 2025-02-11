import json
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class VacancyPredictor:
    def __init__(self, json_path="outputs.vfinder/training_results.json"):
        self.json_path = json_path
        self.model = None
        self._load_data()
        self._train_model()

    def _load_data(self):
        """Carga los datos desde el archivo JSON y los convierte en un DataFrame."""
        with open(self.json_path, "r") as file:
            data = json.load(file)
        self.df = pd.DataFrame(data)

    def _train_model(self):
        """
        Prepara los datos, entrena el modelo de regresión lineal y muestra el error
        cuadrático medio en el conjunto de prueba.
        """
        # Seleccionar las variables predictoras y la variable objetivo
        X = self.df[["sm_mesh_training", "filled_volume", "vecinos"]]
        y = self.df["vacancias"]

        # Dividir en conjuntos de entrenamiento y prueba (test_size=1 para usar todos los datos en test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)

        # Crear y entrenar el modelo
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        # Evaluar el modelo con el Error Cuadrático Medio (MSE)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Error Cuadrático Medio (MSE):", mse)

    def _round_away_from_zero(self, x):
        """
        Redondea el número x "alejándose" de cero:
          - Si x es positivo, se utiliza math.ceil.
          - Si x es negativo, se utiliza math.floor.
          - Si x es 0, se devuelve 0.
        """
        if x > 0:
            return math.ceil(x)
        elif x < 0:
            return math.floor(x)
        else:
            return 0

    def predict_vacancies(self, sm_mesh_training, filled_volume, vecinos):
        """
        Dado tres parámetros (sm_mesh_training, filled_volume y vecinos),
        devuelve el número de vacancias predicho redondeado al entero "más alejado" de cero.
        """
        # Crear un DataFrame para los nuevos datos
        nuevos_datos = pd.DataFrame({
            "sm_mesh_training": [sm_mesh_training],
            "filled_volume": [filled_volume],
            "vecinos": [vecinos]
        })
        # Realizar la predicción (se obtiene un valor float)
        prediction = self.model.predict(nuevos_datos)[0]
        # Redondear la predicción utilizando el método _round_away_from_zero
        return self._round_away_from_zero(prediction)

if __name__ == "__main__":
    # Instanciar el predictor basado en regresión lineal
    predictor = VacancyPredictor("outputs.vfinder/training_results.json")
    
    # Cargar el archivo JSON de referencia que describe una vacancia única.
    # Se asume que el archivo tiene la siguiente estructura:
    # {
    #   "surface_area": [valor_area],
    #   "filled_volume": [valor_filled_volume],
    #   "vecinos": [valor_num_atm]
    # }
    with open("outputs.vfinder/key_single_vacancy.json", "r") as f:
        single_vac = json.load(f)
    # Extraer los valores de referencia (se asume que cada lista tiene un único elemento)
    ref_area = single_vac["surface_area"][0]
    ref_filled_volume = single_vac["filled_volume"][0]
    ref_vecinos = single_vac["vecinos"][0]

    # Cargar el archivo CSV y extraer las últimas tres columnas:
    # Se asume que el CSV tiene columnas en el siguiente orden:
    # [archivo, mejor_radio, area, filled_volume, num_atm]
    df = pd.read_csv("resultados_procesados.csv")
    ultimas_tres = df.iloc[:, -3:]
    
    # Extraer cada columna en un vector aparte
    vector_area = ultimas_tres.iloc[:, 0].values
    vector_filled_volume = ultimas_tres.iloc[:, 1].values
    vector_num_atm = ultimas_tres.iloc[:, 2].values
    
    # Inicializar un contador total
    total_count = 0
    
    # Iterar sobre los vectores utilizando zip e índice i
    for i, (area, filled_volume, num_atm) in enumerate(zip(vector_area, vector_filled_volume, vector_num_atm)):
        
        # Comparar los valores iterados con los de la vacancia única
        if (math.isclose(area, ref_area, rel_tol=0.1) and 
            math.isclose(filled_volume, ref_filled_volume, rel_tol=0.1) and 
            (num_atm == ref_vecinos)):
            # Si son similares, se predice directamente 1 vacancia
            vacancias_pred = 1
            print(f"Iteración {i}: Valores similares a vacancia única, predicción forzada a 1")
            total_count+=1
        else:
            # En caso contrario, se utiliza el modelo de machine learning
            vacancias_pred = predictor.predict_vacancies(area, filled_volume, num_atm)
            print(f"Iteración {i}: Área = {area}, Filled Volume = {filled_volume}, cluster_size = {num_atm} -> Vacancias Predichas = {vacancias_pred}")
            total_count+=vacancias_pred

    print(total_count)
