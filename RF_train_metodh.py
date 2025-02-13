import json
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score

class VacancyPredictorRF:
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
        Prepara los datos, entrena el modelo Random Forest y muestra el error
        cuadrático medio en el conjunto de prueba.
        """
        # Seleccionar las variables predictoras y la variable objetivo
        X = self.df[["sm_mesh_training", "filled_volume", "vecinos"]]
        y = self.df["vacancias"]

        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Crear y entrenar el modelo Random Forest
        self.model = RandomForestRegressor(random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluar el modelo con el Error Cuadrático Medio (MSE)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Error Cuadrático Medio (MSE) con Random Forest:", mse)

    def _round_up(self, x):
        """
        Redondea el número x hacia arriba utilizando math.ceil.
        """
        return math.ceil(x)

    def predict_vacancies(self, sm_mesh_training, filled_volume, vecinos):
        """
        Dado tres parámetros (sm_mesh_training, filled_volume y vecinos),
        devuelve el número de vacancias predicho redondeado hacia arriba.
        """
        # Crear un DataFrame para los nuevos datos
        nuevos_datos = pd.DataFrame({
            "sm_mesh_training": [sm_mesh_training],
            "filled_volume": [filled_volume],
            "vecinos": [vecinos]
        })
        # Realizar la predicción (se obtiene un valor float)
        prediction = self.model.predict(nuevos_datos)[0]
        # Redondear la predicción utilizando el método _round_up
        return self._round_up(prediction)

if __name__ == "__main__":
    # Crear la instancia del predictor
    predictor_rf = VacancyPredictorRF("outputs.vfinder/training_results.json")

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
            total_count += 1
            print(f"Iteración {i}: Valores similares a vacancia única, predicción forzada a 1")
        else:
            # En caso contrario, se utiliza el modelo de machine learning
            vacancias_pred = predictor_rf.predict_vacancies(area, filled_volume, num_atm)
            print(f"Iteración {i}: Área = {area}, Filled Volume = {filled_volume}, cluster_size = {num_atm} -> Vacancias Predichas = {vacancias_pred}")
            total_count += vacancias_pred
    print("\nContador total:", total_count)


from sklearn.model_selection import KFold, cross_val_score

class XGBoostVacancyPredictor:
    def __init__(self, training_data_path="outputs.vfinder/training_results.json", model_path="outputs.json/xgboost_model.json", n_splits=5, random_state=42):
        """
        Inicializa el predictor:
         - training_data_path: ruta al archivo JSON con los datos de entrenamiento.
         - model_path: ruta donde se guarda el modelo entrenado.
         - n_splits: número de pliegues para validación cruzada.
         - random_state: semilla para reproducibilidad.
        """
        self.training_data_path = training_data_path
        self.model_path = model_path
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.random_state)
        self._load_data_and_train()
    
    def _load_data_and_train(self):
        """Carga los datos, realiza validación cruzada, entrena el modelo y lo guarda."""
        # Cargar los datos de entrenamiento desde el archivo JSON
        with open(self.training_data_path, "r") as f:
            data = json.load(f)
        
        # Extraer las variables:
        # - Características (features): sm_mesh_training, filled_volume y vecinos.
        # - Objetivo (target): vacancias.
        X = np.column_stack((data["sm_mesh_training"],
                              data["filled_volume"],
                              data["vecinos"]))
        y = np.array(data["vacancias"])
        
        # Definir el esquema de validación cruzada con KFold.
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Realizar la validación cruzada calculando el error cuadrático medio (MSE).
        scores = cross_val_score(self.model, X, y, scoring='neg_mean_squared_error', cv=kfold)
        mse_scores = -scores  # Se convierten a positivos para interpretarlos
        print("Puntuaciones MSE en cada pliegue XGBoost :", mse_scores)
        print("MSE promedio XGBoost :", mse_scores.mean())
        
        # Entrenar el modelo sobre todos los datos y guardar el modelo entrenado.
        self.model.fit(X, y)
        self.model.save_model(self.model_path)
        #print("Modelo XGBoost entrenado y guardado en '{}'".format(self.model_path))
    
    def predict(self, sample_input):
        """
        Realiza una predicción dado un sample input.
        
        Args:
            sample_input (list o numpy.ndarray): Debe tener la forma (n_samples, 3)
                donde cada entrada representa [sm_mesh, filled_volume, vecinos].
                
        Returns:
            numpy.ndarray: Predicción realizada por el modelo.
        """
        sample_input = np.array(sample_input)
        prediction = self.model.predict(sample_input)
        return prediction
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

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