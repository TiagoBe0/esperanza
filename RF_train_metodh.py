import json
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score

# Importar la configuración de columnas desde inputs_param.py
from input_params import PREDICTOR_COLUMNS

class VacancyPredictorRF:
    def __init__(self, json_path="outputs.vfinder/training_results.json"):
        self.json_path = json_path
        self.model = None
        # Se usan las columnas definidas por el usuario en el archivo de parámetros
        self.columns = PREDICTOR_COLUMNS
        # Mapeo entre los nombres de columnas en el JSON de entrenamiento y los nombres en el CSV
        self.column_mapping = {
            "sm_mesh_training": "area",
            "filled_volume": "filled_volume",
            "vecinos": "num_atm"
        }
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
        # Seleccionar las variables predictoras según la configuración y la variable objetivo
        X = self.df[self.columns]
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

    def predict_vacancies(self, **kwargs):
        """
        Dado un conjunto de parámetros (en keyword arguments) que correspondan a las columnas configuradas,
        devuelve el número de vacancias predicho redondeado hacia arriba.
        Ejemplo de uso:
            predictor.predict_vacancies(sm_mesh_training=valor, vecinos=valor)
        """
        # Se arma un DataFrame con las columnas configuradas
        data = {col: [kwargs[col]] for col in self.columns}
        nuevos_datos = pd.DataFrame(data)
        prediction = self.model.predict(nuevos_datos)[0]
        return self._round_up(prediction)

    def predict_from_csv(self, csv_path):
        """
        Carga el CSV con los datos y predice las vacancias para cada registro usando las columnas configuradas.
        Retorna el DataFrame actualizado con las predicciones.
        """
        df = pd.read_csv(csv_path)
        # Se asume que el CSV tiene las columnas: area, filled_volume, num_atm en el bloque final
        ultimas_tres = df.iloc[:, -3:]
        predictions = []
        for idx, row in ultimas_tres.iterrows():
            features = {}
            # Se arma el diccionario de features usando el mapeo: 
            # las claves corresponden a los nombres del entrenamiento y los valores se extraen del CSV.
            for col in self.columns:
                csv_col = self.column_mapping.get(col, col)
                features[col] = row[csv_col]
            pred = self.predict_vacancies(**features)
            predictions.append(pred)
        df["predicted_vacancies"] = predictions
        return df
"""
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
    ref_area = single_vac["surface_area"][0]
    ref_filled_volume = single_vac["filled_volume"][0]
    ref_vecinos = single_vac["vecinos"][0]

    # Cargar el archivo CSV y extraer las últimas tres columnas:
    # Se asume que el CSV tiene columnas en el siguiente orden:
    # [archivo, mejor_radio, area, filled_volume, num_atm]
    df_csv = pd.read_csv("resultados_procesados.csv")
    ultimas_tres = df_csv.iloc[:, -3:]
    vector_area = ultimas_tres.iloc[:, 0].values
    vector_filled_volume = ultimas_tres.iloc[:, 1].values
    vector_num_atm = ultimas_tres.iloc[:, 2].values

    total_count = 0
    for i, (area, filled_volume, num_atm) in enumerate(zip(vector_area, vector_filled_volume, vector_num_atm)):
        # Comparar los valores del CSV con los de la vacancia única
        if (math.isclose(area, ref_area, rel_tol=0.1) and 
            math.isclose(filled_volume, ref_filled_volume, rel_tol=0.1) and 
            (num_atm == ref_vecinos)):
            vacancias_pred = 1
            total_count += 1
            print(f"Iteración {i}: Valores similares a vacancia única, predicción forzada a 1")
        else:
            # Se arma el diccionario de features para la predicción según las columnas configuradas
            features = {}
            for col in predictor_rf.columns:
                csv_col = predictor_rf.column_mapping.get(col, col)
                if csv_col == "area":
                    features[col] = area
                elif csv_col == "filled_volume":
                    features[col] = filled_volume
                elif csv_col == "num_atm":
                    features[col] = num_atm
            vacancias_pred = predictor_rf.predict_vacancies(**features)
            print(f"Iteración {i}: Características {features} -> Vacancias Predichas = {vacancias_pred}")
            total_count += vacancias_pred
    print("\nContador total:", total_count)
"""

from sklearn.model_selection import KFold, cross_val_score

class XGBoostVacancyPredictor:
    def __init__(self, training_data_path="outputs.vfinder/training_results.json", 
                 model_path="outputs.json/xgboost_model.json", n_splits=5, random_state=42):
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
        # Se usa la configuración definida por el usuario en inputs_param.py
        self.columns = PREDICTOR_COLUMNS  
        self._load_data_and_train()
    
    def _load_data_and_train(self):
        """Carga los datos, realiza validación cruzada, entrena el modelo y lo guarda."""
        # Cargar los datos de entrenamiento desde el archivo JSON
        with open(self.training_data_path, "r") as f:
            data = json.load(f)
        
        # Extraer las variables predictoras según las columnas configuradas y la variable objetivo "vacancias"
        feature_list = []
        for col in self.columns:
            if col in data:
                feature_list.append(data[col])
            else:
                raise ValueError(f"La columna '{col}' no se encuentra en los datos de entrenamiento.")
        X = np.column_stack(feature_list)
        y = np.array(data["vacancias"])
        
        # Definir el esquema de validación cruzada con KFold.
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Realizar la validación cruzada calculando el Error Cuadrático Medio (MSE)
        scores = cross_val_score(self.model, X, y, scoring='neg_mean_squared_error', cv=kfold)
        mse_scores = -scores  # Convertir a valores positivos para interpretarlos
        print("Puntuaciones MSE en cada pliegue XGBoost:", mse_scores)
        print("MSE promedio XGBoost:", mse_scores.mean())
        
        # Entrenar el modelo sobre todos los datos y guardar el modelo entrenado.
        self.model.fit(X, y)
        self.model.save_model(self.model_path)
        print(f"Modelo XGBoost entrenado y guardado en '{self.model_path}'")
    
    def predict(self, sample_input):
        """
        Realiza una predicción dado un sample input.
        
        Args:
            sample_input (list o numpy.ndarray): Debe tener la forma (n_samples, n_features)
                donde n_features es la cantidad de columnas definidas en inputs_param.py.
                
        Returns:
            numpy.ndarray: Predicción realizada por el modelo.
        """
        sample_input = np.array(sample_input)
        prediction = self.model.predict(sample_input)
        return prediction
"""
if __name__ == "__main__":
    # Ejemplo de uso:
    # Se asume que en inputs_param.py se define, por ejemplo:
    # PREDICTOR_COLUMNS = ['sm_mesh_training', 'vecinos']
    predictor = XGBoostVacancyPredictor("outputs.vfinder/training_results.json", "outputs.json/xgboost_model.json")
    
    # Ejemplo de predicción: 
    # Si se configuraron 2 columnas, sample_input debe tener 2 valores por muestra.
    sample = [[100, 14]]  # Reemplaza por los valores reales
    prediction = predictor.predict(sample)
    print("Predicción:", prediction)
    """
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Importar la configuración de columnas desde inputs_param.py
# Por ejemplo, en inputs_param.py se define:
# PREDICTOR_COLUMNS = ['sm_mesh_training', 'vecinos']
from input_params import PREDICTOR_COLUMNS

class VacancyPredictor:
    def __init__(self, json_path="outputs.vfinder/training_results.json"):
        """
        Inicializa el predictor:
         - json_path: ruta al archivo JSON con los datos de entrenamiento.
        """
        self.json_path = json_path
        self.model = None
        # Se utilizan las columnas definidas por el usuario en el archivo de parámetros
        self.columns = PREDICTOR_COLUMNS
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
        # Seleccionar las variables predictoras según la configuración y la variable objetivo "vacancias"
        X = self.df[self.columns]
        y = self.df["vacancias"]

        # Dividir en conjuntos de entrenamiento y prueba (test_size=0.3 para separar el 30% de test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

    def predict_vacancies(self, **kwargs):
        """
        Dado un conjunto de parámetros (como keyword arguments) que correspondan a las columnas
        configuradas, devuelve el número de vacancias predicho redondeado al entero "más alejado" de cero.
        
        Ejemplo de uso:
            predictor.predict_vacancies(sm_mesh_training=valor, vecinos=valor)
        
        Nota: Solo se utilizarán las columnas definidas en inputs_param.py.
        """
        # Crear un DataFrame para los nuevos datos usando únicamente las columnas configuradas
        nuevos_datos = pd.DataFrame({col: [kwargs[col]] for col in self.columns})
        # Realizar la predicción (se obtiene un valor float)
        prediction = self.model.predict(nuevos_datos)[0]
        # Redondear la predicción utilizando el método _round_away_from_zero
        return self._round_away_from_zero(prediction)
"""
if __name__ == "__main__":
    # Ejemplo de uso:
    # Se asume que en inputs_param.py se define, por ejemplo:
    # PREDICTOR_COLUMNS = ['sm_mesh_training', 'vecinos']
    predictor = VacancyPredictor("outputs.vfinder/training_results.json")
    
    # Ejemplo de predicción:
    # Si se configuraron 2 columnas, se deben pasar los valores correspondientes
    resultado = predictor.predict_vacancies(sm_mesh_training=100, vecinos=14)
    print("Vacancias predichas:", resultado)
"""