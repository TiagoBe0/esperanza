import json
import math
import numpy as np
import pandas as pd

# Scikit-learn y XGBoost
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Importar la configuración de columnas definida por el usuario
from input_params import PREDICTOR_COLUMNS


# ===============================
# Funciones Auxiliares
# ===============================
def load_json_data(json_path):
    """Carga los datos desde un archivo JSON y retorna un DataFrame."""
    with open(json_path, "r") as file:
        data = json.load(file)
    return pd.DataFrame(data)


# ===============================
# Predictor con Random Forest
# ===============================
class VacancyPredictorRF:
    def __init__(self, json_path="outputs.vfinder/training_results.json"):
        """
        Inicializa el predictor con Random Forest.
        - json_path: ruta al archivo JSON con los datos de entrenamiento.
        """
        self.json_path = json_path
        self.columns = PREDICTOR_COLUMNS
        # Mapeo entre nombres de columnas en el JSON y en el CSV
        self.column_mapping = {
            "sm_mesh_training": "area",
            "filled_volume": "filled_volume",
            "vecinos": "num_atm"
        }
        self.df = load_json_data(self.json_path)
        self.model = self._train_model()

    def _train_model(self):
        """
        Prepara los datos, entrena el modelo Random Forest y muestra el Error Cuadrático Medio.
        """
        X = self.df[self.columns]
        y = self.df["vacancias"]

        # División en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Entrenamiento del modelo
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Evaluación del modelo
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Error Cuadrático Medio (MSE) con Random Forest:", mse)
        return model

    def _round_up(self, x):
        """Redondea el número 'x' hacia arriba."""
        return math.ceil(x)

    def predict_vacancies(self, **kwargs):
        """
        Predice el número de vacancias utilizando los parámetros recibidos (keyword arguments).
        Ejemplo de uso:
            predictor.predict_vacancies(sm_mesh_training=valor, vecinos=valor)
        """
        # Construir DataFrame con las columnas configuradas
        data = {col: [kwargs[col]] for col in self.columns}
        nuevos_datos = pd.DataFrame(data)
        prediction = self.model.predict(nuevos_datos)[0]
        return self._round_up(prediction)

    def predict_from_csv(self, csv_path):
        """
        Carga un CSV y predice las vacancias para cada registro usando el mapeo de columnas.
        Retorna el DataFrame actualizado con una columna 'predicted_vacancies'.
        """
        df_csv = pd.read_csv(csv_path)
        ultimas_tres = df_csv.iloc[:, -3:]
        predictions = []

        for _, row in ultimas_tres.iterrows():
            features = {}
            # Construir el diccionario de características usando el mapeo definido
            for col in self.columns:
                csv_col = self.column_mapping.get(col, col)
                features[col] = row[csv_col]
            pred = self.predict_vacancies(**features)
            predictions.append(pred)

        df_csv["predicted_vacancies"] = predictions
        return df_csv


# ===============================
# Predictor con XGBoost
# ===============================
class XGBoostVacancyPredictor:
    def __init__(self, training_data_path="outputs.vfinder/training_results.json", 
                 model_path="outputs.json/xgboost_model.json", n_splits=5, random_state=42):
        """
        Inicializa el predictor con XGBoost.
         - training_data_path: ruta al archivo JSON con los datos de entrenamiento.
         - model_path: ruta donde se guardará el modelo entrenado.
         - n_splits: número de pliegues para validación cruzada.
         - random_state: semilla para reproducibilidad.
        """
        self.training_data_path = training_data_path
        self.model_path = model_path
        self.n_splits = n_splits
        self.random_state = random_state
        self.columns = PREDICTOR_COLUMNS
        self.model = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.random_state)
        self._load_data_and_train()

    def _load_data_and_train(self):
        """Carga los datos, realiza validación cruzada, entrena el modelo y lo guarda."""
        with open(self.training_data_path, "r") as f:
            data = json.load(f)

        # Extraer características y variable objetivo
        feature_list = []
        for col in self.columns:
            if col in data:
                feature_list.append(data[col])
            else:
                raise ValueError(f"La columna '{col}' no se encuentra en los datos de entrenamiento.")
        X = np.column_stack(feature_list)
        y = np.array(data["vacancias"])

        # Validación cruzada con KFold
        kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, X, y, scoring='neg_mean_squared_error', cv=kfold)
        mse_scores = -scores
        print("Puntuaciones MSE en cada pliegue XGBoost:", mse_scores)
        print("MSE promedio XGBoost:", mse_scores.mean())

        # Entrenar el modelo con todos los datos y guardar el modelo
        self.model.fit(X, y)
        self.model.save_model(self.model_path)
        print(f"Modelo XGBoost entrenado y guardado en '{self.model_path}'")

    def predict(self, sample_input):
        """
        Realiza una predicción dado un sample input.
        - sample_input: list o numpy.ndarray con forma (n_samples, n_features)
          (n_features debe coincidir con la cantidad de columnas definidas en inputs_param.py)
        Retorna:
            numpy.ndarray: Predicción realizada por el modelo.
        """
        sample_input = np.array(sample_input)
        prediction = self.model.predict(sample_input)
        return prediction


# ===============================
# Predictor con Regresión Lineal
# ===============================
class VacancyPredictor:
    def __init__(self, json_path="outputs.vfinder/training_results.json"):
        """
        Inicializa el predictor con un modelo de regresión lineal.
         - json_path: ruta al archivo JSON con los datos de entrenamiento.
        """
        self.json_path = json_path
        self.columns = PREDICTOR_COLUMNS
        self.df = load_json_data(self.json_path)
        self.model = self._train_model()

    def _train_model(self):
        """
        Prepara los datos, entrena el modelo de regresión lineal y muestra el Error Cuadrático Medio.
        """
        X = self.df[self.columns]
        y = self.df["vacancias"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("Error Cuadrático Medio (MSE):", mse)
        return model

    def _round_away_from_zero(self, x):
        """
        Redondea 'x' alejándose de cero:
          - Para x positivo usa math.ceil.
          - Para x negativo usa math.floor.
          - Para 0, retorna 0.
        """
        if x > 0:
            return math.ceil(x)
        elif x < 0:
            return math.floor(x)
        else:
            return 0

    def predict_vacancies(self, **kwargs):
        """
        Predice el número de vacancias utilizando los parámetros recibidos.
        Ejemplo:
            predictor.predict_vacancies(sm_mesh_training=valor, vecinos=valor)
        Solo se usan las columnas definidas en inputs_param.py.
        """
        nuevos_datos = pd.DataFrame({col: [kwargs[col]] for col in self.columns})
        prediction = self.model.predict(nuevos_datos)[0]
        return self._round_away_from_zero(prediction)


# ===============================
# Bloque de Ejecución (Ejemplos de uso)
# ===============================
if __name__ == "__main__":
    # --- Ejemplo con Random Forest ---
    print("=== Predictor con Random Forest ===")
    predictor_rf = VacancyPredictorRF("outputs.vfinder/training_results.json")

    # Ejemplo de predicción individual:
    prediccion_rf = predictor_rf.predict_vacancies(sm_mesh_training=100, vecinos=14)
    print("Vacancias predichas (RF):", prediccion_rf)

    # Ejemplo de procesamiento de CSV:
    # df_resultados = predictor_rf.predict_from_csv("resultados_procesados.csv")
    # print(df_resultados.head())

    # --- Ejemplo con XGBoost ---
    print("\n=== Predictor con XGBoost ===")
    predictor_xgb = XGBoostVacancyPredictor(
        training_data_path="outputs.vfinder/training_results.json",
        model_path="outputs.json/xgboost_model.json"
    )
    # Ejemplo de predicción con XGBoost:
    sample_input = [[100, 14]]  # Ajustar según la cantidad de columnas configuradas
    prediccion_xgb = predictor_xgb.predict(sample_input)
    print("Predicción (XGBoost):", prediccion_xgb)

    # --- Ejemplo con Regresión Lineal ---
    print("\n=== Predictor con Regresión Lineal ===")
    predictor_lr = VacancyPredictor("outputs.vfinder/training_results.json")
    prediccion_lr = predictor_lr.predict_vacancies(sm_mesh_training=100, vecinos=14)
    print("Vacancias predichas (Regresión Lineal):", prediccion_lr)

    # --- Ejemplo especial de comparación en CSV ---
    # Cargar referencia para una vacancia única
    with open("outputs.vfinder/key_single_vacancy.json", "r") as f:
        single_vac = json.load(f)
    ref_area = single_vac["surface_area"][0]
    ref_filled_volume = single_vac["filled_volume"][0]
    ref_vecinos = single_vac["vecinos"][0]

    # Procesar CSV y forzar predicción 1 si los valores son similares a la vacancia única
    df_csv = pd.read_csv("resultados_procesados.csv")
    ultimas_tres = df_csv.iloc[:, -3:]
    total_count = 0

    for i, (area, filled_volume, num_atm) in enumerate(zip(ultimas_tres.iloc[:, 0],
                                                            ultimas_tres.iloc[:, 1],
                                                            ultimas_tres.iloc[:, 2])):
        if (math.isclose(area, ref_area, rel_tol=0.1) and 
            math.isclose(filled_volume, ref_filled_volume, rel_tol=0.1) and 
            (num_atm == ref_vecinos)):
            vacancias_pred = 1
            print(f"Iteración {i}: Valores similares a vacancia única, predicción forzada a 1")
        else:
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
