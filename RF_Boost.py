import json
import numpy as np
import xgboost as xgb
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

# Ejemplo de uso desde el método main:
if __name__ == "__main__":
    # Instanciar el predictor
    predictor = XGBoostVacancyPredictor()
    
    # Definir un ejemplo:
    # Área de superficie (sm_mesh): 350.0, Volumen llenado (filled_volume): 700.0, Número de vecinos: 65
    sample_input = [[350.0, 700.0, 65]]
    prediction = predictor.predict(sample_input)
    print("Predicción para vacancias:", prediction[0])
