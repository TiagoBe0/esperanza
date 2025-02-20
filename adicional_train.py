import json
import pandas as pd
import numpy as np
from scipy.optimize import brentq

class VacancyPredictorCurve:
    def __init__(self, training_json_path, csv_path, degree=3):
        """
        Inicializa el predictor con los paths de los datos de entrenamiento y del CSV,
        además del grado del polinomio a ajustar.
        """
        self.training_json_path = training_json_path
        self.csv_path = csv_path
        self.degree = degree
        self.training_data = None
        self.vacancias_train = None
        self.surface_area_train = None
        self.poly = None
        self.min_area_train = None
        self.max_area_train = None

    def load_training_data(self, as_dataframe=False):
        """
        Carga los datos de entrenamiento desde un archivo JSON.
        Si as_dataframe es True, retorna un DataFrame de pandas.
        """
        with open(self.training_json_path, "r") as f:
            data = json.load(f)
        if as_dataframe:
            data = pd.DataFrame(data)
        self.training_data = data
        return self.training_data

    def prepare_training_data(self):
        """
        Prepara los datos de entrenamiento extrayendo las columnas 'vacancias' y 'sm_mesh_training'
        a partir del tercer elemento, y define el rango de áreas de entrenamiento.
        """
        if self.training_data is None:
            raise ValueError("Los datos de entrenamiento no han sido cargados.")
        self.vacancias_train = self.training_data["vacancias"].iloc[2:]
        self.surface_area_train = self.training_data["sm_mesh_training"].iloc[2:]
        self.min_area_train = self.surface_area_train.min()
        self.max_area_train = self.surface_area_train.max()

    def fit_curve(self):
        """
        Ajusta un polinomio de grado 'degree' a los datos de entrenamiento.
        """
        if self.vacancias_train is None or self.surface_area_train is None:
            raise ValueError("Los datos de entrenamiento no han sido preparados.")
        coef = np.polyfit(self.vacancias_train, self.surface_area_train, deg=self.degree)
        self.poly = np.poly1d(coef)
        return self.poly

    def predict_vacancies_from_area(self, observed_area, vacancy_range=(1, 9), area_range=(None, None)):
        """
        Predice el número de vacancias para un área observada.
        Si el área está fuera del rango de entrenamiento se 'clampa' al valor mínimo o máximo.
        """
        if self.poly is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta fit_curve primero.")
        min_area, max_area = area_range
        if min_area is not None and observed_area < min_area:
            return vacancy_range[0]
        if max_area is not None and observed_area > max_area:
            return vacancy_range[1]
        def f(x):
            return self.poly(x) - observed_area
        try:
            vac_pred = brentq(f, vacancy_range[0], vacancy_range[1])
            return vac_pred
        except ValueError:
            return None

    def plot_training_fit(self):
        """
        Genera un gráfico de los datos de entrenamiento y el polinomio ajustado.
        """
        if self.vacancias_train is None or self.surface_area_train is None:
            raise ValueError("Los datos de entrenamiento no han sido preparados.")
        if self.poly is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta fit_curve primero.")
        x_fit = np.linspace(self.vacancias_train.min(), self.vacancias_train.max(), 100)
        y_fit = self.poly(x_fit)


    def predict_from_csv(self):
        """
        Carga el CSV con los datos de defecto y predice las vacancias para cada valor de 'area'.
        Retorna el DataFrame actualizado con las predicciones.
        """
        csv_data = pd.read_csv(self.csv_path)
        predictions = []
        for idx, row in csv_data.iterrows():
            observed_area = row["area"]
            pred = self.predict_vacancies_from_area(
                observed_area,
                vacancy_range=(1, 9),
                area_range=(self.min_area_train, self.max_area_train)
            )
            predictions.append(pred)
        csv_data["predicted_vacancies"] = predictions
        return csv_data

