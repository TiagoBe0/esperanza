import os
import json
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import (
    ExpressionSelectionModifier,
    DeleteSelectedModifier,
    ClusterAnalysisModifier,
    ConstructSurfaceModifier,
    InvertSelectionModifier
)
from input_params import CONFIG
from format_keys import ExportClusterList,KeyFilesSeparator
from cluster_descriptor import SurfaceProcessor,ClusterDumpProcessor
from main_vacancy_analysis import TrainingProcessor
from single_vacancy import  SingleVacancyProcessor
from single_vacancy import DiVacancyProcessor

from RF_train_metodh import VacancyPredictorRF,XGBoostVacancyPredictor,VacancyPredictor

import math
import pandas as pd

import xgboost as xgb
class ClusterProcessor:
    def __init__(self):
        """
        Inicializa la clase cargando la configuración y creando las carpetas de salida.
        """
        # Se asume que CONFIG es una lista y se utiliza el primer elemento.
        self.configuracion = CONFIG[0]
        self.nombre_archivo = self.configuracion['defect']
        self.radio_sonda = self.configuracion['radius']
        self.smoothing_leveled = self.configuracion['smoothing level']
        self.cutoff_radius = self.configuracion['cutoff radius']
        
        # Rutas de salida
        self.outputs_dump = "outputs.dump"
        self.outputs_json = "outputs.json"
        os.makedirs(self.outputs_dump, exist_ok=True)
        os.makedirs(self.outputs_json, exist_ok=True)
    
    def run(self):
        """
        Ejecuta el procesamiento del archivo de defectos:
          - Aplica modificadores para construir la malla superficial y analizar clusters.
          - Exporta la tabla de clusters a un archivo JSON (mostrando solo el número de áreas clave).
          - Exporta un dump con las áreas clave y, para cada cluster, genera un archivo individual.
        """
        # Importar el archivo de entrada y aplicar modificadores.
        pipeline = import_file(self.nombre_archivo)
        pipeline.modifiers.append(ConstructSurfaceModifier(
            radius=self.radio_sonda,
            smoothing_level=self.smoothing_leveled,
            identify_regions=True,
            select_surface_particles=True
        ))
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        pipeline.modifiers.append(ClusterAnalysisModifier(
            cutoff=self.cutoff_radius,
            sort_by_size=True,
            unwrap_particles=True,
            compute_com=True
        ))
        
        data = pipeline.compute()
        num_clusters = data.attributes["ClusterAnalysis.cluster_count"]
        
        # Guardar el número de clusters en un JSON.
        datos_clusters = {"num_clusters": num_clusters}
        clusters_json_path = os.path.join(self.outputs_json, "clusters.json")
        with open(clusters_json_path, "w") as archivo:
            json.dump(datos_clusters, archivo, indent=4)
        
        # Exportar el dump de áreas clave.
        key_areas_dump_path = os.path.join(self.outputs_dump, "key_areas.dump")
        try:
            export_file(pipeline, key_areas_dump_path, "lammps/dump", 
                        columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
            pipeline.modifiers.clear()
        except Exception as e:
            print(f"Error al exportar el dump de áreas clave: {e}")
        
        # Generar y exportar un archivo individual para cada cluster.
        clusters = [f"Cluster=={i}" for i in range(1, num_clusters + 1)]
        for i, cluster_expr in enumerate(clusters, start=1):
            pipeline_2 = import_file(key_areas_dump_path)
            pipeline_2.modifiers.append(ClusterAnalysisModifier(
                cutoff=self.cutoff_radius,
                cluster_coloring=True,
                unwrap_particles=True,
                sort_by_size=True
            ))
            pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=cluster_expr))
            pipeline_2.modifiers.append(InvertSelectionModifier())
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            
            output_file = os.path.join(self.outputs_dump, f"key_area_{i}.dump")
            try:
                export_file(pipeline_2, output_file, "lammps/dump", 
                            columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
                pipeline_2.modifiers.clear()
            except Exception as e:
                print(f"Error al exportar {output_file}: {e}")
        
        # Salida clave
        print(f"Número de áreas clave encontradas: {num_clusters}")

    @staticmethod
    def extraer_encabezado(file_path: str) -> list:
        """
        Extrae el encabezado de un archivo dump, considerando todas las líneas hasta (e incluyendo)
        la línea que comienza con "ITEM: ATOMS".
        """
        encabezado = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    encabezado.append(line)
                    if line.strip().startswith("ITEM: ATOMS"):
                        break
        except Exception as e:
            print(f"Error al extraer encabezado de {file_path}: {e}")
        return encabezado

import json
import math
import pandas as pd
from RF_train_metodh import VacancyPredictorRF  # Se importa para el método RF

import json
import math
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
import math
import pandas as pd
import numpy as np
import xgboost as xgb

class VacancyPredictionRunner:
    def __init__(self, predictor, archivo,predictor_rf=None, predictor_xgb=None, other_method=False):
        # Se recibe el nombre (o ruta) del archivo de entrada.
        self.archivo = archivo

        # Cargar la referencia para vacancia única.
        with open("outputs.vfinder/key_single_vacancy.json", "r") as f:
            single_vac = json.load(f)
        self.ref_area = single_vac["surface_area"][0]
        self.ref_filled_volume = single_vac["filled_volume"][0]
        self.ref_vecinos = single_vac["vecinos"][0]

        # Cargar la referencia para divacancia.
        with open("outputs.vfinder/key_divacancy.json", "r") as f:
            diva_vac = json.load(f)
        self.ref_area_diva = diva_vac["surface_area"][0]
        self.ref_filled_volume_diva = diva_vac["filled_volume"][0]
        self.ref_vecinos_diva = diva_vac["vecinos"][0]

        self.predictor = predictor
        self.predictor_rf = predictor_rf
        self.other_method = other_method

        # Si no se proporciona un predictor XGBoost, se carga el modelo desde archivo.
        if predictor_xgb is None:
            self.predictor_xgb = xgb.XGBRegressor(objective='reg:squarederror')
            self.predictor_xgb.load_model("xgboost_model.json")
        else:
            self.predictor_xgb = predictor_xgb

        self.vector_area = None
        self.vector_filled_volume = None
        self.vector_num_atm = None
        # Vector con valores reales, si están disponibles en el CSV.
        self.vector_true = None  
        # Diccionario para almacenar resultados de cada método.
        self.results = {}

    def load_data(self):
        df = pd.read_csv("resultados_procesados.csv")
        ultimas_tres = df.iloc[:, -3:]
        self.vector_area = ultimas_tres.iloc[:, 0].values
        self.vector_filled_volume = ultimas_tres.iloc[:, 1].values
        self.vector_num_atm = ultimas_tres.iloc[:, 2].values
        # Se asume que en el CSV existe una columna 'vacancies' con los valores reales.
        print(self.vector_area)
        print(self.vector_filled_volume)
        print(self.vector_num_atm)
        if "vacancies" in df.columns:
            self.vector_true = df["vacancies"].values
        else:
            print("No se encontró la columna 'vacancies'. No se calcularán errores cuadráticos.")
            self.vector_true = None

    def predict_linear(self):
        total_count = 0
        predictions = []
        errors = []
        for i, (area, filled_volume, num_atm) in enumerate(zip(self.vector_area, self.vector_filled_volume, self.vector_num_atm)):
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or 
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or 
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"Iteración {i}: Valores similares a vacancia única, predicción forzada a 1")
            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.4) and 
                  math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.4) and  
                  (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"Iteración {i}: Valores similares a divacancia, predicción forzada a 2")
            else:
                vacancias_pred = self.predictor.predict_vacancies(area, filled_volume, num_atm)
                total_count += vacancias_pred
                print(f"Iteración {i}: Área = {area}, Filled Volume = {filled_volume}, cluster_size = {num_atm} -> Vacancias Predichas = {abs(vacancias_pred)}")
            predictions.append(vacancias_pred)
            if self.vector_true is not None:
                error = (vacancias_pred - self.vector_true[i])**2
                errors.append(error)
            else:
                errors.append(None)
        print(f"Predicción lineal regresión: {abs(total_count)}")
        if self.vector_true is not None:
            mse = sum(errors)/len(errors)
            print(f"Error cuadrático medio (Linear): {mse}")
        return total_count, predictions, errors

    def predict_rf(self):
        if not self.other_method or self.predictor_rf is None:
            print("El método RF no está habilitado o no se proporcionó predictor_rf.")
            return None, [], []
        total_count = 0
        predictions = []
        errors = []
        for i, (area, filled_volume, num_atm) in enumerate(zip(self.vector_area, self.vector_filled_volume, self.vector_num_atm)):
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or 
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or 
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"Iteración {i}: Valores similares a vacancia única, predicción forzada a 1")
            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.4) and 
                  math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.4) and 
                  (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"Iteración {i}: Valores similares a divacancia, predicción forzada a 2")
            else:
                vacancias_pred = self.predictor_rf.predict_vacancies(area, filled_volume, num_atm)
                total_count += vacancias_pred
                print(f"Iteración {i}: Área = {area}, Filled Volume = {filled_volume}, cluster_size = {num_atm} -> Vacancias Predichas = {vacancias_pred}")
            predictions.append(vacancias_pred)
            if self.vector_true is not None:
                error = (vacancias_pred - self.vector_true[i])**2
                errors.append(error)
            else:
                errors.append(None)
        print("\nContador total random forest:", total_count)
        if self.vector_true is not None and errors:
            mse = sum(errors)/len(errors)
            print(f"Error cuadrático medio (RF): {mse}")
        return total_count, predictions, errors

    def predict_xgb(self):
        total_count = 0
        predictions = []
        errors = []
        for i, (area, filled_volume, num_atm) in enumerate(zip(self.vector_area, self.vector_filled_volume, self.vector_num_atm)):
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or 
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or 
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"Iteración {i}: [XGBoost] Valores similares a vacancia única, predicción forzada a 1")
            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.4) and 
                  math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.4) and  
                  (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"Iteración {i}: [XGBoost] Valores similares a divacancia, predicción forzada a 2")
            else:
                sample_input = np.array([[area, filled_volume, num_atm]])
                vacancias_pred = self.predictor_xgb.predict(sample_input)[0]
                total_count += vacancias_pred
                print(f"Iteración {i}: [XGBoost] Área = {area}, Filled Volume = {filled_volume}, cluster_size = {num_atm} -> Vacancias Predichas = {vacancias_pred}")
            predictions.append(vacancias_pred)
            if self.vector_true is not None:
                error = (vacancias_pred - self.vector_true[i])**2
                errors.append(error)
            else:
                errors.append(None)
        print("Contador total XGBoost:", total_count)
        if self.vector_true is not None and errors:
            mse = sum(errors)/len(errors)
            print(f"Error cuadrático medio (XGBoost): {mse}")
        return total_count, predictions, errors

    def export_results(self):
        dfs = []
        for method, data in self.results.items():
            num_rows = len(data["predictions"])
            df_method = pd.DataFrame({
                "iteration": list(range(num_rows)),
                "method": [method] * num_rows,
                "prediction": data["predictions"],
                "true_value": self.vector_true if self.vector_true is not None else [None] * num_rows,
                "squared_error": data["errors"]
            })
            dfs.append(df_method)
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            # Extraer el nombre base del archivo (por ejemplo, "pfinal_16" de "inputs.dump/pfinal_16")
            base_name = os.path.basename(self.archivo)
            # Construir la ruta de salida concatenando la carpeta deseada y el nombre base con extensión .csv
            output_dir = "outputs.vfinder"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{base_name}.csv")
            final_df.to_csv(output_file, index=False)
            print(f"Resultados exportados a '{output_file}'")
        else:
            print("No hay resultados para exportar.")
    def export_totals(self):
        # Extraer los totales de cada método
        totals_dict = { method: data["total"] for method, data in self.results.items() }
        # Convertir el diccionario a DataFrame
        df_totals = pd.DataFrame(list(totals_dict.items()), columns=["modelo", "contador_total"])
        # Crear directorio de salida
        output_dir = "outputs.vfinder"
        base_name = os.path.basename(self.archivo)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base_name}_totales_vacancias.csv")
        # Exportar a CSV
        df_totals.to_csv(output_file, index=False)
        print(f"Contadores totales exportados a '{output_file}'")

    def run(self):
        self.load_data()
        self.results = {}
        print("Ejecutando predicción con regresión lineal:")
        total_linear, predictions_linear, errors_linear = self.predict_linear()
        self.results["linear"] = {"total": total_linear, "predictions": predictions_linear, "errors": errors_linear}
        if self.other_method:
            print("Ejecutando predicción con random forest:")
            total_rf, predictions_rf, errors_rf = self.predict_rf()
            self.results["rf"] = {"total": total_rf, "predictions": predictions_rf, "errors": errors_rf}
        print("Ejecutando predicción con XGBoost:")
        total_xgb, predictions_xgb, errors_xgb = self.predict_xgb()
        self.results["xgb"] = {"total": total_xgb, "predictions": predictions_xgb, "errors": errors_xgb}
        
        self.export_results()
        self.export_totals()




if __name__ == "__main__":
        #ahora se trabaja con la muestra relajada y se hace el entrenamiento 
    config = CONFIG[0]
    relax = config['relax']                          # Archivo dump original.
    radius_training = config['radius_training']      # Radio de entrenamiento.
    radius = config['radius']                        # Radio para la malla superficial.
    smoothing_level_training = config['smoothing_level_training']  # Nivel de suavizado para training.
    other_method = config['other method']  
    processor_2 = TrainingProcessor(relax, radius_training, radius, smoothing_level_training,config['strees'])
    processor_2.run()

    processor_3 = SingleVacancyProcessor(config)
    processor_3.run()

    processor_4 = DiVacancyProcessor(config)
    processor_4.run()
    ### 




    processor = ClusterProcessor()
    processor.run()

    # Clasificación de clusters en bien y mal separados
    config = CONFIG[0]
    separator = KeyFilesSeparator(config, os.path.join("outputs.json", "clusters.json"))
    separator.run()


    json_path = "outputs.json/key_archivos.json"
    archivos = ClusterDumpProcessor.cargar_lista_archivos_criticos(json_path)
    from super_critic_cluster import ClusterDumpProcessorS
    for archivo in archivos:
        try:
            processors = ClusterDumpProcessorS(archivo, decimals=5)
            processors.load_data()
            processors.process_clusters()
            processors.export_updated_file(f"{archivo}")
        except Exception as e:
            print(f"Error procesando {archivo}: {e}")
    for archivo in archivos:
        try:
            processor = ClusterDumpProcessor(archivo, decimals=5)
            processor.load_data()
            processor.process_clusters()
            processor.export_updated_file(f"{archivo}")
        except Exception as e:
            print(f"Error procesando {archivo}: {e}")

    # Clasificación adicional de clusters (según KeyFilesSeparator)
    config = CONFIG[0]
    separator = KeyFilesSeparator(config, os.path.join("outputs.json", "clusters.json"))
    separator.run()
    


    ##aca se exportan los dump segun el cluster redefinidoy s arma la lista definitiva de defectos
    processor_0 = ExportClusterList("outputs.json/key_archivos.json")
    processor_0.process_files()




    ##aca se extraern las caracteristicas defiitnitavvas de cada defecto 
    processor_1 = SurfaceProcessor()
    processor_1.process_all_files()
    processor_1.export_results()




    #predicciones finales 
    # Instanciar el predictor basado en regresión lineal
    # Cargar el archivo JSON de referencia que describe una vacancia única.
    # Se asume que el archivo tiene la siguiente estructura:
    # {
    #   "surface_area": [valor_area],
    #   "filled_volume": [valor_filled_volume],
    #   "vecinos": [valor_num_atm]
    # }
    
    predictorXG = XGBoostVacancyPredictor()
    
    predictor = VacancyPredictor("outputs.vfinder/training_results.json")
    predictor_rf = VacancyPredictorRF("outputs.vfinder/training_results.json")
    
    # Se habilita el uso de métodos adicionales (RF y XGBoost)
    runner = VacancyPredictionRunner(predictor,config['defect'], predictor_rf=predictor_rf, predictor_xgb=predictorXG,other_method=True)
    runner.run()

                
                    
                
            