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
from key_files import KeyFilesSeparator
from clustering_reffiner import ClusterDumpProcessor
from format_keys import ExportClusterList
from cluster_descriptor import SurfaceProcessor
from main_vacancy_analysis import TrainingProcessor
from single_vacancy import  SingleVacancyProcessor
from Di_vacancy import DiVacancyProcessor
from RF_train import VacancyPredictor
from RF_train_metodh import VacancyPredictorRF
import math
import pandas as pd
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




if __name__ == "__main__":
    processor = ClusterProcessor()
    processor.run()

    # Clasificación de clusters en bien y mal separados
    config = CONFIG[0]
    separator = KeyFilesSeparator(config, os.path.join("outputs.json", "clusters.json"))
    separator.run()


    json_path = "outputs.json/key_archivos.json"
    archivos = ClusterDumpProcessor.cargar_lista_archivos_criticos(json_path)
    
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
    #predicciones finales 
    # Instanciar el predictor basado en regresión lineal
    predictor = VacancyPredictor("outputs.vfinder/training_results.json")
    
    # Cargar el archivo JSON de referencia que describe una vacancia única.
    # Se asume que el archivo tiene la siguiente estructura:
    # {
    #   "surface_area": [valor_area],
    #   "filled_volume": [valor_filled_volume],
    #   "vecinos": [valor_num_atm]
    # }
    # Cargar la referencia para vacancia única.
    with open("outputs.vfinder/key_single_vacancy.json", "r") as f:
        single_vac = json.load(f)
    ref_area = single_vac["surface_area"][0]
    ref_filled_volume = single_vac["filled_volume"][0]
    ref_vecinos = single_vac["vecinos"][0]

    # Cargar la referencia para divacancia.
    with open("outputs.vfinder/key_divacancy.json", "r") as f:
        diva_vac = json.load(f)
    ref_area_diva = diva_vac["surface_area"][0]
    ref_filled_volume_diva = diva_vac["filled_volume"][0]
    ref_vecinos_diva = diva_vac["vecinos"][0]

    # Cargar el CSV y extraer las últimas tres columnas.
    df = pd.read_csv("resultados_procesados.csv")
    ultimas_tres = df.iloc[:, -3:]
    vector_area = ultimas_tres.iloc[:, 0].values
    vector_filled_volume = ultimas_tres.iloc[:, 1].values
    vector_num_atm = ultimas_tres.iloc[:, 2].values

    total_count = 0

    for i, (area, filled_volume, num_atm) in enumerate(zip(vector_area, vector_filled_volume, vector_num_atm)):
        # Condición para vacancia única.
        if (math.isclose(area, ref_area, rel_tol=0.2) or
            math.isclose(filled_volume, ref_filled_volume, rel_tol=0.2) or
            (num_atm == ref_vecinos)):
            vacancias_pred = 1
            total_count += 1
            print(f"Iteración {i}: Valores similares a vacancia única, predicción forzada a 1")
        # Condición para divacancia.
        elif (math.isclose(area, ref_area_diva, rel_tol=0.4) or 
            math.isclose(filled_volume, ref_filled_volume_diva, rel_tol=0.4) or 
            (num_atm == ref_vecinos_diva)):
            vacancias_pred = 2
            total_count += 2
            print(f"Iteración {i}: Valores similares a divacancia, predicción forzada a 2")
        else:
            vacancias_pred = predictor.predict_vacancies(area, filled_volume, num_atm)
            total_count += vacancias_pred
            print(f"Iteración {i}: Área = {area}, Filled Volume = {filled_volume}, cluster_size = {num_atm} -> Vacancias Predichas = {vacancias_pred}")

    print(total_count)

    if other_method:
        from RF_train_metodh import VacancyPredictorRF
        predictor_rf = VacancyPredictorRF("outputs.vfinder/training_results.json")
        
        # Cargar la referencia para vacancia única.
        with open("outputs.vfinder/key_single_vacancy.json", "r") as f:
            single_vac = json.load(f)
        ref_area = single_vac["surface_area"][0]
        ref_filled_volume = single_vac["filled_volume"][0]
        ref_vecinos = single_vac["vecinos"][0]
        
        # Cargar la referencia para divacancia.
        with open("outputs.vfinder/key_divacancy.json", "r") as f:
            diva_vac = json.load(f)
        ref_area_diva = diva_vac["surface_area"][0]
        ref_filled_volume_diva = diva_vac["filled_volume"][0]
        ref_vecinos_diva = diva_vac["vecinos"][0]
        
        df = pd.read_csv("resultados_procesados.csv")
        ultimas_tres = df.iloc[:, -3:]
        vector_area = ultimas_tres.iloc[:, 0].values
        vector_filled_volume = ultimas_tres.iloc[:, 1].values
        vector_num_atm = ultimas_tres.iloc[:, 2].values
        
        total_count = 0
        
        for i, (area, filled_volume, num_atm) in enumerate(zip(vector_area, vector_filled_volume, vector_num_atm)):
            if (math.isclose(area, ref_area, rel_tol=0.1) or
                math.isclose(filled_volume, ref_filled_volume, rel_tol=0.1) or
                (num_atm == ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"Iteración {i}: Valores similares a vacancia única, predicción forzada a 1")
            elif (math.isclose(area, ref_area_diva, rel_tol=0.1) or
                math.isclose(filled_volume, ref_filled_volume_diva, rel_tol=0.1) or
                (num_atm == ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"Iteración {i}: Valores similares a divacancia, predicción forzada a 2")
            else:
                vacancias_pred = predictor_rf.predict_vacancies(area, filled_volume, num_atm)
                total_count += vacancias_pred
                print(f"Iteración {i}: Área = {area}, Filled Volume = {filled_volume}, cluster_size = {num_atm} -> Vacancias Predichas = {vacancias_pred}")
        print("\nContador total:", total_count)

                
                    
                
            