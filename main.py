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
    
    processor = TrainingProcessor(relax, radius_training, radius, smoothing_level_training)
    processor.run()