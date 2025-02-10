import os
import json
import math
import sys
import numpy as np
import pandas as pd
from ovito.io import import_file, export_file
from ovito.modifiers import (ExpressionSelectionModifier,
                             DeleteSelectedModifier,
                             ConstructSurfaceModifier,
                             VoronoiAnalysisModifier,
                             InvertSelectionModifier,
                             ClusterAnalysisModifier)
from va_input_params import LAYERS

class ClusterFilter:
    def __init__(self, input_file, output_file=None):
        self.input_file = input_file
        self.output_file = output_file if output_file is not None else input_file
        self.data = None

    def load_data(self):
        with open(self.input_file, 'r') as f:
            self.data = json.load(f)

    def filter_clusters(self):
        clusters = self.data.get("clusters", [])
        filtered = [cluster for cluster in clusters if not all(value == 0 for value in cluster)]
        self.data["clusters"] = filtered
        self.data["num_clusters"] = len(filtered)

    def save_data(self):
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(self.data, f, indent=4)

    def run(self):
        self.load_data()
        self.filter_clusters()
        self.save_data()


class ClusterProcessor:
    def __init__(self, layers=LAYERS):
        # Se toma el primer elemento de LAYERS
        self.primer_elemento = layers[0]
        self.relax = self.primer_elemento['relax']
        self.defect = self.primer_elemento['defect']
        self.radius = self.primer_elemento['radius']
        self.smoothing_level = self.primer_elemento['smoothing level']
        self.cutoff_radius = self.primer_elemento['cutoff radius']
        
        # Listas para almacenar los datos de los clusters
        self.clusters = []
        self.surface_area = []
        self.vecinos = []
        self.menor_norma = []
        self.mayores_norma = []
        self.coordenadas_cm = []

    def load_clusters(self, json_file):
        """Carga y parsea los datos del JSON con los clusters."""
        with open(json_file, 'r') as f:
            datos = json.load(f)
        self.clusters = datos.get("clusters", [])
        self._parse_clusters()

    def _parse_clusters(self):
        """Extrae las columnas de la matriz de clusters y las almacena en listas."""
        # Reiniciar las listas
        self.surface_area = []
        self.vecinos = []
        self.menor_norma = []
        self.mayores_norma = []
        self.coordenadas_cm = []
        for fila in self.clusters:
            self.surface_area.append(fila[0])
            self.vecinos.append(fila[1])
            self.menor_norma.append(fila[2])
            self.mayores_norma.append(fila[3])
            self.coordenadas_cm.append(fila[4:7])
    
    def get_max_cluster(self):
        """Obtiene el valor máximo de 'mayores_norma', el índice y las coordenadas del centro de masa."""
        if not self.mayores_norma:
            raise ValueError("No se han cargado datos de 'mayores_norma'.")
        max_radius = np.max(self.mayores_norma)
        indice_max = np.argmax(self.mayores_norma)
        centro_masa_max = self.coordenadas_cm[indice_max]
        return max_radius, indice_max, centro_masa_max

    def process_pipeline_ids(self, output_ids_file):
        """
        Realiza modificaciones en el pipeline a partir del cluster con mayor norma y exporta 
        los IDs al archivo especificado.
        """
        max_radius, indice_max, centro_masa_max = self.get_max_cluster()
        # Construir la condición para seleccionar partículas dentro de la esfera definida.
        condition = (
            f"(Position.X-{centro_masa_max[0]})*(Position.X-{centro_masa_max[0]})+"
            f"(Position.Y-{centro_masa_max[1]})*(Position.Y-{centro_masa_max[1]})+"
            f"(Position.Z-{centro_masa_max[2]})*(Position.Z-{centro_masa_max[2]})<="
            f"{max_radius*max_radius}"
        )
        pipeline = import_file(self.relax)
        pipeline.modifiers.append(ExpressionSelectionModifier(expression=condition))
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        try:
            export_file(pipeline, output_ids_file, "lammps/dump",
                        columns=["Particle Identifier", "Particle Type",
                                 "Position.X", "Position.Y", "Position.Z"])
            pipeline.modifiers.clear()
            print(f"Pipeline exportado a {output_ids_file}")
        except Exception as e:
            print(f"Error al exportar el archivo: {e}")

    @staticmethod
    def extraer_ids(archivo):
        """
        Lee el archivo y extrae los IDs de las partículas,
        saltándose las primeras 9 líneas (por ejemplo, encabezados).
        """
        ids = []
        with open(archivo, 'r') as f:
            # Saltar las primeras 9 líneas (ajustar según el archivo)
            for _ in range(9):
                next(f)
            # Extraer el primer valor de cada línea
            for linea in f:
                valores = linea.split()
                ids.append(valores[0])
        return ids

    @staticmethod
    def crear_condicion_ids(ids_eliminar):
        """
        A partir de una lista de IDs, crea una cadena de condición con la forma:
        ParticleIdentifier==id1 || ParticleIdentifier==id2 || ...
        """
        condicion = " || ".join([f"ParticleIdentifier=={id}" for id in ids_eliminar])
        return condicion

    def compute_max_distance(self, data):
        """
        Calcula el centro de masa a partir de las posiciones de las partículas en 'data'
        y devuelve la mayor distancia entre el centro de masa y las partículas.
        """
        positions = data.particles.positions
        center_of_mass = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center_of_mass, axis=1)
        return np.max(distances)
    
    def compute_min_distance(self, data):
        """
        Calcula el centro de masa a partir de las posiciones de las partículas en 'data'
        y devuelve la menor distancia entre el centro de masa y las partículas.
        """
        positions = data.particles.positions
        center_of_mass = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center_of_mass, axis=1)
        return np.min(distances)

    def run_training(self, ids_file, output_training_file):
        """
        Ejecuta la iteración en la que se eliminan partículas incrementalmente, 
        se calcula la malla superficial, se cuentan los vecinos y se calculan las mayores
        y menores distancias al centro de masa. Los resultados se exportan en formato JSON.
        """
        total_ids = self.extraer_ids(ids_file)
        print("Total de IDs extraídos:", total_ids)
        pipeline_2 = import_file(self.relax)
        sm_mesh_training = []
        vacancias = []
        vecinos = []
        max_distancias = []  # Mayor distancia al centro de masa
        min_distancias = []  # Menor distancia al centro de masa
        
        for index in range(len(total_ids)):
            # Seleccionar los primeros (index+1) IDs para eliminar
            ids_a_eliminar = total_ids[:index+1]
            condition_f = self.crear_condicion_ids(ids_a_eliminar)
            pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=condition_f))
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            pipeline_2.modifiers.append(ConstructSurfaceModifier(
                radius=self.radius,
                smoothing_level=16,
                identify_regions=True,
                select_surface_particles=True
            ))
            data_2 = pipeline_2.compute()
            sm_elip = data_2.attributes.get('ConstructSurfaceMesh.surface_area', 0)
            sm_mesh_training.append(sm_elip)
            vacancias.append(index+1)
            
            pipeline_2.modifiers.append(InvertSelectionModifier())
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            data_3 = pipeline_2.compute()
            max_dist = self.compute_max_distance(data_3)
            min_dist = self.compute_min_distance(data_3)
            max_distancias.append(max_dist)
            min_distancias.append(min_dist)
            vecinos.append(data_3.particles.count)
            
            pipeline_2.modifiers.clear()

        datos_exportar = {
            "sm_mesh_training": sm_mesh_training,
            "vacancias": vacancias,
            "vecinos": vecinos,
            "max_distancias": max_distancias,
            "min_distancias": min_distancias
        }
        with open(output_training_file, "w") as f:
            json.dump(datos_exportar, f, indent=4)
        print(f"Resultados exportados a '{output_training_file}'.")

# -------------------------------------------------------------------
# Método main para probar las clases
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo de prueba para ClusterFilter:
    input_filter = "input_clusters.json"      # Reemplaza con la ruta de tu archivo de entrada para clusters
    output_filter = "filtered_clusters.json"    # Archivo de salida para clusters filtrados
    cluster_filter = ClusterFilter(input_filter, output_filter)
    cluster_filter.run()
    print(f"Cluster filtering completado. Archivo filtrado: {output_filter}")
    
    # Ejemplo de prueba para ClusterProcessor:
    # Se asume que el archivo JSON con datos de clusters (filtrados o de otro origen) está disponible.
    clusters_json = output_filter  # O la ruta que corresponda a tu archivo con datos de clusters
    processor = ClusterProcessor()
    processor.load_clusters(clusters_json)
    
    try:
        max_radius, indice_max, centro_masa_max = processor.get_max_cluster()
        print("Máximo cluster:", max_radius, "en índice:", indice_max, "con centro de masa:", centro_masa_max)
    except Exception as e:
        print("Error al obtener el máximo cluster:", e)
    
    # Procesar pipeline de IDs usando el cluster con mayor norma:
    output_ids_file = "outputs.vfinder/ids.training.dump"
    processor.process_pipeline_ids(output_ids_file)
    
    # Ejecutar la fase de entrenamiento y exportar resultados:
    training_output_file = "outputs.vfinder/training_results.json"
    processor.run_training(output_ids_file, training_output_file)
