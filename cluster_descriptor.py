import os
import json
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import ConstructSurfaceModifier
from input_params import CONFIG
from key_files import KeyFilesSeparator
from clustering_reffiner import ClusterDumpProcessor

class SurfaceProcessor:
    def __init__(self, config=CONFIG[0], json_path="outputs.json/key_archivos.json", radi=None):
        """
        Inicializa el procesador cargando la configuración y la lista de archivos finales.
        
        Args:
            config (dict): Configuración extraída de input_params.
            json_path (str): Ruta al archivo JSON que contiene las listas de clusters.
            radi (list): Lista de radios a evaluar. Si es None se utiliza el valor por defecto.
        """
        self.config = config
        # Se utiliza únicamente el nivel de suavizado (smoothing level) para este procesamiento.
        self.smoothing_level = config["smoothing level"]
        self.radi = radi if radi is not None else [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.json_path = json_path
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        # Se extrae la lista de archivos que deben procesarse (clusters_final).
        self.clusters_final = self.data.get("clusters_final", [])
        self.results_matrix = None

    def process_surface_for_file(self, archivo):
        """
        Para un archivo dump, itera sobre los radios definidos, aplica el ConstructSurfaceModifier
        con cada radio, computa el pipeline y extrae el área de superficie y el volumen sólido.
        
        Args:
            archivo (str): Ruta al archivo dump.
        
        Returns:
            tuple: (best_pipeline, best_radius, best_area, best_filled_volume, cluster_size)
                best_pipeline: Pipeline con el mejor resultado.
                best_radius: El radio que produjo la mayor área.
                best_area: El área de superficie obtenida con ese radio.
                best_filled_volume: El volumen sólido obtenido con ese radio.
                cluster_size: Número total de partículas en el archivo.
        """
        best_area = 0
        best_filled_volume = 0
        best_radius = None
        best_pipeline = None

        for r in self.radi:
            pipeline = import_file(archivo)
            pipeline.modifiers.append(ConstructSurfaceModifier(
                radius=r,
                smoothing_level=self.smoothing_level,
                identify_regions=True,
                select_surface_particles=True
            ))
            data = pipeline.compute()
            cluster_size = data.particles.count

            try:
                area = data.attributes['ConstructSurfaceMesh.surface_area']
            except Exception as e:
                print(f"Error obteniendo área para radio {r} en {archivo}: {e}")
                area = 0

            try:
                filled_volume = data.attributes['ConstructSurfaceMesh.filled_volume']
            except Exception as e:
                print(f"Error obteniendo filled volume para radio {r} en {archivo}: {e}")
                filled_volume = 0

            if area > best_area:
                best_area = area
                best_filled_volume = filled_volume
                best_radius = r
                best_pipeline = pipeline

        return best_pipeline, best_radius, best_area, best_filled_volume, cluster_size

    def process_all_files(self):
        """
        Procesa cada archivo en la lista clusters_final y almacena los resultados en una matriz.
        Cada fila tiene el formato:
            [nombre_archivo, mejor_radio, área, filled_volume, número de partículas]
        
        Returns:
            np.array: Matriz de resultados.
        """
        results = []
        for archivo in self.clusters_final:
            bp, br, ba, fv, num_atm = self.process_surface_for_file(archivo)
            results.append([archivo, br, ba, fv, num_atm])
        self.results_matrix = np.array(results)
        return self.results_matrix

    def export_results(self, output_csv="resultados_procesados.csv"):
        """
        Exporta la matriz de resultados a un archivo CSV.
        
        Args:
            output_csv (str): Nombre del archivo CSV de salida.
        """
        if self.results_matrix is None:
            self.process_all_files()
        np.savetxt(output_csv, self.results_matrix, delimiter=",", fmt="%s",
                   header="archivo,mejor_radio,area,filled_volume,num_atm", comments="")
        print(f"Resultados exportados a {output_csv}")

