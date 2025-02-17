import os
import json
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier
import math
import numpy as np
from input_params import CONFIG

class ExportClusterList:
    def __init__(self, json_path="outputs.json/key_archivos.json"):
        self.json_path = json_path
        self.load_config()
    
    def load_config(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.clusters_criticos = self.data.get("clusters_criticos", [])
        self.clusters_final = self.data.get("clusters_final", [])
    
    def save_config(self):
        self.data["clusters_final"] = self.clusters_final
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)
    
    def obtener_grupos_cluster(self, file_path):
        clusters = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        atom_header_line = None
        for i, line in enumerate(lines):
            if line.startswith("ITEM: ATOMS"):
                atom_header_line = line.strip()
                data_start = i + 1
                break
        if atom_header_line is None:
            raise ValueError("No se encontró la sección 'ITEM: ATOMS' en el archivo.")
        header_parts = atom_header_line.split()[2:]
        try:
            cluster_index = header_parts.index("Cluster")
        except ValueError:
            raise ValueError("La columna 'Cluster' no se encontró en la cabecera.")
        for line in lines[data_start:]:
            if line.startswith("ITEM:"):
                break
            if line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) <= cluster_index:
                continue
            clusters.append(parts[cluster_index])
        unique_clusters = set(clusters)
        return unique_clusters, clusters

    def process_files(self):
        for archivo in self.clusters_criticos:
            try:
                unique_clusters, _ = self.obtener_grupos_cluster(archivo)
            except Exception as e:
                continue
            for i in range(0, len(unique_clusters)):
                pipeline = import_file(archivo)
                pipeline.modifiers.append(ExpressionSelectionModifier(expression=f"Cluster!={i}"))
                pipeline.modifiers.append(DeleteSelectedModifier())
                try:
                    nuevo_archivo = f"{archivo}.{i}"
                    export_file(pipeline, nuevo_archivo, "lammps/dump", 
                                columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
                    pipeline.modifiers.clear()
                    self.clusters_final.append(nuevo_archivo)
                except Exception as e:
                    pass
        self.save_config()

class KeyFilesSeparator:
    def __init__(self, config, clusters_json_path):
        self.config = config
        self.cluster_tolerance = config.get("cluster tolerance", 1.7)
        self.clusters_json_path = clusters_json_path
        self.lista_clusters_final = []
        self.lista_clusters_criticos = []
        self.num_clusters = self.cargar_num_clusters()

    def cargar_num_clusters(self):
        if not os.path.exists(self.clusters_json_path):
            return 0
        with open(self.clusters_json_path, "r", encoding="utf-8") as f:
            datos = json.load(f)
        num = datos.get("num_clusters", 0)
        return num

    def extraer_coordenadas(self, file_path):
        coordenadas = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return coordenadas
        start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("ITEM: ATOMS"):
                start_index = i + 1
                break
        if start_index is None:
            return coordenadas
        for line in lines[start_index:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                coordenadas.append((x, y, z))
            except ValueError:
                continue
        return coordenadas

    def calcular_centro_de_masa(self, coordenadas):
        arr = np.array(coordenadas)
        if arr.size == 0:
            return None
        centro = arr.mean(axis=0)
        return tuple(centro)

    def calcular_dispersion(self, coordenadas, centro_de_masa):
        if coordenadas is None or (hasattr(coordenadas, '__len__') and len(coordenadas) == 0) or centro_de_masa is None:
            return [], 0
        distancias = []
        cx, cy, cz = centro_de_masa
        for (x, y, z) in coordenadas:
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
            distancias.append(d)
        dispersion = np.std(distancias)
        return distancias, dispersion

    def construir_matriz_coordenadas(self, archivo):
        coords = self.extraer_coordenadas(archivo)
        matriz = []
        for (x, y, z) in coords:
            matriz.append([x, y, z, 0])
        return np.array(matriz)

    def separar_archivos(self):
        for i in range(1, self.num_clusters + 1):
            ruta_archivo = f"outputs.dump/key_area_{i}.dump"
            coords = self.extraer_coordenadas(ruta_archivo)
            centroide = self.calcular_centro_de_masa(coords)
            distancias, dispersion = self.calcular_dispersion(coords, centroide)
            if dispersion > self.cluster_tolerance:
                self.lista_clusters_criticos.append(ruta_archivo)
            else:
                self.lista_clusters_final.append(ruta_archivo)

    def exportar_listas(self, output_path):
        datos_exportar = {
            "clusters_criticos": self.lista_clusters_criticos,
            "clusters_final": self.lista_clusters_final
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(datos_exportar, f, indent=4)

    def run(self):
        self.separar_archivos()
        self.exportar_listas("outputs.json/key_archivos.json")
