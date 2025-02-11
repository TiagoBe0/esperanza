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

class ClusterDumpProcessor:
    """
    Clase para procesar archivos dump, aplicar clustering y exportar un nuevo dump
    manteniendo el encabezado original.
    """
    def __init__(self, file_path: str, decimals: int = 5):
        self.file_path = file_path
        self.decimals = decimals
        self.matriz_total = None  # Matriz de datos [id, type, x, y, z, cluster]
        self.header = None        # Encabezado original del dump
        self.subset = None        # Subconjunto de coordenadas (x, y, z)
        # Se setea el atributo divisions_of_cluster a partir del archivo CONFIG.
        # Se asume que CONFIG es una lista de diccionarios y que se usa el primero.
        self.divisions_of_cluster = CONFIG[0]['divisions_of_cluster']


    def load_data(self):
        """Carga los datos y el encabezado del archivo dump."""
        self.matriz_total = self.extraer_datos_completos(self.file_path, self.decimals)
        self.header = self.extraer_encabezado(self.file_path)
        if self.matriz_total.size == 0:
            raise ValueError(f"No se pudieron extraer datos de {self.file_path}")
        self.subset = self.matriz_total[:, 2:5]

    def calcular_dispersion(self, points: np.ndarray) -> float:
        """
        Calcula la dispersión de un conjunto de puntos como la distancia promedio
        de cada punto al centro de masa.
        """
        if points.shape[0] == 0:
            return 0.0
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        return np.mean(distances)

    def process_clusters(self):
        """
        Aplica KMeans sobre self.subset y actualiza la columna de clusters en la matriz de datos.
        Calcula la dispersión (distancia promedio al centro de masa) y, en función de ella,
        decide si se deben usar 2 o 3 clusters:
          - Si la dispersión es menor que un umbral (por ejemplo, 1.0), se usan 2 clusters.
          - De lo contrario, se usan 3 clusters.
        """
        centro_masa_global = np.mean(self.subset, axis=0)
        p1, p2, distancia_maxima = self.find_farthest_points(self.subset)
        dispersion = self.calcular_dispersion(self.subset)
        threshold = self.divisions_of_cluster # Umbral ajustable según la escala de tus datos
        if distancia_maxima < threshold:
            n_used = 2
        else:
            n_used = 3

        etiquetas = self.aplicar_kmeans(self.subset, p1, p2, centro_masa_global, n_clusters=n_used)
        if etiquetas.shape[0] != self.matriz_total.shape[0]:
            raise ValueError("El número de etiquetas no coincide con la matriz total.")
        self.matriz_total[:, 5] = etiquetas
        print(f"Número de áreas clave encontradas: {n_used}")

    def export_updated_file(self, output_file: str = None):
        """
        Exporta la matriz de datos actualizada a un archivo de texto con el encabezado original.
        """
        if output_file is None:
            output_file = f"{self.file_path}_actualizado.txt"
        fmt = ("%d %d %.5f %.5f %.5f %d")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(self.header)
                np.savetxt(f, self.matriz_total, fmt=fmt, delimiter=" ")
            print("Datos exportados exitosamente a:", output_file)
        except Exception as e:
            print(f"Error al exportar {output_file}: {e}")

    @staticmethod
    def cargar_lista_archivos_criticos(json_path: str) -> list:
        """
        Carga la lista de archivos críticos desde un archivo JSON.
        Se espera que la clave en el JSON sea "clusters_criticos".
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            return datos.get("clusters_criticos", [])
        except FileNotFoundError:
            print(f"El archivo {json_path} no existe.")
            return []
        except json.JSONDecodeError as e:
            print(f"Error al decodificar el archivo JSON: {e}")
            return []

    @staticmethod
    def extraer_datos_completos(file_path: str, decimals: int = 5) -> np.ndarray:
        """
        Lee un archivo dump y extrae una matriz de datos con formato:
        [id, type, x, y, z, cluster].
        """
        datos = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"No se encontró el archivo: {file_path}")
            return np.array([])
        start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("ITEM: ATOMS"):
                start_index = i + 1
                break
        if start_index is None:
            print(f"No se encontró la sección 'ITEM: ATOMS' en {file_path}.")
            return np.array([])
        for line in lines[start_index:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                id_val = int(parts[0])
                type_val = int(parts[1])
                x = round(float(parts[2]), decimals)
                y = round(float(parts[3]), decimals)
                z = round(float(parts[4]), decimals)
                cluster_val = int(parts[5])
                datos.append([id_val, type_val, x, y, z, cluster_val])
            except ValueError:
                continue
        return np.array(datos)

    @staticmethod
    def extraer_encabezado(file_path: str) -> list:
        """
        Extrae el encabezado de un archivo dump (hasta e incluyendo la línea que empieza con "ITEM: ATOMS").
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

    @staticmethod
    def aplicar_kmeans(coordenadas: np.ndarray, p1, p2, centro_masa_global, n_clusters: int) -> np.ndarray:
        """
        Aplica KMeans para dividir el conjunto de coordenadas en n_clusters usando centros fijos.
        - Si n_clusters == 2, se usan p1 y p2 como centros iniciales.
        - Si n_clusters == 3, se usan p1, p2 y centro_masa_global como centros iniciales.
        
        Args:
            coordenadas (np.ndarray): Arreglo (N, 3) con las coordenadas.
            p1: Primer punto extremo.
            p2: Segundo punto extremo.
            centro_masa_global: Centro de masa global de las coordenadas.
            n_clusters (int): Número de clusters (2 o 3).
            
        Returns:
            np.ndarray: Etiquetas asignadas a cada punto.
        """
        from sklearn.cluster import KMeans
        if n_clusters == 2:
            init_centers = np.array([p1, p2])
        elif n_clusters == 3:
            init_centers = np.array([p1, p2, centro_masa_global])
        else:
            raise ValueError("Solo se admite n_clusters igual a 2 o 3.")
        kmeans = KMeans(n_clusters=n_clusters,
                        init=init_centers,
                        n_init=1,
                        max_iter=300,
                        tol=1,
                        random_state=42)
        etiquetas = kmeans.fit_predict(coordenadas)
        return etiquetas

    @staticmethod
    def find_farthest_points(coordenadas: np.ndarray):
        """
        Dado un conjunto de coordenadas, encuentra los dos puntos que están más alejados
        y retorna esos dos puntos junto con la distancia máxima.
        """
        pts = np.array(coordenadas)
        n = pts.shape[0]
        if n < 2:
            return None, None, 0
        diffs = pts[:, None, :] - pts[None, :, :]
        distancias = np.sqrt(np.sum(diffs**2, axis=-1))
        idx = np.unravel_index(np.argmax(distancias), distancias.shape)
        distancia_maxima = distancias[idx]
        punto1 = pts[idx[0]]
        punto2 = pts[idx[1]]
        return punto1, punto2, distancia_maxima
"""
if __name__ == "__main__":
    # Cargar lista de archivos críticos desde el JSON.
    json_path = "outputs.json/key_archivos.json"
    archivos = ClusterDumpProcessor.cargar_lista_archivos_criticos(json_path)
    
    for archivo in archivos:
        try:
            processor = ClusterDumpProcessor(archivo, decimals=5)
            processor.load_data()
            processor.process_clusters()
            processor.export_updated_file(f"{archivo}_actualizado.txt")
        except Exception as e:
            print(f"Error procesando {archivo}: {e}")
    
    # Clasificación adicional de clusters (según KeyFilesSeparator)
    config = CONFIG[0]
    separator = KeyFilesSeparator(config, os.path.join("outputs.json", "clusters.json"))
    separator.run()
    """