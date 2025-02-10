import os
import json
import math
import numpy as np
from input_params import CONFIG

class KeyFilesSeparator:
    def __init__(self, config, clusters_json_path):
        """
        Inicializa la clase con la configuración y la ruta del archivo JSON con información de clusters.
        
        Args:
            config (dict): Diccionario de configuración (por ejemplo, CONFIG[0]).
            clusters_json_path (str): Ruta del archivo JSON (por ejemplo, "outputs.json/clusters.json").
        """
        self.config = config
        self.cluster_tolerance = config.get("cluster tolerance", 1.7)
        self.clusters_json_path = clusters_json_path
        self.lista_clusters_final = []
        self.lista_clusters_criticos = []
        self.num_clusters = self.cargar_num_clusters()

    def cargar_num_clusters(self):
        """Carga el número de clusters desde el archivo JSON."""
        if not os.path.exists(self.clusters_json_path):
            print(f"El archivo {self.clusters_json_path} no existe.")
            return 0
        with open(self.clusters_json_path, "r", encoding="utf-8") as f:
            datos = json.load(f)
        num = datos.get("num_clusters", 0)
        print("Número de clusters:", num)
        return num

    def extraer_coordenadas(self, file_path):
        """
        Lee un archivo con el formato dado y extrae las coordenadas (x, y, z)
        de las líneas que siguen a la cabecera 'ITEM: ATOMS'.
        
        Args:
            file_path (str): Ruta del archivo a leer.
        
        Returns:
            list of tuple: Lista de tuplas (x, y, z) con los valores extraídos.
        """
        coordenadas = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"No se encontró el archivo: {file_path}")
            return coordenadas

        start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("ITEM: ATOMS"):
                start_index = i + 1  # Los datos comienzan en la siguiente línea.
                break

        if start_index is None:
            print("No se encontró la sección 'ITEM: ATOMS' en el archivo.")
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
        """
        Calcula el centro de masa de un conjunto de puntos 3D.
        
        Args:
            coordenadas (list of tuple): Lista de tuplas (x, y, z).
        
        Returns:
            tuple: Centro de masa (x, y, z) o None si la lista está vacía.
        """
        arr = np.array(coordenadas)
        if arr.size == 0:
            return None
        centro = arr.mean(axis=0)
        return tuple(centro)

    def calcular_dispersion(self, coordenadas, centro_de_masa):
        """
        Calcula las distancias Euclidianas de cada punto al centro de masa y la dispersión (desviación estándar)
        de estas distancias.
        
        Args:
            coordenadas (list of tuple): Lista de tuplas (x, y, z).
            centro_de_masa (tuple): Tupla (cx, cy, cz).
        
        Returns:
            tuple: (distancias, dispersion)
        """
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
        """
        Extrae las coordenadas (x, y, z) de un archivo y arma una matriz
        en la que cada fila es [x, y, z, 0].
        
        Args:
            archivo (str): Ruta del archivo.
            
        Returns:
            np.ndarray: Matriz de dimensiones (n, 4).
        """
        coords = self.extraer_coordenadas(archivo)
        matriz = []
        for (x, y, z) in coords:
            matriz.append([x, y, z, 0])
        return np.array(matriz)

    def separar_archivos(self):
        """
        Itera sobre los archivos (con nombre "outputs.dump/key_area_{i}.dump") y, 
        según la dispersión de las distancias al centro de masa, clasifica cada archivo
        en dos listas: 'lista_clusters_criticos' y 'lista_clusters_final'.
        """
        for i in range(1, self.num_clusters + 1):
            ruta_archivo = f"outputs.dump/key_area_{i}.dump"
            coords = self.extraer_coordenadas(ruta_archivo)
            centroide = self.calcular_centro_de_masa(coords)
            #print("Coordenadas extraídas (centroide):", centroide)
            distancias, dispersion = self.calcular_dispersion(coords, centroide)
            #print("Distancias al centro:", distancias)
            print("Desviación estándar de las distancias:", dispersion)
            if dispersion > self.cluster_tolerance:
                print("Cluster a revisar:", ruta_archivo)
                self.lista_clusters_criticos.append(ruta_archivo)
            else:
                self.lista_clusters_final.append(ruta_archivo)
        print("Lista de clusters finales:")
        print(self.lista_clusters_final)
        print("Lista de clusters críticos:")
        print(self.lista_clusters_criticos)

    def exportar_listas(self, output_path):
        """
        Exporta las dos listas (clusters críticos y clusters finales) a un archivo JSON.
        
        Args:
            output_path (str): Ruta del archivo de salida.
        """
        datos_exportar = {
            "clusters_criticos": self.lista_clusters_criticos,
            "clusters_final": self.lista_clusters_final
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(datos_exportar, f, indent=4)
        print(f"Las listas se han exportado exitosamente a {output_path}")

    def run(self):
        """
        Ejecuta el proceso completo:
          - Separa los archivos en función de la dispersión.
          - Exporta las listas resultantes a un archivo JSON.
        """
        self.separar_archivos()
        self.exportar_listas("outputs.json/key_archivos.json")

"""
# --- Método main ---
if __name__ == "__main__":
    # Importa la configuración (se asume que CONFIG es una lista)
    from input_params import CONFIG
    config = CONFIG[0]
    # Instanciar la clase KeyFilesSeparator
    separator = KeyFilesSeparator(config, os.path.join("outputs.json", "clusters.json"))
    # Ejecutar el proceso completo
    separator.run()
"""