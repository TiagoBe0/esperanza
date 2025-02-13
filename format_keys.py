import os
import json
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier
import os
import json
import math
import numpy as np
from input_params import CONFIG

class ExportClusterList:
    def __init__(self, json_path="outputs.json/key_archivos.json"):
        """
        Inicializa el procesador de clusters cargando la configuración del archivo JSON.
        
        Args:
            json_path (str): Ruta al archivo JSON que contiene las listas de archivos.
        """
        self.json_path = json_path
        self.load_config()
    
    def load_config(self):
        """Carga el archivo JSON de configuración y extrae las listas de clusters críticos y finales."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.clusters_criticos = self.data.get("clusters_criticos", [])
        self.clusters_final = self.data.get("clusters_final", [])
    
    def save_config(self):
        """Actualiza el JSON de configuración con la nueva lista de clusters finales."""
        self.data["clusters_final"] = self.clusters_final
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)
    
    def obtener_grupos_cluster(self, file_path):
        """
        Abre el archivo de dump, localiza la sección "ITEM: ATOMS", extrae la columna 'Cluster'
        y retorna el conjunto de valores únicos, para saber en cuántos grupos se dividieron las partículas.
        
        Args:
            file_path (str): Ruta al archivo de dump.
            
        Returns:
            tuple: (unique_clusters, clusters)
                unique_clusters: Conjunto de valores únicos en la columna Cluster.
                clusters: Lista con todos los valores extraídos de la columna Cluster.
        """
        clusters = []
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Buscar la línea que inicia la sección de átomos
        atom_header_line = None
        for i, line in enumerate(lines):
            if line.startswith("ITEM: ATOMS"):
                atom_header_line = line.strip()
                data_start = i + 1  # Los datos comienzan en la siguiente línea
                break

        if atom_header_line is None:
            raise ValueError("No se encontró la sección 'ITEM: ATOMS' en el archivo.")

        # La cabecera se define después de "ITEM: ATOMS" (por ejemplo: id type x y z Cluster)
        header_parts = atom_header_line.split()[2:]  # se ignoran "ITEM:" y "ATOMS"
        try:
            cluster_index = header_parts.index("Cluster")
        except ValueError:
            raise ValueError("La columna 'Cluster' no se encontró en la cabecera.")

        # Recorrer las líneas siguientes hasta que se encuentre otro ITEM: o se acaben los datos
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
        """
        Procesa cada archivo crítico:
          - Extrae la información de clusters.
          - Para cada grupo identificado (se asume numeración de 0 a n-1) se exporta un archivo
            que contiene únicamente las partículas de ese grupo.
          - Se añade el nuevo archivo a la lista de clusters finales.
          - Finalmente, se actualiza el archivo JSON.
        """
        for archivo in self.clusters_criticos:
            try:
                unique_clusters, _ = self.obtener_grupos_cluster(archivo)
                print(f"Valores únicos en la columna 'Cluster' de {archivo}: {unique_clusters}")
                if len(unique_clusters) == 2:
                    print("El archivo contiene 2 grupos de clusters.")
                elif len(unique_clusters) == 3:
                    print("El archivo contiene 3 grupos de clusters.")
                else:
                    print(f"El archivo {archivo} contiene {len(unique_clusters)} grupos, se esperaba 2 o 3.")
            except Exception as e:
                print(f"Error al procesar el archivo {archivo}: {e}")
                continue

            # Se asume que los clusters están numerados de 0 a n-1
            for i in range(0, len(unique_clusters)):
                pipeline = import_file(archivo)
                # Seleccionar las partículas que NO pertenecen al cluster i
                pipeline.modifiers.append(ExpressionSelectionModifier(expression=f"Cluster!={i}"))
                # Eliminar las partículas seleccionadas, dejando únicamente el cluster i
                pipeline.modifiers.append(DeleteSelectedModifier())
                try:
                    nuevo_archivo = f"{archivo}.{i}"
                    export_file(pipeline, nuevo_archivo, "lammps/dump", 
                                columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
                    pipeline.modifiers.clear()
                    self.clusters_final.append(nuevo_archivo)
                    print(f"Archivo exportado: {nuevo_archivo} agregado a clusters_final.")
                except Exception as e:
                    print(f"Error al exportar {archivo} para cluster {i}: {e}")
        self.save_config()
        #print("clusters_final actualizado en el archivo JSON.")


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
