import os
import json
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier

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

