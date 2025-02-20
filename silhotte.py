import json
import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from ovito.io import import_file

# ================================================================
# Clase de utilidades para cargar datos y encabezados
# ================================================================

class UtilidadesClustering:
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
    def extraer_datos_completos(file_path: str) -> np.ndarray:
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
                x = round(float(parts[2]), 4)
                y = round(float(parts[3]), 4)
                z = round(float(parts[4]), 4)
                cluster_val = int(parts[5])
                datos.append([id_val, type_val, x, y, z, cluster_val])
            except ValueError:
                continue

        return np.array(datos)

    @staticmethod
    def extraer_encabezado(file_path: str) -> list:
        """
        Extrae el encabezado de un archivo dump (hasta e incluyendo la línea que empieza con 'ITEM: ATOMS').
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

# ================================================================
# Funciones auxiliares de fusión y subdivisión
# ================================================================

def merge_clusters(labels, c1, c2):
    """
    Fusiona c2 en c1, re-etiquetando c2 -> c1.
    Retorna un nuevo array de etiquetas (copiado).
    """
    new_labels = np.copy(labels)
    new_labels[new_labels == c2] = c1
    return new_labels

def compute_dispersion(coords, labels):
    """
    Retorna un dict {cluster_label: dispersion_std}, donde la dispersión
    es el std de las distancias al centro de masa de cada clúster.
    """
    dispersion_dict = {}
    for c in np.unique(labels):
        mask = (labels == c)
        if not np.any(mask):
            dispersion_dict[c] = np.nan
            continue
        
        cluster_coords = coords[mask]
        center_of_mass = cluster_coords.mean(axis=0)
        distances = np.linalg.norm(cluster_coords - center_of_mass, axis=1)
        dispersion_dict[c] = distances.std()
    return dispersion_dict

def silhouette_mean(coords, labels):
    """
    Calcula y devuelve el silhouette medio.
    """
    sil_vals = silhouette_samples(coords, labels)
    return np.mean(sil_vals)

def try_all_merges(coords, labels):
    """
    Evalúa todas las fusiones posibles de pares de clústeres.
    Devuelve una lista de resultados:
       [ ((c1, c2), fused_labels, silhouette_mean, dispersion_dict, dispersion_sum), ... ]
    """
    clusters_unique = np.unique(labels)
    results = []
    
    for i in range(len(clusters_unique)):
        for j in range(i+1, len(clusters_unique)):
            c1 = clusters_unique[i]
            c2 = clusters_unique[j]
            fused_labels = merge_clusters(labels, c1, c2)
            new_unique = np.unique(fused_labels)
            if len(new_unique) == 1:
                continue
            s_mean = silhouette_mean(coords, fused_labels)
            disp_dict = compute_dispersion(coords, fused_labels)
            disp_sum = np.nansum(list(disp_dict.values()))
            results.append(((c1, c2), fused_labels, s_mean, disp_dict, disp_sum))
    
    return results

def get_worst_cluster(dispersion_dict):
    """
    Dado un diccionario {cluster_label: dispersion}, retorna (worst_cluster, max_disp).
    """
    worst_cluster = None
    max_disp = -1
    for c_label, d_val in dispersion_dict.items():
        if d_val > max_disp:
            max_disp = d_val
            worst_cluster = c_label
    return worst_cluster, max_disp

def kmeans_three_points(coords):
    """
    Aplica KMeans(n_clusters=3) con centroides iniciales = centro de masa + 2 puntos más alejados.
    Retorna (sub_labels, sub_silhouette, sub_dispersion_dict, sub_dispersion_sum).
    """
    center_of_mass = coords.mean(axis=0)
    distances = np.linalg.norm(coords - center_of_mass, axis=1)
    far_idxs = np.argsort(distances)[-2:]
    initial_centers = np.vstack([center_of_mass, coords[far_idxs[0]], coords[far_idxs[1]]])
    
    kmeans = KMeans(n_clusters=3, init=initial_centers, n_init=1, random_state=42)
    kmeans.fit(coords)
    
    sub_labels = kmeans.labels_
    sub_sil = np.mean(silhouette_samples(coords, sub_labels))
    sub_disp_dict = compute_dispersion(coords, sub_labels)
    sub_disp_sum = np.nansum(list(sub_disp_dict.values()))
    
    return sub_labels, sub_sil, sub_disp_dict, sub_disp_sum
def iterative_fusion_and_subdivision(coords, init_labels, threshold=1.5, max_iterations=10):
    """
    Proceso iterativo que:
      - Fusiona pares de clústeres (eligiendo la fusión que minimiza la dispersión total).
      - Identifica el clúster con mayor dispersión.
      - Si esa dispersión supera el umbral y el clúster tiene al menos 14 átomos, lo subdivide en 3 mediante kmeans.
    Devuelve las etiquetas finales actualizadas.
    """
    labels = np.copy(init_labels)
    iteration = 0
    while iteration < max_iterations:
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            print(f"\n[SALIDA] Sólo queda 1 clúster (iter={iteration}).")
            break
        
        print(f"\n--- Iteración {iteration}, clústeres presentes: {unique_labels} ---")
        merge_candidates = try_all_merges(coords, labels)
        if not merge_candidates:
            print("[SALIDA] No hay fusiones posibles.")
            break
        
        best_merge = min(merge_candidates, key=lambda x: x[4])
        (pair, fused_labels, fused_sil, fused_disp_dict, fused_disp_sum) = best_merge
        print(f"  Fusión elegida: {pair}, disp_total={fused_disp_sum:.3f}, sil={fused_sil:.3f}")
        labels = fused_labels
        
        worst_cluster, max_disp = get_worst_cluster(fused_disp_dict)
        print(f"  Clúster con mayor dispersión: {worst_cluster}, valor={max_disp:.3f}")
        
        # Extraer la máscara y el número de átomos en el clúster con mayor dispersión
        mask = (labels == worst_cluster)
        n_atoms = np.sum(mask)
        if max_disp > threshold:
            if n_atoms < 14:
                print(f"  > El clúster {worst_cluster} tiene solo {n_atoms} átomos (<14). No se subdivide.")
            else:
                print(f"  > Dispersión > {threshold} y {n_atoms} átomos. Subdividiendo clúster {worst_cluster} con k-means(3)...")
                coords_worst = coords[mask]
                sub_labels, sub_sil, sub_disp_dict, sub_disp_sum = kmeans_three_points(coords_worst)
                print(f"    Subdividido en subclústeres: {np.unique(sub_labels)}, disp_total={sub_disp_sum:.3f}")
                
                offset = worst_cluster * 10
                new_sub_labels = offset + sub_labels
                new_labels_global = np.copy(labels)
                new_labels_global[mask] = new_sub_labels
                labels = new_labels_global
        else:
            print(f"  > Dispersión <= {threshold}. No se subdivide nada.")
        
        iteration += 1
    
    print(f"\n[FIN] Iteraciones completadas = {iteration}. Clústeres finales: {np.unique(labels)}")
    return labels


# ================================================================
# Clase principal para procesar archivos y exportar el resultado
# ================================================================

class ClusterProcessor:
    def __init__(self, file_path: str, threshold: float = 1.5, max_iterations: int = 10):
        self.file_path = file_path
        
        
        self.threshold = threshold
        self.max_iterations = max_iterations
        
        # Extraer datos completos y encabezado del archivo dump
        self.matriz_total = UtilidadesClustering.extraer_datos_completos(file_path)
        self.header = UtilidadesClustering.extraer_encabezado(file_path)
        
        # Usar OVITO para obtener las coordenadas y los clusters iniciales
        pipeline = import_file(file_path)
        data = pipeline.compute()
        self.coords = data.particles.positions
        self.init_labels = data.particles["Cluster"].array

    def process_clusters(self):
        """
        Ejecuta el proceso iterativo de fusión y subdivisión y actualiza los clusters.
        Luego remapea las etiquetas finales a un rango consecutivo (0, 1, 2, ...).
        """
        self.final_labels = iterative_fusion_and_subdivision(self.coords, self.init_labels, self.threshold, self.max_iterations)
        print("\nClústeres finales en 'final_labels':", np.unique(self.final_labels))
        
        # Remapeo de etiquetas a consecutivos: 0, 1, 2, ...
        unique = np.unique(self.final_labels)
        mapping = {old: new for new, old in enumerate(unique)}
        self.final_labels = np.vectorize(mapping.get)(self.final_labels)
        print("Clústeres remapeados:", np.unique(self.final_labels))
        
        if self.matriz_total.shape[0] == self.final_labels.shape[0]:
            self.matriz_total[:, 5] = self.final_labels
        else:
            print("¡Atención! La cantidad de filas en la matriz de datos y los clusters no coincide.")

    def export_updated_file(self, output_file: str = None):
        """
        Exporta la matriz de datos actualizada a un archivo de texto 
        con el encabezado original.
        """
        if output_file is None:
            output_file = f"{self.file_path}"
        fmt = ("%d %d %.5f %.5f %.5f %d")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(self.header)
                np.savetxt(f, self.matriz_total, fmt=fmt, delimiter=" ")
            print("Datos exportados exitosamente a:", output_file)
        except Exception as e:
            print(f"Error al exportar {output_file}: {e}")

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    # 1) Cargar lista de archivos críticos desde JSON
    lista_criticos = UtilidadesClustering.cargar_lista_archivos_criticos("outputs.json/key_archivos.json")
    
    for archivo in lista_criticos:
        print("\n======================================")
        print(f"Procesando archivo: {archivo}")
        
        # Crear instancia de ClusterProcessor para el archivo actual
        processor = ClusterProcessor(archivo, threshold=1.5, max_iterations=3)
        
        # Ejecutar el proceso de clustering
        processor.process_clusters()
        
        # Exportar el archivo actualizado con los clusters remapeados
        processor.export_updated_file()
