import os
import json
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import ConstructSurfaceModifier
from input_params import CONFIG
from format_keys import KeyFilesSeparator
import json
import numpy as np
import json
import numpy as np
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from ovito.io import import_file

class SurfaceProcessor:
    def __init__(self, config=CONFIG[0], json_path="outputs.json/key_archivos.json", radi=None, threshold_file="outputs.vfinder/key_single_vacancy.json"):
        self.config = config
        self.smoothing_level = config["smoothing level"]
        self.radi = radi if radi is not None else [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.json_path = json_path
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.clusters_final = self.data.get("clusters_final", [])
        self.results_matrix = None
        with open(threshold_file, "r", encoding="utf-8") as f:
            threshold_data = json.load(f)
        self.min_area_threshold = threshold_data["surface_area"][0] / 2
        self.min_filled_volume_threshold = threshold_data["filled_volume"][0] / 22

    def process_surface_for_file(self, archivo):
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
                area = 0
            try:
                filled_volume = data.attributes['ConstructSurfaceMesh.filled_volume']
            except Exception as e:
                filled_volume = 0
            if area > best_area:
                best_area = area
                best_filled_volume = filled_volume
                best_radius = r
                best_pipeline = pipeline
        if best_area < self.min_area_threshold or best_filled_volume < self.min_filled_volume_threshold:
            return None, None, None, None, None
        return best_pipeline, best_radius, best_area, best_filled_volume, cluster_size

    def process_all_files(self):
        results = []
        for archivo in self.clusters_final:
            bp, br, ba, fv, num_atm = self.process_surface_for_file(archivo)
            if bp is not None:
                results.append([archivo, br, ba, fv, num_atm])
        self.results_matrix = np.array(results)
        return self.results_matrix

    def export_results(self, output_csv="resultados_procesados.csv"):
        if self.results_matrix is None:
            self.process_all_files()
        np.savetxt(output_csv, self.results_matrix, delimiter=",", fmt="%s",
                   header="archivo,mejor_radio,area,filled_volume,num_atm", comments="")

class ClusterDumpProcessor:
    def __init__(self, file_path: str, decimals: int = 5):
        self.file_path = file_path
        self.matriz_total = None
        self.header = None
        self.subset = None
        self.divisions_of_cluster = CONFIG[0]['divisions_of_cluster']

    def load_data(self):
        self.matriz_total = self.extraer_datos_completos(self.file_path)
        self.header = self.extraer_encabezado(self.file_path)
        if self.matriz_total.size == 0:
            raise ValueError(f"No se pudieron extraer datos de {self.file_path}")
        self.subset = self.matriz_total[:, 2:5]

    def calcular_dispersion(self, points: np.ndarray) -> float:
        if points.shape[0] == 0:
            return 0.0
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        return np.mean(distances)

    def process_clusters(self):
        centro_masa_global = np.mean(self.subset, axis=0)
        p1, p2, distancia_maxima = self.find_farthest_points(self.subset)
        dispersion = self.calcular_dispersion(self.subset)
        threshold = self.divisions_of_cluster
        etiquetas = self.aplicar_kmeans(self.subset, p1, p2, centro_masa_global, n_clusters=3)
        if etiquetas.shape[0] != self.matriz_total.shape[0]:
            raise ValueError("El número de etiquetas no coincide con la matriz total.")
        self.matriz_total[:, 5] = etiquetas

    def ejecutar_silhotte(self):
        lista_criticos = UtilidadesClustering.cargar_lista_archivos_criticos("outputs.json/key_archivos.json")
        processor = ClusterProcessor(self.archivo, decimals=4, threshold=1.2, max_iterations=10)
        processor.process_clusters()
        processor.export_updated_file()

    def separar_coordenadas_por_cluster(self) -> dict:
        if self.matriz_total is None:
            raise ValueError("Los datos no han sido cargados. Ejecuta load_data() primero.")
        clusters_dict = {0, 1, 2}
        etiquetas_unicas = np.unique(self.matriz_total[:, 5])
        for etiqueta in etiquetas_unicas:
            coords = self.matriz_total[self.matriz_total[:, 5] == etiqueta][:, 2:5]
            clusters_dict[int(etiqueta)] = coords
        return clusters_dict

    def export_updated_file(self, output_file: str = None):
        if output_file is None:
            output_file = f"{self.file_path}_actualizado.txt"
        fmt = ("%d %d %.5f %.5f %.5f %d")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(self.header)
                np.savetxt(f, self.matriz_total, fmt=fmt, delimiter=" ")
        except Exception as e:
            pass

    @staticmethod
    def cargar_lista_archivos_criticos(json_path: str) -> list:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            return datos.get("clusters_criticos", [])
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as e:
            return []

    @staticmethod
    def extraer_datos_completos(file_path: str) -> np.ndarray:
        datos = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return np.array([])
        start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("ITEM: ATOMS"):
                start_index = i + 1
                break
        if start_index is None:
            return np.array([])
        for line in lines[start_index:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                id_val = int(parts[0])
                type_val = int(parts[1])
                x = round(float(parts[2]), 5)
                y = round(float(parts[3]), 5)
                z = round(float(parts[4]), 5)
                cluster_val = int(parts[5])
                datos.append([id_val, type_val, x, y, z, cluster_val])
            except ValueError:
                continue
        return np.array(datos)

    @staticmethod
    def extraer_encabezado(file_path: str) -> list:
        encabezado = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    encabezado.append(line)
                    if line.strip().startswith("ITEM: ATOMS"):
                        break
        except Exception as e:
            pass
        return encabezado

    @staticmethod
    def aplicar_kmeans(coordenadas: np.ndarray, p1, p2, centro_masa_global, n_clusters: int) -> np.ndarray:
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

def merge_clusters(labels, c1, c2):
    new_labels = np.copy(labels)
    new_labels[new_labels == c2] = c1
    return new_labels

def compute_dispersion(coords, labels):
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
    sil_vals = silhouette_samples(coords, labels)
    return np.mean(sil_vals)

def try_all_merges(coords, labels):
    clusters_unique = np.unique(labels)
    results = []
    for i in range(len(clusters_unique)):
        for j in range(i + 1, len(clusters_unique)):
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
    worst_cluster = None
    max_disp = -1
    for c_label, d_val in dispersion_dict.items():
        if d_val > max_disp:
            max_disp = d_val
            worst_cluster = c_label
    return worst_cluster, max_disp

def kmeans_three_points(coords):
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

def iterative_fusion_and_subdivision(coords, init_labels, threshold=1.2, max_iterations=10):
    labels = np.copy(init_labels)
    iteration = 0
    while iteration < max_iterations:
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            break
        merge_candidates = try_all_merges(coords, labels)
        if not merge_candidates:
            break
        best_merge = min(merge_candidates, key=lambda x: x[4])
        (pair, fused_labels, fused_sil, fused_disp_dict, fused_disp_sum) = best_merge
        labels = fused_labels
        worst_cluster, max_disp = get_worst_cluster(fused_disp_dict)
        if max_disp > threshold:
            mask = (labels == worst_cluster)
            coords_worst = coords[mask]
            sub_labels, sub_sil, sub_disp_dict, sub_disp_sum = kmeans_three_points(coords_worst)
            offset = worst_cluster * 10
            new_sub_labels = offset + sub_labels
            new_labels_global = np.copy(labels)
            new_labels_global[mask] = new_sub_labels
            labels = new_labels_global
        iteration += 1
    return labels

class ClusterProcessor:
    def __init__(self, file_path: str, threshold: float = 1.2, max_iterations: int = 10):
        self.file_path = file_path
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.matriz_total = UtilidadesClustering.extraer_datos_completos(file_path)
        self.header = UtilidadesClustering.extraer_encabezado(file_path)
        pipeline = import_file(file_path)
        data = pipeline.compute()
        self.coords = data.particles.positions
        self.init_labels = data.particles["Cluster"].array

    def process_clusters(self):
        self.final_labels = iterative_fusion_and_subdivision(self.coords, self.init_labels, self.threshold, self.max_iterations)
        unique = np.unique(self.final_labels)
        mapping = {old: new for new, old in enumerate(unique)}
        self.final_labels = np.vectorize(mapping.get)(self.final_labels)
        if self.matriz_total.shape[0] == self.final_labels.shape[0]:
            self.matriz_total[:, 5] = self.final_labels

    def export_updated_file(self, output_file: str = None):
        if output_file is None:
            output_file = f"{self.file_path}"
        fmt = ("%d %d %.5f %.5f %.5f %d")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(self.header)
                np.savetxt(f, self.matriz_total, fmt=fmt, delimiter=" ")
        except Exception as e:
            pass
