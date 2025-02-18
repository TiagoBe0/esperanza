import os
import json
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, ClusterAnalysisModifier, ConstructSurfaceModifier, InvertSelectionModifier
from input_params import CONFIG,PREDICTOR_COLUMNS
from format_keys import ExportClusterList, KeyFilesSeparator
from cluster_descriptor import SurfaceProcessor, ClusterDumpProcessor
from main_vacancy_analysis import TrainingProcessor
from single_vacancy import SingleVacancyProcessor, DiVacancyProcessor
from RF_train_metodh import VacancyPredictorRF, XGBoostVacancyPredictor, VacancyPredictor
import math
from sklearn.metrics import silhouette_samples
from adicional_train import VacancyPredictorCurve
from sklearn.cluster import KMeans
import pandas as pd
import xgboost as xgb
import logging
from generador_de_datos import SyntheticDataGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="log.txt", filemode="w")
logger = logging.getLogger(__name__)
logger.info("Inicio del programa.")

class ClusterProcessor:
    def __init__(self):
        self.configuracion = CONFIG[0]
        self.nombre_archivo = self.configuracion['defect']
        self.radio_sonda = self.configuracion['radius']
        self.smoothing_leveled = self.configuracion['smoothing level']
        self.cutoff_radius = self.configuracion['cutoff radius']
        self.outputs_dump = "outputs.dump"
        self.outputs_json = "outputs.json"
        os.makedirs(self.outputs_dump, exist_ok=True)
        os.makedirs(self.outputs_json, exist_ok=True)
    
    def run(self):
        pipeline = import_file(self.nombre_archivo)
        pipeline.modifiers.append(ConstructSurfaceModifier(radius=self.radio_sonda, smoothing_level=self.smoothing_leveled, identify_regions=True, select_surface_particles=True))
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=self.cutoff_radius, sort_by_size=True, unwrap_particles=True, compute_com=True))
        data = pipeline.compute()
        num_clusters = data.attributes["ClusterAnalysis.cluster_count"]
        datos_clusters = {"num_clusters": num_clusters}
        clusters_json_path = os.path.join(self.outputs_json, "clusters.json")
        with open(clusters_json_path, "w") as archivo:
            json.dump(datos_clusters, archivo, indent=4)
        key_areas_dump_path = os.path.join(self.outputs_dump, "key_areas.dump")
        try:
            export_file(pipeline, key_areas_dump_path, "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
            pipeline.modifiers.clear()
        except Exception as e:
            pass
        clusters = [f"Cluster=={i}" for i in range(1, num_clusters + 1)]
        for i, cluster_expr in enumerate(clusters, start=1):
            pipeline_2 = import_file(key_areas_dump_path)
            pipeline_2.modifiers.append(ClusterAnalysisModifier(cutoff=self.cutoff_radius, cluster_coloring=True, unwrap_particles=True, sort_by_size=True))
            pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=cluster_expr))
            pipeline_2.modifiers.append(InvertSelectionModifier())
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            output_file = os.path.join(self.outputs_dump, f"key_area_{i}.dump")
            try:
                export_file(pipeline_2, output_file, "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
                pipeline_2.modifiers.clear()
            except Exception as e:
                pass
        print(f"Número de áreas clave encontradas: {num_clusters}")
    
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
import json
import math
import os
import numpy as np
import pandas as pd
import xgboost as xgb

from input_params import PREDICTOR_COLUMNS

class VacancyPredictionRunner:
    def __init__(self, archivo, 
                 predictor_small, predictor_large, 
                 predictor_rf_small, predictor_rf_large, 
                 predictor_xgb_small, predictor_xgb_large, 
                 other_method=False):
     
        self.archivo = archivo

        with open("outputs.vfinder/key_single_vacancy.json", "r") as f:
            single_vac = json.load(f)
        self.ref_area = single_vac["surface_area"][0]
        self.ref_filled_volume = single_vac["filled_volume"][0]
        self.ref_vecinos = single_vac["vecinos"][0]

        with open("outputs.vfinder/key_divacancy.json", "r") as f:
            diva_vac = json.load(f)
        self.ref_area_diva = diva_vac["surface_area"][0]
        self.ref_filled_volume_diva = diva_vac["filled_volume"][0]
        self.ref_vecinos_diva = diva_vac["vecinos"][0]

        self.predictor_small = predictor_small      # Regresión lineal con pocos datos
        self.predictor_large = predictor_large      # Regresión lineal con más datos
        self.predictor_rf_small = predictor_rf_small    # Random Forest con pocos datos
        self.predictor_rf_large = predictor_rf_large    # Random Forest con más datos
        self.predictor_xgb_small = predictor_xgb_small  # XGBoost con pocos datos
        self.predictor_xgb_large = predictor_xgb_large  # XGBoost con más datos

        self.other_method = other_method

        self.vector_area = None
        self.vector_filled_volume = None
        self.vector_num_atm = None
        self.vector_true = None
        self.results = {}
        self.COLUMNS_TRAINING = PREDICTOR_COLUMNS

    def load_data(self):
        df = pd.read_csv("resultados_procesados.csv")
        ultimas_tres = df.iloc[:, -3:]
        self.vector_area = ultimas_tres.iloc[:, 0].values
        self.vector_filled_volume = ultimas_tres.iloc[:, 1].values
        self.vector_num_atm = ultimas_tres.iloc[:, 2].values
        if "vacancias" in df.columns:
            self.vector_true = df["vacancias"].values
        else:
            print("No se encontró la columna 'vacancies'. No se calcularán errores cuadráticos.")
            self.vector_true = None

    def predict_linear(self):
        print("Columnas activas para el predictor lineal (regresión):", self.predictor_small.columns)
        total_count = 0
        predictions = []
        errors = []
        threshold = 5 * self.ref_area  
        for i, (area, filled_volume, num_atm) in enumerate(zip(self.vector_area, 
                                                                self.vector_filled_volume, 
                                                                self.vector_num_atm)):
            # Comparación para forzar predicción
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or 
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or 
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"Iteración {i}: Valores similares a vacancia única, predicción forzada a 1")
            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.2) or 
                  math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.2) or 
                  (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"Iteración {i}: Valores similares a divacancia, predicción forzada a 2")
            else:
                features = {}
                if "sm_mesh_training" in self.predictor_small.columns:
                    features["sm_mesh_training"] = area
                if "filled_volume" in self.predictor_small.columns:
                    features["filled_volume"] = filled_volume
                if "vecinos" in self.predictor_small.columns:
                    features["vecinos"] = num_atm

                if area < threshold:
                    vacancias_pred = self.predictor_small.predict_vacancies(**features)
                    print(f"Iteración {i}: [Linear Small] Características {features} -> Vacancias Predichas = {abs(vacancias_pred)}")
                else:
                    vacancias_pred = self.predictor_large.predict_vacancies(**features)
                    print(f"Iteración {i}: [Linear Large] Características {features} -> Vacancias Predichas = {abs(vacancias_pred)}")
                total_count += vacancias_pred

            predictions.append(vacancias_pred)
            if self.vector_true is not None:
                error = (vacancias_pred - self.vector_true[i])**2
                errors.append(error)
            else:
                errors.append(None)
        print(f"Predicción lineal regresión: {abs(total_count)}")
        if self.vector_true is not None:
            mse = sum(errors) / len(errors)
            print(f"Error cuadrático medio (Linear): {mse}")
        return total_count, predictions, errors

    def predict_rf(self):
        if not self.other_method or self.predictor_rf_small is None or self.predictor_rf_large is None:
            print("El método RF no está habilitado o no se proporcionaron ambos objetos predictor_rf.")
            return None, [], []
        total_count = 0
        predictions = []
        errors = []
        threshold = 5 * self.ref_area
        for i, (area, filled_volume, num_atm) in enumerate(zip(self.vector_area, 
                                                                self.vector_filled_volume, 
                                                                self.vector_num_atm)):
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or 
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or 
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"Iteración {i}: [RF] Valores similares a vacancia única, predicción forzada a 1")
            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.2) or 
                  math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.2) or 
                  (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"Iteración {i}: [RF] Valores similares a divacancia, predicción forzada a 2")
            else:
                features = {}
                if "sm_mesh_training" in self.predictor_rf_small.columns:
                    features["sm_mesh_training"] = area
                if "filled_volume" in self.predictor_rf_small.columns:
                    features["filled_volume"] = filled_volume
                if "vecinos" in self.predictor_rf_small.columns:
                    features["vecinos"] = num_atm

                if area < threshold:
                    vacancias_pred = self.predictor_rf_small.predict_vacancies(**features)
                    print(f"Iteración {i}: [RF Small] Características {features} -> Vacancias Predichas = {vacancias_pred}")
                else:
                    vacancias_pred = self.predictor_rf_large.predict_vacancies(**features)
                    print(f"Iteración {i}: [RF Large] Características {features} -> Vacancias Predichas = {vacancias_pred}")
                total_count += vacancias_pred

            predictions.append(vacancias_pred)
            if self.vector_true is not None:
                error = (vacancias_pred - self.vector_true[i])**2
                errors.append(error)
            else:
                errors.append(None)
        print("\nContador total Random Forest:", total_count)
        if self.vector_true is not None and errors:
            mse = sum(errors) / len(errors)
            print(f"Error cuadrático medio (RF): {mse}")
        return total_count, predictions, errors

    def predict_xgb(self):
        total_count = 0
        predictions = []
        errors = []
        threshold = 5 * self.ref_area
        for i, (area, filled_volume, num_atm) in enumerate(zip(self.vector_area, 
                                                                self.vector_filled_volume, 
                                                                self.vector_num_atm)):
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or 
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or 
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"Iteración {i}: [XGBoost] Valores similares a vacancia única, predicción forzada a 1")
            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.2) or 
                  math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.2) or 
                  (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"Iteración {i}: [XGBoost] Valores similares a divacancia, predicción forzada a 2")
            else:
                features = {}
                if "sm_mesh_training" in self.predictor_xgb_small.columns:
                    features["sm_mesh_training"] = area
                if "filled_volume" in self.predictor_xgb_small.columns:
                    features["filled_volume"] = filled_volume
                if "vecinos" in self.predictor_xgb_small.columns:
                    features["vecinos"] = num_atm

                # Crear el array de entrada según el orden de columnas
                if area < threshold:
                    sample_input = np.array([[features[col] for col in self.predictor_xgb_small.columns]])
                    vacancias_pred = self.predictor_xgb_small.predict(sample_input)[0]
                    print(f"Iteración {i}: [XGBoost Small] Características {features} -> Vacancias Predichas = {vacancias_pred}")
                else:
                    sample_input = np.array([[features[col] for col in self.predictor_xgb_large.columns]])
                    vacancias_pred = self.predictor_xgb_large.predict(sample_input)[0]
                    print(f"Iteración {i}: [XGBoost Large] Características {features} -> Vacancias Predichas = {vacancias_pred}")
                total_count += vacancias_pred

            predictions.append(vacancias_pred)
            if self.vector_true is not None:
                error = (vacancias_pred - self.vector_true[i])**2
                errors.append(error)
            else:
                errors.append(None)
        print("Contador total XGBoost:", total_count)
        if self.vector_true is not None and errors:
            mse = sum(errors) / len(errors)
            print(f"Error cuadrático medio (XGBoost): {mse}")
        return total_count, predictions, errors

    def export_results(self):
        dfs = []
        for method, data in self.results.items():
            num_rows = len(data["predictions"])
            df_method = pd.DataFrame({
                "iteration": list(range(num_rows)),
                "method": [method] * num_rows,
                "prediction": data["predictions"],
                "true_value": self.vector_true if self.vector_true is not None else [None] * num_rows,
                "squared_error": data["errors"]
            })
            dfs.append(df_method)
        if dfs:
            final_df = pd.concat(dfs, ignore_index=True)
            base_name = os.path.basename(self.archivo)
            output_dir = "outputs.vfinder"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{base_name}.csv")
            final_df.to_csv(output_file, index=False)
        else:
            print("No hay resultados para exportar.")

    def export_totals(self):
        totals_dict = {method: data["total"] for method, data in self.results.items()}
        df_totals = pd.DataFrame(list(totals_dict.items()), columns=["modelo", "contador_total"])
        output_dir = "outputs.vfinder"
        base_name = os.path.basename(self.archivo)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{base_name}_totales_vacancias.csv")
        df_totals.to_csv(output_file, index=False)
        print(f"Contadores totales exportados a '{output_file}'")

    def run(self):
        self.load_data()
        self.results = {}
        print("Ejecutando predicción con regresión lineal:")
        total_linear, predictions_linear, errors_linear = self.predict_linear()
        self.results["linear"] = {"total": total_linear, "predictions": predictions_linear, "errors": errors_linear}

        if self.other_method:
            print("Ejecutando predicción con random forest:")
            total_rf, predictions_rf, errors_rf = self.predict_rf()
            self.results["rf"] = {"total": total_rf, "predictions": predictions_rf, "errors": errors_rf}

        print("Ejecutando predicción con XGBoost:")
        total_xgb, predictions_xgb, errors_xgb = self.predict_xgb()
        self.results["xgb"] = {"total": total_xgb, "predictions": predictions_xgb, "errors": errors_xgb}

        self.export_results()
        self.export_totals()

class UtilidadesClustering:
    @staticmethod
    def cargar_lista_archivos_criticos(json_path: str) -> list:
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
    def cargar_min_atoms(json_path: str) -> int:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            vecinos = datos.get("vecinos", [])
            if vecinos and isinstance(vecinos, list):
                return int(vecinos[0])
            else:
                print("No se encontró el valor de 'vecinos' en el archivo JSON, se usa valor por defecto 14.")
                return 14
        except Exception as e:
            print(f"Error al cargar {json_path}: {e}. Se usa valor por defecto 14.")
            return 14

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

def kmeans_default(coords):
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(coords)
    sub_labels = kmeans.labels_
    sub_sil = np.mean(silhouette_samples(coords, sub_labels))
    sub_disp_dict = compute_dispersion(coords, sub_labels)
    sub_disp_sum = np.nansum(list(sub_disp_dict.values()))
    return sub_labels, sub_sil, sub_disp_dict, sub_disp_sum

def iterative_fusion_and_subdivision(coords, init_labels, threshold=1.5, max_iterations=10, min_atoms=14):
    labels = np.copy(init_labels)
    iteration = 0
    while iteration < max_iterations:
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            #print(f"\n[SALIDA] Sólo queda 1 clúster (iter={iteration}).")
            break
       #print(f"\n--- Iteración {iteration}, clústeres presentes: {unique_labels} ---")
        merge_candidates = try_all_merges(coords, labels)
        if not merge_candidates:
            #print("[SALIDA] No hay fusiones posibles.")
            break
        best_merge = min(merge_candidates, key=lambda x: x[4])
        (pair, fused_labels, fused_sil, fused_disp_dict, fused_disp_sum) = best_merge
        unique_after_merge = np.unique(fused_labels)
        if len(unique_after_merge) == 2:
            valid_merge = True
            for c in unique_after_merge:
                n_atoms = np.sum(fused_labels == c)
                if n_atoms < min_atoms:
                    valid_merge = False
                    #print(f"  > La fusión {pair} produce el clúster {c} con solo {n_atoms} átomos (<{min_atoms}).")
                    break
            if not valid_merge:
                #print("  > Se revierte la fusión y no se actualizan los clusters en esta iteración.")
                iteration += 1
                continue
        #print(f"  Fusión elegida: {pair}, disp_total={fused_disp_sum:.3f}, sil={fused_sil:.3f}")
        labels = fused_labels
        worst_cluster, max_disp = get_worst_cluster(fused_disp_dict)
        #print(f"  Clúster con mayor dispersión: {worst_cluster}, valor={max_disp:.3f}")
        mask = (labels == worst_cluster)
        n_atoms = np.sum(mask)
        if max_disp > threshold:
            if n_atoms < min_atoms:
                print(f"  > El clúster {worst_cluster} tiene solo {n_atoms} átomos (<{min_atoms}). No se subdivide.")
            else:
                #print(f"  > Dispersión > {threshold} y {n_atoms} átomos. Subdividiendo clúster {worst_cluster} con KMeans(3)...")
                coords_worst = coords[mask]
                sub_labels, sub_sil, sub_disp_dict, sub_disp_sum = kmeans_three_points(coords_worst)
                #print(f"    Subdividido en sub-clústeres: {np.unique(sub_labels)}, disp_total={sub_disp_sum:.3f}")
                offset = worst_cluster * 10
                new_sub_labels = offset + sub_labels
                new_labels_global = np.copy(labels)
                new_labels_global[mask] = new_sub_labels
                labels = new_labels_global
        else:
            print(f"  > Dispersión <= {threshold}. No se subdivide nada.")
        iteration += 1
    #print(f"\n[FIN] Iteraciones completadas = {iteration}. Clústeres finales: {np.unique(labels)}")
    return labels

class ClusterProcessorMachine:
    def __init__(self, file_path: str, threshold: float = 1.2, max_iterations: int = 10, min_atoms: int = None):
        self.file_path = file_path
        self.threshold = threshold
        self.max_iterations = max_iterations
        if min_atoms is None:
            self.min_atoms = UtilidadesClustering.cargar_min_atoms("outputs.vfinder/key_single_vacancy.json")
        else:
            self.min_atoms = min_atoms
        self.matriz_total = UtilidadesClustering.extraer_datos_completos(file_path)
        self.header = UtilidadesClustering.extraer_encabezado(file_path)
        pipeline = import_file(file_path)
        data = pipeline.compute()
        self.coords = data.particles.positions
        self.init_labels = data.particles["Cluster"].array
    
    def process_clusters(self):
        self.final_labels = iterative_fusion_and_subdivision(self.coords, self.init_labels, self.threshold, self.max_iterations, self.min_atoms)
        #print("\nClústeres finales en 'final_labels':", np.unique(self.final_labels))
        unique = np.unique(self.final_labels)
        mapping = {old: new for new, old in enumerate(unique)}
        self.final_labels = np.vectorize(mapping.get)(self.final_labels)
        #print("Clústeres remapeados:", np.unique(self.final_labels))
        if self.matriz_total.shape[0] == self.final_labels.shape[0]:
            self.matriz_total[:, 5] = self.final_labels
        else:
            print("¡Atención! La cantidad de filas en la matriz de datos y los clusters no coincide.")
    
    def export_updated_file(self, output_file: str = None):
        if output_file is None:
            output_file = f"{self.file_path}"
        fmt = ("%d %d %.5f %.5f %.5f %d")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(self.header)
                np.savetxt(f, self.matriz_total, fmt=fmt, delimiter=" ")
            #print("Datos exportados exitosamente a:", output_file)
        except Exception as e:
            print(f"Error al exportar {output_file}: {e}")

if __name__ == "__main__":
    config = CONFIG[0]
    relax = config['relax']
    radius_training = config['radius_training']
    radius = config['radius']
    smoothing_level_training = config['smoothing_level_training']
    other_method = config['other method']
    processor_2 = TrainingProcessor(relax, radius_training, radius, smoothing_level_training, config['strees'])
    processor_2.run()
    processor_3 = SingleVacancyProcessor(config)
    processor_3.run()
    processor_4 = DiVacancyProcessor(config)
    processor_4.run()
    processor = ClusterProcessor()
    processor.run()
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
    lista_criticos = UtilidadesClustering.cargar_lista_archivos_criticos("outputs.json/key_archivos.json")
    for archivo in lista_criticos:
        #print("\n======================================")
        #print(f"Procesando archivo: {archivo}")
        processor = ClusterProcessorMachine(archivo, threshold=config['cluster tolerance'], max_iterations=config['iteraciones_clusterig'])
        processor.process_clusters()
        processor.export_updated_file()
    config = CONFIG[0]
    separator = KeyFilesSeparator(config, os.path.join("outputs.json", "clusters.json"))
    separator.run()
    processor_0 = ExportClusterList("outputs.json/key_archivos.json")
    processor_0.process_files()
    processor_1 = SurfaceProcessor()
    processor_1.process_all_files()
    processor_1.export_results()


    ########generacion de datos sinteticos
    # Ruta al archivo JSON que contiene los datos originales
    if config['generic_data']:
        
        json_file_path = "outputs.vfinder/training_results.json"  

        with open(json_file_path, "r") as f:
            data_original = json.load(f)
        
        generator = SyntheticDataGenerator(data_original, num_points=100, interpolation_kind='linear')
        synthetic_data = generator.generate()
        
        output_json_path = "outputs.vfinder/training_results.json"
        generator.export_to_json(output_json_path, synthetic_data)


    predictor_small = VacancyPredictor("outputs.vfinder/training_results.json")
    predictor_large = VacancyPredictor("outputs.vfinder/training_results.json")

    predictor_rf_small = VacancyPredictorRF("outputs.vfinder/defect_data.json")
    predictor_rf_large = VacancyPredictorRF("outputs.vfinder/training_results.json")

    predictor_xgb_small = XGBoostVacancyPredictor("outputs.vfinder/defect_data.json")
    predictor_xgb_large = XGBoostVacancyPredictor("outputs.vfinder/training_results.json")

    runner = VacancyPredictionRunner(
        archivo=config['defect'],
        predictor_small=predictor_small,
        predictor_large=predictor_large,
        predictor_rf_small=predictor_rf_small,
        predictor_rf_large=predictor_rf_large,
        predictor_xgb_small=predictor_xgb_small,
        predictor_xgb_large=predictor_xgb_large,
        other_method=True
    )
    runner.run()



    predictor = VacancyPredictorCurve("outputs.vfinder/training_results.json", "resultados_procesados.csv", degree=3)
    
    predictor.load_training_data(as_dataframe=True)
    predictor.prepare_training_data()
    
    poly = predictor.fit_curve()
    print("Coeficientes del polinomio:", poly.coeffs)
    print("Rango de área entrenamiento:", predictor.min_area_train, "a", predictor.max_area_train)
    
    
    csv_with_predictions = predictor.predict_from_csv()
    print(csv_with_predictions[["area", "predicted_vacancies"]])


    if config['chat']:
        from chat import CSVInterpreter

        interpreter = CSVInterpreter(f"outputs.vfinder/{config['defect']}.csv")
        resultado = interpreter.interpret(prompt="Interpreta los siguientes datos:")
        print(resultado)