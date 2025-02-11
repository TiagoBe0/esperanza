import os
import json
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import (ExpressionSelectionModifier,
                             DeleteSelectedModifier,
                             ConstructSurfaceModifier,
                             InvertSelectionModifier,
                             AffineTransformationModifier)
from input_params import CONFIG
from single_vacancy import SingleVacancyProcessor
class TrainingProcessor:
    def __init__(self, relax_file, radius_training, radius, smoothing_level_training,strees, output_dir="outputs.vfinder"):
        """
        Inicializa el procesador con la ruta del archivo original (relax_file) y parámetros de entrenamiento.
        
        Args:
            relax_file (str): Ruta al archivo dump original.
            radius_training (float): Radio para la selección en el training dump.
            radius (float): Radio utilizado en el ConstructSurfaceModifier.
            smoothing_level_training (float): Nivel de suavizado para el ConstructSurfaceModifier.
            output_dir (str): Directorio donde se exportarán los resultados.
        """
        self.relax_file = relax_file
        self.radius_training = radius_training
        self.radius = radius
        self.smoothing_level_training = smoothing_level_training
        self.output_dir = output_dir
        self.ids_dump_file = os.path.join(self.output_dir, "ids.training.dump")
        self.training_results_file = os.path.join(self.output_dir, "training_results.json")
        self.strees = strees
    
    @staticmethod
    def obtener_centro(file_path):
        """
        Abre el archivo dump y extrae la sección BOX BOUNDS para calcular el centro de la caja.
        
        Args:
            file_path (str): Ruta al archivo dump.
            
        Returns:
            tuple: (center_x, center_y, center_z)
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        box_bounds_index = None
        for i, line in enumerate(lines):
            if line.startswith("ITEM: BOX BOUNDS"):
                box_bounds_index = i
                break
        if box_bounds_index is None:
            raise ValueError("No se encontró la sección 'BOX BOUNDS' en el archivo.")
        x_bounds = lines[box_bounds_index + 1].split()
        y_bounds = lines[box_bounds_index + 2].split()
        z_bounds = lines[box_bounds_index + 3].split()
        x_min, x_max = map(float, x_bounds)
        y_min, y_max = map(float, y_bounds)
        z_min, z_max = map(float, z_bounds)
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        center_z = (z_min + z_max) / 2.0
        return center_x, center_y, center_z
    
    def export_training_dump(self):
        """
        Exporta un dump de entrenamiento a partir del archivo original.
        Calcula el centro de la caja, define una condición (una esfera de radio radius_training)
        e invierte la selección para eliminar las partículas dentro de esa esfera.
        Se verifica que exista el directorio de salida; si no existe, se crea.
        """
        output_dir = os.path.dirname(self.ids_dump_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        centro = TrainingProcessor.obtener_centro(self.relax_file)
        print("Centro de la caja:", centro)
        pipeline = import_file(self.relax_file)
        condition = (
            f"(Position.X - {centro[0]})*(Position.X - {centro[0]}) + "
            f"(Position.Y - {centro[1]})*(Position.Y - {centro[1]}) + "
            f"(Position.Z - {centro[2]})*(Position.Z - {centro[2]}) <= {self.radius_training * self.radius_training}"
        )
        pipeline.modifiers.append(ExpressionSelectionModifier(expression=condition))
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        try:
            export_file(pipeline, self.ids_dump_file, "lammps/dump", 
                        columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"])
            pipeline.modifiers.clear()
            print(f"Dump exportado correctamente a {self.ids_dump_file}")
        except Exception as e:
            print(f"Error al exportar el dump de áreas clave: {e}")
    
    def extract_particle_ids(self):
        """
        Importa el dump exportado y extrae la propiedad "Particle Identifier" convirtiéndola en lista.
        
        Returns:
            list: Lista de IDs de partículas.
        """
        pipeline = import_file(self.ids_dump_file)
        data = pipeline.compute()
        particle_ids = data.particles["Particle Identifier"]
        return np.array(particle_ids).tolist()
    
    @staticmethod
    def crear_condicion_ids(ids_eliminar):
        """
        A partir de una lista de IDs, crea una cadena de condición con la forma:
        ParticleIdentifier==id1 || ParticleIdentifier==id2 || ...
        
        Args:
            ids_eliminar (list): Lista de IDs a eliminar.
            
        Returns:
            str: Condición para usar en el ExpressionSelectionModifier.
        """
        return " || ".join([f"ParticleIdentifier=={id}" for id in ids_eliminar])
    
    def compute_max_distance(self, data):
        """
        Calcula la mayor distancia entre el centro de masa y las partículas.
        
        Args:
            data: Objeto DataCollection de OVITO.
            
        Returns:
            float: Máxima distancia.
        """
        positions = data.particles.positions
        center_of_mass = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center_of_mass, axis=1)
        return np.max(distances)
    
    def compute_min_distance(self, data):
        """
        Calcula la menor distancia entre el centro de masa y las partículas.
        
        Args:
            data: Objeto DataCollection de OVITO.
            
        Returns:
            float: Mínima distancia.
        """
        positions = data.particles.positions
        center_of_mass = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center_of_mass, axis=1)
        return np.min(distances)
    
    def run_training(self):
        """
        Ejecuta el proceso completo:
          1. Exporta el dump de entrenamiento.
          2. Extrae los IDs de las partículas.
          3. Iterativamente, para cada subconjunto de IDs, aplica un pipeline que elimina
             dichas partículas, calcula la malla superficial y obtiene distancias al centro de masa.
          4. Exporta los resultados en un archivo JSON.
        """
        # Paso 1: Exportar el dump de entrenamiento.
        self.export_training_dump()
        
        # Paso 2: Extraer IDs de partículas.
        particle_ids_list = self.extract_particle_ids()
        print("IDs extraídos:", particle_ids_list)
        
        # Paso 3: Procesamiento incremental.
        pipeline_2 = import_file(self.relax_file)
        
        pipeline_2.modifiers.append(AffineTransformationModifier(
            operate_on={'particles','cell'},
            transformation=[[self.strees[0], 0, 0, 0],
                            [0, self.strees[1], 0, 0],
                            [0, 0, self.strees[2], 0]]
        ))
        sm_mesh_training = []
        vacancias = []
        vecinos = []
        max_distancias = []
        min_distancias = []
        
        for index in range(len(particle_ids_list)):
            ids_a_eliminar = particle_ids_list[:index+1]
            condition_f = TrainingProcessor.crear_condicion_ids(ids_a_eliminar)
            pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=condition_f))
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            pipeline_2.modifiers.append(ConstructSurfaceModifier(
                radius=self.radius,
                smoothing_level=self.smoothing_level_training,
                identify_regions=True,
                select_surface_particles=True
            ))
            data_2 = pipeline_2.compute()
            sm_elip = data_2.attributes.get('ConstructSurfaceMesh.surface_area', 0)
            filled_vol = data_2.attributes.get('ConstructSurfaceMesh.void_volume', 0)
            sm_mesh_training.append(sm_elip)
            vacancias.append(index+1)
            
            pipeline_2.modifiers.append(InvertSelectionModifier())
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            data_3 = pipeline_2.compute()
            max_dist = self.compute_max_distance(data_3)
            min_dist = self.compute_min_distance(data_3)
            max_distancias.append(filled_vol)
            min_distancias.append(min_dist)
            vecinos.append(data_3.particles.count)
            
            pipeline_2.modifiers.clear()
        
        datos_exportar = {
            "sm_mesh_training": sm_mesh_training,
            "filled_volume": max_distancias,
            "vacancias": vacancias,
            "vecinos": vecinos
        }
        output_dir = os.path.dirname(self.training_results_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(self.training_results_file, "w") as f:
            json.dump(datos_exportar, f, indent=4)
        print(f"Resultados exportados a '{self.training_results_file}'.")
    
    def run(self):
        """Método para ejecutar la cadena completa del proceso de entrenamiento."""
        self.run_training()

# -------------------------------------------------------------------
# Ejecución principal
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Se extraen parámetros de configuración desde CONFIG.
    config = CONFIG[0]
    relax = config['relax']                          # Archivo dump original.
    radius_training = config['radius_training']      # Radio de entrenamiento.
    radius = config['radius']                        # Radio para la malla superficial.
    smoothing_level_training = config['smoothing_level_training']  # Nivel de suavizado para training.
    
    processor = TrainingProcessor(relax, radius_training, radius, smoothing_level_training)
    processor.run()
    
