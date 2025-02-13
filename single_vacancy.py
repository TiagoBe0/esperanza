import ovito
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier,AffineTransformationModifier,InvertSelectionModifier,DeleteSelectedModifier, ConstructSurfaceModifier, DeleteSelectedModifier
import json
import random
import numpy as np
from input_params import CONFIG as LAYERS
import os

class SingleVacancyProcessor:
    def __init__(self, layer):
        # Se utiliza el primer elemento de LAYERS para inicializar parámetros.
        primer_elemento = LAYERS[0]
        self.layer = primer_elemento
        self.relax = primer_elemento['relax']
        self.cutoff_radius = primer_elemento['cutoff radius']
        self.radius = primer_elemento['radius']
        self.smoothing_level = primer_elemento['smoothing level']
        self.strees = primer_elemento['strees']
        self.pipeline = None
        self.sms_sv = []
        self.nb_sv = []
        self.fl_vol = []

    def run(self):
        # Importar el archivo de relajación.
        self.pipeline = import_file(self.relax)
        data = self.pipeline.compute()
        particle_ids = data.particles["Particle Identifier"]
        ids = np.array(particle_ids).tolist()
        if not ids:
            print("No se encontraron IDs en el archivo.")
            return
        
        # Seleccionar un ID aleatorio.
        id_aleatorio = int(random.choice(ids))
        print("Partícula seleccionada para eliminar:", id_aleatorio)
        self.pipeline.modifiers.append(AffineTransformationModifier(
            operate_on={'particles','cell'},
            transformation=[[self.strees[0], 0, 0, 0],
                            [0, self.strees[1], 0, 0],
                            [0, 0, self.strees[2], 0]]
        ))
        # Seleccionar la partícula aleatoria
        self.pipeline.modifiers.append(ExpressionSelectionModifier(expression=f'ParticleIdentifier=={id_aleatorio}'))
        # Eliminar la partícula seleccionada (sin invertir la selección)
        self.pipeline.modifiers.append(DeleteSelectedModifier())
        # Opcional: Recompute para que se actualice la estructura sin la partícula eliminada.
        data_after_deletion = self.pipeline.compute()
        vecinos = data_after_deletion.particles.count

        # Aplicar el ConstructSurfaceModifier para obtener las métricas del vacío en la estructura modificada.
        self.pipeline.modifiers.append(ConstructSurfaceModifier(
            radius=self.radius,
            smoothing_level=self.smoothing_level,
            identify_regions=True,
            select_surface_particles=True
        ))
        data_with_surface = self.pipeline.compute()
        filled_volume = data_with_surface.attributes['ConstructSurfaceMesh.void_volume']
        surface_area = data_with_surface.attributes['ConstructSurfaceMesh.surface_area']

        # Guardar los resultados
        self.sms_sv.append(surface_area)
        self.pipeline.modifiers.append(InvertSelectionModifier())
        self.pipeline.modifiers.append(DeleteSelectedModifier())
        self.nb_sv.append(self.pipeline.compute().particles.count)
        self.fl_vol.append(filled_volume)

        # Exportar el archivo procesado.
        try:
            export_file(self.pipeline, "outputs.dump/single_vacancy_training.dump", "lammps/dump", 
                        columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"
                            ])
            self.pipeline.modifiers.clear()
        except Exception as e:
            print(f"Error al exportar el archivo: {e}")
        
        # Crear el diccionario con los datos y exportarlo a JSON.
        datos = {'surface_area': self.sms_sv, 'filled_volume': self.fl_vol, 'vecinos': self.nb_sv}
        output_path = 'outputs.vfinder/key_single_vacancy.json'
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, 'w') as f:
            json.dump(datos, f, indent=4)
        print("Procesamiento completado. Archivo generado en:", output_path)





class DiVacancyProcessor:
    def __init__(self, layer):
        # Se utiliza el primer elemento de LAYERS para inicializar parámetros.
        primer_elemento = LAYERS[0]
        self.layer = primer_elemento
        self.relax = primer_elemento['relax']
        self.cutoff_radius = primer_elemento['cutoff radius']
        self.radius = primer_elemento['radius']
        self.smoothing_level = primer_elemento['smoothing level']
        self.strees = primer_elemento['strees']
        self.pipeline = None
        self.sms_sv = []
        self.nb_sv = []
        self.fl_vol = []

    def run(self):
        # Importar el archivo de relajación y obtener la configuración inicial
        self.pipeline = import_file(self.relax)
        data = self.pipeline.compute()

        # Extraer los identificadores y las posiciones de las partículas
        particle_ids = np.array(data.particles["Particle Identifier"])
        if particle_ids.size == 0:
            print("No se encontraron IDs en el archivo.")
            return

        posX = np.array(data.particles["Position.X"])
        posY = np.array(data.particles["Position.Y"])
        posZ = np.array(data.particles["Position.Z"])
        positions = np.column_stack((posX, posY, posZ))

        # Seleccionar un índice aleatorio y obtener su ID y posición
        idx_random = random.choice(range(len(particle_ids)))
        id_random = int(particle_ids[idx_random])
        position_random = positions[idx_random]
        print("Partícula seleccionada para eliminar:", id_random)

        # Calcular las distancias entre la partícula aleatoria y todas las demás
        distances = np.linalg.norm(positions - position_random, axis=1)
        distances[idx_random] = np.inf  # Evitar la autoselección

        # Encontrar la partícula más cercana (vecina)
        idx_nn = np.argmin(distances)
        id_nn = int(particle_ids[idx_nn])
        print("Partícula vecina más cercana a eliminar:", id_nn)
        self.pipeline.modifiers.append(AffineTransformationModifier(
            operate_on={'particles','cell'},
            transformation=[[self.strees[0], 0, 0, 0],
                            [0, self.strees[1], 0, 0],
                            [0, 0, self.strees[2], 0]]
        ))
        # Seleccionar y eliminar ambas partículas (aleatoria y su vecina)
        self.pipeline.modifiers.append(ExpressionSelectionModifier(
            expression=f'ParticleIdentifier=={id_random} || ParticleIdentifier=={id_nn}'
        ))
        self.pipeline.modifiers.append(DeleteSelectedModifier())
        data_after_deletion = self.pipeline.compute()
        vecinos = data_after_deletion.particles.count

        # Aplicar el ConstructSurfaceModifier para obtener las métricas del vacío
        self.pipeline.modifiers.append(ConstructSurfaceModifier(
            radius=self.radius,
            smoothing_level=self.smoothing_level,
            identify_regions=True,
            select_surface_particles=True
        ))
        data_with_surface = self.pipeline.compute()
        filled_volume = data_with_surface.attributes['ConstructSurfaceMesh.void_volume']
        surface_area = data_with_surface.attributes['ConstructSurfaceMesh.surface_area']

        # Guardar los resultados en listas
        self.sms_sv.append(surface_area)
        self.pipeline.modifiers.append(InvertSelectionModifier())
        self.pipeline.modifiers.append(DeleteSelectedModifier())
        self.nb_sv.append(self.pipeline.compute().particles.count)
        self.fl_vol.append(filled_volume)

        # Exportar el archivo procesado
        try:
            export_file(self.pipeline, "outputs.dump/divacancy_training.dump", "lammps/dump", 
                        columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"])
            self.pipeline.modifiers.clear()
        except Exception as e:
            print(f"Error al exportar el archivo: {e}")

        # Crear y exportar el diccionario con los datos en un archivo JSON
        datos = {'surface_area': self.sms_sv, 'filled_volume': self.fl_vol, 'vecinos': self.nb_sv}
        output_path = 'outputs.vfinder/key_divacancy.json'
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, 'w') as f:
            json.dump(datos, f, indent=4)
        print("Procesamiento completado. Archivo generado en:", output_path)
