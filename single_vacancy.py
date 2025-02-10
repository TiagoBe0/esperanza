import ovito
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, VoronoiAnalysisModifier, ClusterAnalysisModifier
import os
import json
import random
import re
from input_params import CONFIG as LAYERS

class SingleVacancyProcessor:
    def __init__(self, layer):
        primer_elemento = LAYERS[0]
        self.layer = primer_elemento
        self.relax = primer_elemento['relax']
        self.cutoff_radius = primer_elemento['cutoff radius']
        self.radius = primer_elemento['radius']
        self.smoothing_level = primer_elemento['smoothing level']
        self.pipeline = None
        self.sms_sv = []
        self.nb_sv = []



    def run(self):
        self.pipeline = import_file(self.relax)
        data = self.pipeline.compute()
        if not ids:
            print("No se encontraron IDs en el archivo.")
            return
        id_aleatorio = int(random.choice(ids))
        self.pipeline.modifiers.append(ExpressionSelectionModifier(expression=f'ParticleIdentifier=={id_aleatorio}'))
        self.pipeline.modifiers.append(DeleteSelectedModifier())
        self.pipeline.modifiers.append(VoronoiAnalysisModifier(compute_indices=True))
        self.pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=self.cutoff_radius, cluster_coloring=True, unwrap_particles=True, sort_by_size=True))
        data = self.pipeline.compute()
        vecinos = data.particles.count
        try:
            export_file(self.pipeline, "single_vacancy_training.dump", "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Atomic Volume", "Cavity Radius", "Max Face Order"])
            self.pipeline.modifiers.clear()
        except Exception as e:
            print(f"Error al exportar el archivo: {e}")
        datos = {'sms_sv': self.sms_sv, 'nb_sv': self.nb_sv}
        output_path = 'outputs.json/key_single_vacancy.json'
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, 'w') as f:
            json.dump(datos, f, indent=4)
