import json
import numpy as np
from scipy.interpolate import interp1d

class SyntheticDataGenerator:
    def __init__(self, data, num_points=100, interpolation_kind='linear'):

        self.data = data
        self.num_points = num_points
        self.interpolation_kind = interpolation_kind
        
        required_keys = ["sm_mesh_training", "filled_volume", "vacancias", "vecinos"]
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"La clave '{key}' no se encuentra en los datos.")
        
        self.vacancias = np.array(self.data["vacancias"])
    
    def generate(self):
        vac_new = np.linspace(self.vacancias.min(), self.vacancias.max(), self.num_points)
        
        interp_sm = interp1d(self.vacancias, self.data["sm_mesh_training"], kind=self.interpolation_kind)
        sm_new = interp_sm(vac_new)
        
        interp_filled = interp1d(self.vacancias, self.data["filled_volume"], kind=self.interpolation_kind)
        filled_new = interp_filled(vac_new)
        
        interp_vecinos = interp1d(self.vacancias, self.data["vecinos"], kind=self.interpolation_kind)
        vecinos_new = np.round(interp_vecinos(vac_new)).astype(int)
        
        synthetic_data = {
            "sm_mesh_training": sm_new.tolist(),
            "filled_volume": filled_new.tolist(),
            "vacancias": vac_new.tolist(),
            "vecinos": vecinos_new.tolist()
        }
        return synthetic_data

    def export_to_json(self, output_path, data):
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Datos exportados a {output_path}")

if __name__ == "__main__":
    json_file_path = "outputs.vfinder/training_results.json"  

    with open(json_file_path, "r") as f:
        data_original = json.load(f)
    
    generator = SyntheticDataGenerator(data_original, num_points=100, interpolation_kind='linear')
    synthetic_data = generator.generate()
    
    output_json_path = "outputs.vfinder/synthetic_data.json"
    generator.export_to_json(output_json_path, synthetic_data)
