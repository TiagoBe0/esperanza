import os
import glob
import pandas as pd

def cargar_datos(lista_archivos):
    """
    Recibe una lista de archivos CSV, los procesa en orden (ordenados alfabéticamente)
    y agrupa las predicciones según la columna 'method'.
    """
    # Asegurarse de que la lista esté ordenada
    lista_archivos = sorted(lista_archivos)
    
    datos_modelos = {}
    for archivo in lista_archivos:
        df = pd.read_csv(archivo)
        df.columns = df.columns.str.strip()  # Elimina espacios en blanco en los nombres de columna
        
        if 'method' not in df.columns or 'prediction' not in df.columns:
            print(f"El archivo {archivo} no tiene las columnas esperadas ('method' y 'prediction').")
            continue
        
        for _, row in df.iterrows():
            metodo = row['method']
            prediction = row['prediction']
            if pd.isna(prediction):
                continue
            if metodo not in datos_modelos:
                datos_modelos[metodo] = []
            datos_modelos[metodo].append(prediction)
    
    return datos_modelos

# Función de exportación (se mantiene igual)
def export_results(results, archivo_base):
    """
    Exporta los resultados agrupados en 'results' a un archivo CSV.
    Cada columna representará un método y se ordenarán por 'iteration' y 'method'.
    
    - results: diccionario con la estructura { método: { "predictions": [...], "errors": [...] } }
    - archivo_base: nombre base del archivo de salida
    """
    dfs = []
    for method, data in results.items():
        num_rows = len(data["predictions"])
        df_method = pd.DataFrame({
            "iteration": list(range(num_rows)),
            "method": [method] * num_rows,
            "prediction": data["predictions"],
            "true_value": data.get("vector_true", [None] * num_rows),
            "squared_error": data["errors"]
        })
        dfs.append(df_method)
    
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        # Ordena el DataFrame por 'iteration' y 'method'
        final_df = final_df.sort_values(by=["iteration", "method"]).reset_index(drop=True)
        
        output_dir = "outputs.csv"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{archivo_base}.csv")
        final_df.to_csv(output_file, index=False)
        print(f"Resultados exportados en {output_file}")
    else:
        print("No hay resultados para exportar.")

# Ejemplo de uso:
if __name__ == "__main__":
    # Si deseas obtener la lista de archivos desde un directorio:
    ruta = "outputs.csv/"  # Cambia esta ruta según sea necesario
    #lista_archivos = ['outputs.csv/CHESE_18_x1.csv','outputs.csv/CHESE_18_x2.csv','outputs.csv/CHESE_18_x3.csv','outputs.csv/CHESE_18_x4.csv','outputs.csv/CHESE_18_x5.csv']
    lista_archivos=['outputs_tt.csv/chesse15_totales_vacancias.csv','outputs_tt.csv/chese15x2_totales_vacancias.csv','outputs_tt.csv/chese15x3_totales_vacancias.csv','outputs_tt.csv/chese15x4_totales_vacancias.csv','outputs_tt.csv/chese15x5_totales_vacancias.csv']
    # O bien, puedes definir la lista manualmente:
    # lista_archivos = ["archivo1.csv", "archivo2.csv", "archivo3.csv"]
    
    datos = cargar_datos(lista_archivos)
    
    # Mostrar los datos agrupados por método
    for metodo, predicciones in datos.items():
        print(f"Método: {metodo}, Predicciones: {predicciones}")
    
    # Supongamos que 'results' es un diccionario con la siguiente estructura:
    # { 'linear': {"predictions": [...], "errors": [...], "vector_true": [...] }, ... }
    # Aquí solo se usa 'datos' como ejemplo, pero deberás integrar estos datos en tu estructura de resultados.
    # Por ejemplo, si ya tienes un objeto "self.results", podrías llamarlo de esta forma:
    # export_results(self.results, "nombre_base")
    
    # Para este ejemplo, armamos un 'results' simple:
    results = {}
    for metodo, preds in datos.items():
        results[metodo] = {
            "predictions": preds,
            "errors": [None] * len(preds),  # Aquí podrías calcular o asignar los errores correspondientes
            "vector_true": [None] * len(preds)
        }
    
    # Exportar resultados usando un nombre base para el archivo de salida
    export_results(results, "resultados_combinados")
