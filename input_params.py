# CONFIG: Lista de diccionarios con la configuración para el procesamiento y entrenamiento.
CONFIG = [{
    # Ruta al archivo de entrada para el cálculo de relajación de la estructura.
    'relax' : 'inputs.dump/main.0',
    
    # Ruta al archivo de entrada que describe el defecto o vacancia (por ejemplo, void_7).
    'defect' : 'inputs.dump/tests/CHESE_18',

    # Parámetro de radio para cálculos de área de influencia u otros procesos geométricos.
    'radius' : 2,
    
    # Nivel de suavizado aplicado a los datos durante el preprocesamiento.
    'smoothing level' : 0, 
    
    # Nivel de suavizado específico para los datos de entrenamiento.
    'smoothing_level_training' : 0, 
    
    # Radio de corte: define el límite a partir del cual se ignoran interacciones o se delimita el área.
    'cutoff radius' : 3, 
    
    # Radio utilizado durante el entrenamiento para la selección o generación de datos.
    'radius_training': 3,
    
    # Indicador para activar o no métodos alternativos (por ejemplo, utilizar otro algoritmo de predicción).
    'other method': True,
    
    # Lista de parámetros de "strees" , si la muestra defectuosa sufrio una compresion del 20% en el eje x seria [0.8,1,1]
    'strees': [1, 1, 1],
    
    # Tolerancia en la agrupación (clustering): define la distancia máxima para considerar puntos en el mismo grupo.
    'cluster tolerance': 3,
    
    # Factor de división del cluster: puede ajustarse para definir cuántas particiones se desean (por ejemplo, 1, 2 o 3).
    'divisions_of_cluster': 1,  # Valores sugeridos: 1, 2 o 3.
    
    
    # Si es True, se generarán datos sintéticos (generic data) para ampliar el conjunto de entrenamiento.
    'generic_data':     True,
    
    # Indicador para activar o no funcionalidades de chat o interacciones en modo conversacional.
    'chat': False,

    'iteraciones_clusterig':3
}]

# PREDICTOR_COLUMNS: Lista de columnas que se utilizarán para entrenar el modelo predictivo.
PREDICTOR_COLUMNS = ['sm_mesh_training','vecinos','filled_volume']
#PREDICTOR_COLUMNS = ['sm_mesh_training', 'filled_volume', 'vecinos']