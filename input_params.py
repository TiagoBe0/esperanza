         
CONFIG = [{'relax' : 'inputs.dump/main.0',
          'defect' : 'inputs.dump/void_15',
          'radius' : 2,
          'smoothing level' :18, 
          'smoothing_level_training' : 16, 
          'cutoff radius' : 3.5, 
             'iteraciones':1 ,
              'bOvitoModifiers':True, #Si esta desactivado por defecto se utilizara MultiSOM
            'columns_train':[0,1]#surface,vecinos,norma menor,norma mayor
            ,'strees': [1,1,1],
            'cluster tolerance':1.2,
            'divisions_of_cluster':10#1  , 2 o 3
            ,'radius_training':3,
            #TRAINING
            'other method': True,
        'vecinos':True,
        'max_distancias': False,
        'min_distancias': False

          }]