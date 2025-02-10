         
CONFIG = [{'relax' : 'inputs.dump/data.relax',
          'defect' : 'inputs.dump/dump-finalCool.160000',
          'radius' : 2,
          'smoothing level' : 13, 
          'smoothing_level_training' : 17, 
          'cutoff radius' : 3, 
             'iteraciones':1 ,
              'bOvitoModifiers':True, #Si esta desactivado por defecto se utilizara MultiSOM
            'columns_train':[0,1]#surface,vecinos,norma menor,norma mayor
            ,'strees': [1,1,1],
            'cluster tolerance':1.2,
            'divisions_of_cluster':20#1  , 2 o 3
            ,'radius_training':3,
            #TRAINING
            'other method': True,
        'vecinos':True,
        'max_distancias': False,
        'min_distancias': False

          }]