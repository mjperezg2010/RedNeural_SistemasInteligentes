import json
import numpy as np
from neural_network import neural_network

def main():
    
    asdsa = neural_network ()
    asdsa.structure = [2,2,2]
    asdsa.learning_rate = 0.075
    asdsa.epochs = 100
    asdsa.initialize_parameters()
    #asdsa.load_info('part1_red_prueba.json')
    #asdsa.L_layer_model(np.array([[0,0,1,1],[0,1,0,1]]),np.array([[0,0,0,1],[0,1,1,0]]))
    asdsa.L_layer_model(np.transpose(np.array([[0,0,1,1],[0,1,0,1]])),
        np.transpose(np.array([[0,0,0,1],[0,1,1,0]])))

    Xsad,_ = asdsa.L_model_forward( np.array([[0],[0]] ))
    print(Xsad)

    #print(asdsa.structure)
    #print(asdsa.params)
    #asdsa.initialize_parameters([2,2,2])
    
    #Xsad,_ = asdsa.L_model_forward( np.array([[0],[0]] ))
    #print(Xsad)
                


if __name__ == '__main__':
    main()

