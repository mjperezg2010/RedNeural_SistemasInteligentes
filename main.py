import json
import numpy as np
from neural_network import neural_network

def openJson(file_name):
    matrix = []
    with open(file_name) as file:
        data = json.load(file)

        for i in range(len(data['capas'])):
            matrixtemp=[]
            for j in range(len(data['capas'][i]['neuronas'])):
                peso = data['capas'][i]['neuronas'][j]['pesos']
                matrixtemp.append(peso)
            matrix.append(matrixtemp)

        print(np.array(matrix))


    return np.array(matrix),data['entradas']


def main():

    asdsa = neural_network ()
    asdsa.structure = [2,2,2]
    asdsa.learning_rate = 0.075
    asdsa.epochs = 300
    asdsa.initialize_parameters()
    #asdsa.load_info('part1_red_prueba.json')
    #asdsa.L_layer_model(np.array([[0,0,1,1],[0,1,0,1]]),np.array([[0,0,0,1],[0,1,1,0]]))
    asdsa.L_layer_model(np.transpose(np.array([[0,0,1,1],[0,1,0,1]])),

        np.transpose(np.array([[0,0,0,1],[0,1,1,0]])),100)

    

    Xsad,_ = asdsa.L_model_forward( np.array([[0],[0]] ))
    print(Xsad)

    #print(asdsa.structure)
    #print(asdsa.params)
    #asdsa.initialize_parameters([2,2,2])
    
    #Xsad,_ = asdsa.L_model_forward( np.array([[0],[0]] ))
    #print(Xsad)
    


if __name__ == '__main__':
    main()

