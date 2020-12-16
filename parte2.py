import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork
import sys

def load_data(file):
    data=pd.read_csv(file)
    array1 = np.array([data['x1'].values, data['x2'].values])
    array2 = np.array([data['y1'].values, data['y2'].values])

    return array1,array2
def savejson(diccionario):
    entradas=(str)(len(diccionario['W1'][0]))
    tempjson = '{\n     "entradas": '+entradas+',\n     "capas": ['
    for i in range(1,(int)(len(diccionario)/2+1)):
        cont =0
        tempjson = tempjson + '\n       {' + '\n            "neuronas": ['
        for j in diccionario['W'+(str)(i)]:
            tempjson = tempjson+'\n                   {'
            peso=(str)(list(diccionario['b'+(str)(i)][cont])+list(j))
            tempjson=tempjson+'\n                       '+'"pesos": '+peso
            if np.where(diccionario['W'+(str)(i)] == j )[0][0] == len(diccionario['W'+(str)(i)])-1:
                tempjson = tempjson + '\n                   }'
            else:
                tempjson = tempjson + '\n                   },'
            cont = cont+1

        if i == (int)(len(diccionario)/2):
            tempjson = tempjson + '\n            ]\n       }'
        else:
            tempjson = tempjson + '\n            ]\n       },'

    tempjson = tempjson + "\n     ]\n}"

    archivo = open(sys.argv[2], 'w')
    archivo.write(tempjson)
    archivo.close()



def main():
    X,Y = load_data(sys.argv[1])
    red = NeuralNetwork()
    for i in range(20):
        print("Model",i+1)
        red.structure = [2,2,2]
        red.initialize_parameters()
        red.back_propagation1(X,Y,100,[],[],-1,-1)
        print("Params",red.parameters)
        savejson(red.parameters)


if __name__ == '__main__':
    main()