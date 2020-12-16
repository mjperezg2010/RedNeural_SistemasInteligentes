import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data():
    data = pd.read_csv("part3_data_val.csv")
    X = data.loc[:, data.columns != 'clase']
    Ytemp = data['clase']

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    X = pd.DataFrame(scaled)
    Y=[]
    for i in Ytemp.values:
        if i == 'pizza':
            Y.append([1,0,0,0,0])
        elif i == 'hamburguesa':
            Y.append([0, 1, 0, 0, 0])
        elif i == 'arroz_frito':
            Y.append([0, 0, 1, 0, 0])
        elif i == 'ensalada':
            Y.append([0, 0, 0, 1, 0])
        elif i == 'pollo_horneado':
            Y.append([0, 0, 0, 0, 1])

    return np.array(X.values),np.transpose(Y)


def load_data2(file):
    data = pd.read_csv(file)
    X = data.loc[:, data.columns != 'nombre']
    X = X.loc[:, X.columns != 'PC']
    Y = data['PC']

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    X = pd.DataFrame(scaled)

    return np.array(X) ,np.array(Y)

def json1(diccionario):
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

    archivo = open("redaprendida.json", 'w')
    archivo.write(tempjson)
    archivo.close()

    print(tempjson)



#load_data2("part4_pokemon_go_train.csv")
array1 = np.array([[0.01796177, 0.00424671, 0.00081622, -0.0187593, -0.00279611,
                   -0.00357625, -0.00083272],
                  [-0.00630429, -0.00048846, -0.00483746, -0.01319141, 0.00886354,
                   0.00882932, 0.01713119],
                  [0.00056686, -0.00389512, -0.00526073, -0.01530995, 0.00977426,
                   -0.01095149, -0.01178844],
                  [-0.00188407, 0.01496501, 0.00256457, -0.01004957, -0.00691128,
                   0.00607277, -0.00155489]])
b1 = np.array([[0.00031126], [0.00014301], [-0.00041087],[-0.00055001]])
array2 = np.array([[-0.00810052, -0.00311021, 0.00663479, 0.01961822]])
b2 = np.array([[-0.07649589]])
diccionario = {'W1':array1,'b1':b1,'W2':array2,'b2': b2}
json1(diccionario)