import json
import numpy as np
import random

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

        print(random.randint(0,8))

    return np.array(matrix),data['entradas']






def main():
   openJson("part1_red_prueba.json")




if __name__ == '__main__':
    main()

