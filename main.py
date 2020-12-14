import json
import numpy as np

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
   openJson("part1_red_prueba.json")




if __name__ == '__main__':
    main()

