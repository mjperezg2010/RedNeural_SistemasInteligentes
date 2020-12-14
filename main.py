import json

def openJson(file_name):
    matriz = []
    with open(file_name) as file:
        data = json.load(file)

        matriztemp = []
        pesos1 = data['capas'][0]['neuronas'][0]['pesos']
        pesos2 = data['capas'][0]['neuronas'][1]['pesos']

        matriztemp.append(pesos1)
        matriztemp.append(pesos2)

        matriztemp2 = []
        pesos1 = data['capas'][1]['neuronas'][0]['pesos']
        pesos2 = data['capas'][1]['neuronas'][1]['pesos']
        matriztemp2.append(pesos1)
        matriztemp2.append(pesos2)


        matriz.append(matriztemp)
        matriz.append(matriztemp2)

    return matriz,data['entradas']






def main():
   openJson("part1_red_prueba.json")




if __name__ == '__main__':
    main()

