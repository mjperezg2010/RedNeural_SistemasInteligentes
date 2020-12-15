import numpy as np
def plot_mse():
    diccionario = {'nombre' : 'Carlos', 'edad' : 22, 'cursos': ['Python','Django','JavaScript'] }
    matriz1=[[1,2],[2,3],[1,2],[2,3]]
    matriz2 = [[1,2],[2,3],[1,2],[2,3]]
    matriz1 = np.array(matriz1)
    matriz2= np.array(matriz2)
    epocas = {'mse0':[matriz1,matriz2]}

    print(epocas['mse0'][0])
    pass


plot_mse()