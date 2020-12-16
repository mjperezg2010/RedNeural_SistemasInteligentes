from NeuralNetwork import NeuralNetwork
import numpy as np

def main():
    red = NeuralNetwork ()                
    red.structure = [2,2,2]
    red.load_params('part1_red_prueba.json')    
    data = np.array([np.array([[0],[0]]),
                    np.array([[1],[0]]),
                    np.array([[0],[1]]),
                    np.array([[1],[1]]) ])
    print("<0,0>")
    print(red.evaluar(data[0]))
    print("<1,0>")
    print(red.evaluar(data[1]))
    print("<0,1>")
    print(red.evaluar(data[2]))
    print("<1,1>")
    print(red.evaluar(data[3]))

    red.initialize_parameters()

    print("PARAMS:",red.parameters)

    print("<0,0>")
    print(red.evaluar(data[0]))
    print("<1,0>")
    print(red.evaluar(data[1]))
    print("<0,1>")
    print(red.evaluar(data[2]))
    print("<1,1>")
    print(red.evaluar(data[3]))

if __name__ == '__main__':
    main()

