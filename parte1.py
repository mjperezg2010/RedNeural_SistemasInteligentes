from neural_network import neural_network
import numpy as np

def main():
    red = neural_network ()                
    red.load_info('part1_red_prueba.json')    
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

    print("PARAMS:",red.params)

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

