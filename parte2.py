import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork

def load_data(file):
    data=pd.read_csv(file)
    array1 = np.array([data['x1'].values, data['x2'].values])
    array2 = np.array([data['y1'].values, data['y2'].values])

    return array1,array2

def main():
    X,Y = load_data("part2_train_data.csv")                
    red = NeuralNetwork()
    for i in range(20):
        print("Model",i+1)
        red.structure = [2,2,2]
        red.initialize_parameters()
        red.back_propagation1(X,Y,100,[],[],-1,-1)
        print("Params",red.parameters)
        print()

if __name__ == '__main__':
    main()