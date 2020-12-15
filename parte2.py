import pandas
import numpy as np
from neural_network import neural_network

def load_data(file):
    data=pandas.read_csv(file)
    array1 = np.transpose(np.array([data['x1'].values, data['x2'].values]))
    array2 = np.transpose(np.array([data['y1'].values, data['y2'].values]))

    return array1,array2


def main():
    X,Y = load_data("part2_train_data.csv")

    print("nicky1",X)
    print("nicky2",Y)

    red = neural_network()
    for i in range(20):
        print("Model",i+1)
        red.structure = [2,2,2]
        red.initialize_parameters()
        red.L_layer_model(X,Y,100,X_val=[],Y_val=[],epsilon=-1,max_rounds=-1)
        print("Params",red.params)
        print()



if __name__ == '__main__':
    main()