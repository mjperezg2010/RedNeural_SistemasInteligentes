
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from neural_network import neural_network


def load_data(file_name):
    data = pd.read_csv(file_name)
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

    return np.array(X.values),np.array(Y)

'''
nicky1 [[0 0]
 [0 1]
 [1 0]
 [1 1]]
nicky2 [[0 0]
 [0 1]
 [0 1]
 [1 0]]

 '''

def main():
    X,Y = load_data("part3_data_train.csv")
    X_val,Y_val = load_data("part3_data_test.csv")

    red = neural_network()
    red.type = 'classification'

    red.structure = [5,4,5]
    red.initialize_parameters()
    red.L_layer_model(X,Y,50,X_val,Y_val,0.05,3)
'''
    red.structure = [5,16,5]
    red.initialize_parameters()    
    red.L_layer_model(X,Y,50,X_val,Y_val,0.05,3)

    red.structure = [5,32,5]
    red.initialize_parameters()    
    red.L_layer_model(X,Y,50,X_val,Y_val,0.05,3)

    red.structure = [5,64,5]
    red.initialize_parameters()    
    red.L_layer_model(X,Y,50,X_val,Y_val,0.05,3)
   ''' 


if __name__ == '__main__':
    main()
    