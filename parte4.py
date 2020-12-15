import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from neural_network import neural_network
import seaborn


def load_data(file):
    data = pd.read_csv(file)
    X = data.loc[:, data.columns != 'nombre']
    X = X.loc[:, X.columns != 'PC']
    Y = data['PC']

    #scaler = StandardScaler()
    #scaled = scaler.fit_transform(X)
    scaled = np.array(X.values)
    for i in range(len(scaled[0])):
        OldMin = min(scaled[:,i])
        OldMax = max(scaled[:,i])
        NewMax = 1
        NewMin = -1
        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)
        for j in range(len(scaled[:,i])):
            NewValue = (((scaled[:,i][j] - OldMin) * NewRange) / OldRange) + NewMin
            scaled[:,i][j] = NewValue
    print(scaled)


    Y = np.array(Y)

    OldMin = min(Y)
    OldMax = max(Y)
    OldRange = (OldMax - OldMin)

    for i in range(len(Y)):
        NewValue = (((Y[i] - OldMin) * NewRange) / OldRange) + NewMin
        Y[i] = NewValue



    return scaled, Y

def stadistics(y_predicts, y_true):
    print("MSE: ",mean_squared_error(y_true, y_predicts))




def main():
    X,Y = load_data("part4_pokemon_go_train.csv")
    X_val,Y_val = load_data("part4_pokemon_go_validation.csv")
    X_test,Y_test = load_data("part4_pokemon_go_test.csv")

    red = neural_network()
    red.type = 'regression'
    red.flag = True

    red.structure = [7,16,1]
    red.initialize_parameters()
    red.L_layer_model(X,Y.reshape(Y.shape[0],1),50,X_val,Y_val.reshape(Y_val.shape[0],1),0.05,3)

    predicts = []
    for i in X_test:
        y_predict, _ = red.L_model_forward(np.array(i).reshape(len(i), 1))
        predicts.append(list(y_predict.reshape(len(y_predict))))
    stadistics(predicts, Y_test)

    print("--------------------")
'''
    red.structure = [5,16,5]
    red.initialize_parameters()    
    red.L_layer_model(X,Y,50,X_val,Y_val,0.05,3)

    predicts = []
    for i in X_test:
        y_predict, _ = red.L_model_forward(np.array(i).reshape(len(i), 1))
        predicts.append(list(y_predict.reshape(len(y_predict))))
    stadistics(red.get_coded_y(predicts), red.get_coded_y(Y_test))

    print("--------------------")
    red.structure = [5,32,5]
    red.initialize_parameters()    
    red.L_layer_model(X,Y,50,X_val,Y_val,0.05,3)

    predicts = []
    for i in X_test:
        y_predict, _ = red.L_model_forward(np.array(i).reshape(len(i), 1))
        predicts.append(list(y_predict.reshape(len(y_predict))))
    stadistics(red.get_coded_y(predicts), red.get_coded_y(Y_test))

    print("--------------------")
    red.structure = [5,64,5]
    red.initialize_parameters()    
    red.L_layer_model(X,Y,50,X_val,Y_val,0.05,3)

    predicts = []
    for i in X_test:
        y_predict, _ = red.L_model_forward(np.array(i).reshape(len(i), 1))
        predicts.append(list(y_predict.reshape(len(y_predict))))
    stadistics(red.get_coded_y(predicts), red.get_coded_y(Y_test))
'''


if __name__ == '__main__':
    main()
    