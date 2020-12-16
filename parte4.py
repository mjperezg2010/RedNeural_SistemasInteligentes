import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from NeuralNetwork import NeuralNetwork
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
    #print(scaled)


    Y = np.array(Y)
    Y= Y.astype(float)

    OldMin = min(Y)
    OldMax = max(Y)
    NewMax = 1
    NewMin = -1
    NewRange = (NewMax - NewMin)

    OldRange = (OldMax - OldMin)

    for i in range(len(Y)):
        NewValue = (((Y[i] - OldMin) * NewRange) / OldRange) + NewMin
        Y[i] = NewValue    

    return np.transpose(scaled), Y.reshape(1,len(Y))

def stadistics(y_predicts, y_true):    
    print("MSE: ",mean_squared_error(y_true, y_predicts))


def main():
    X,Y = load_data("part4_pokemon_go_train.csv")
    X_val,Y_val = load_data("part4_pokemon_go_validation.csv")
    X_test,Y_test = load_data("part4_pokemon_go_test.csv")

    
    red = NeuralNetwork()
    red.out = 'linear'
    red.structure = [7,16,1]
    red.initialize_parameters()

    print("X",X.shape)
    print("Y",Y.shape)

    red.back_propagation3(X,Y,5,X_val,Y_val,0.05,3)
    
    predicts = []
    original=[]
    X_test_t = np.transpose(X_test)
    Y_test_t = np.transpose(Y_test)
        
    for j in range(len(X_test_t)):
        X_eval = X_test_t[j].reshape(len(X_test_t[j]),1)
        Y_eval = Y_test_t[j].reshape(len(Y_test_t[j]),1)
        y_predict, _ = red.L_model_forward(X_eval)        
        predicts.append(y_predict.reshape(len(y_predict)))
        original.append(Y_eval.reshape(len(Y_eval)))

    stadistics(predicts, original)
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
    