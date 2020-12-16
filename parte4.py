import numpy as np
import pandas as pd
from sklearn.metrics import  mean_squared_error
from sklearn.preprocessing import StandardScaler
from NeuralNetwork import NeuralNetwork
import sys


def load_data(file):
    data = pd.read_csv(file)
    X = data.loc[:, data.columns != 'nombre']
    X = X.loc[:, X.columns != 'PC']
    Y = data['PC']

    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    scaler1 = StandardScaler()

    #Y = np.array(Y)
    temp = np.zeros((1,len(Y)))
    for i in range(len(Y)):
        temp[0][i]=Y.loc[i]
    Y=temp
    Y = scaler1.fit_transform(Y)


    return np.transpose(scaled), Y

def stadistics(y_predicts, y_true):    
    print("MSE: ",mean_squared_error(y_true, y_predicts))


def main():
    X,Y = load_data(sys.argv[1])
    X_val,Y_val = load_data(sys.argv[2])
    X_test,Y_test = load_data(sys.argv[3])

    
    red = NeuralNetwork()
    red.out = 'sigmoid'
    #1
    red.structure = [7,16,1]
    red.initialize_parameters()

    print("X",X.shape)
    print("Y",Y.shape)

    red.back_propagation3(X,Y,50,X_val,Y_val,0.05,3)
    
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

    #2
    red.structure = [7, 32, 1]
    red.initialize_parameters()

    print("X", X.shape)
    print("Y", Y.shape)

    red.back_propagation3(X, Y, 50, X_val, Y_val, 0.05, 3)

    predicts = []
    original = []
    X_test_t = np.transpose(X_test)
    Y_test_t = np.transpose(Y_test)

    for j in range(len(X_test_t)):
        X_eval = X_test_t[j].reshape(len(X_test_t[j]), 1)
        Y_eval = Y_test_t[j].reshape(len(Y_test_t[j]), 1)
        y_predict, _ = red.L_model_forward(X_eval)
        predicts.append(y_predict.reshape(len(y_predict)))
        original.append(Y_eval.reshape(len(Y_eval)))

    stadistics(predicts, original)


    #3
    red.structure = [7, 16,16, 1]
    red.initialize_parameters()

    print("X", X.shape)
    print("Y", Y.shape)

    red.back_propagation3(X, Y, 50, X_val, Y_val, 0.05, 3)

    predicts = []
    original = []
    X_test_t = np.transpose(X_test)
    Y_test_t = np.transpose(Y_test)

    for j in range(len(X_test_t)):
        X_eval = X_test_t[j].reshape(len(X_test_t[j]), 1)
        Y_eval = Y_test_t[j].reshape(len(Y_test_t[j]), 1)
        y_predict, _ = red.L_model_forward(X_eval)
        predicts.append(y_predict.reshape(len(y_predict)))
        original.append(Y_eval.reshape(len(Y_eval)))

    stadistics(predicts, original)


    #4
    red.structure = [7, 32,32, 1]
    red.initialize_parameters()

    print("X", X.shape)
    print("Y", Y.shape)

    red.back_propagation3(X, Y, 50, X_val, Y_val, 0.05, 3)

    predicts = []
    original = []
    X_test_t = np.transpose(X_test)
    Y_test_t = np.transpose(Y_test)

    for j in range(len(X_test_t)):
        X_eval = X_test_t[j].reshape(len(X_test_t[j]), 1)
        Y_eval = Y_test_t[j].reshape(len(Y_test_t[j]), 1)
        y_predict, _ = red.L_model_forward(X_eval)
        predicts.append(y_predict.reshape(len(y_predict)))
        original.append(Y_eval.reshape(len(Y_eval)))

    stadistics(predicts, original)



    #5
    red.structure = [7, 4, 4, 4, 1]
    red.initialize_parameters()

    print("X", X.shape)
    print("Y", Y.shape)

    red.back_propagation3(X, Y, 50, X_val, Y_val, 0.05, 3)

    predicts = []
    original = []
    X_test_t = np.transpose(X_test)
    Y_test_t = np.transpose(Y_test)

    for j in range(len(X_test_t)):
        X_eval = X_test_t[j].reshape(len(X_test_t[j]), 1)
        Y_eval = Y_test_t[j].reshape(len(Y_test_t[j]), 1)
        y_predict, _ = red.L_model_forward(X_eval)
        predicts.append(y_predict.reshape(len(y_predict)))
        original.append(Y_eval.reshape(len(Y_eval)))

    stadistics(predicts, original)

if __name__ == '__main__':
    main()
    