import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from NeuralNetwork import NeuralNetwork
import seaborn


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

    return np.transpose(np.array(X.values)),np.transpose(np.array(Y))


def stadistics(y_predict,y_true):    

    # f1
    results = f1_score(y_true, y_predict, average=None)
    acum = 0
    total = len(results)
    cont =0
    for j in results:
        print("F1",cont,":",j)
        acum = acum + j
        cont = cont+1
    f1 = acum / total
    print("Promedio F1 por clase: ",f1)
    #Accuracy
    print("Accuracy: ",accuracy_score(y_true, y_predict))
    #matriz
    matrix = confusion_matrix(y_true, y_predict)
    seaborn.heatmap(matrix, cmap='inferno', cbar=False, annot=True, fmt="")
    plt.title("Matriz Confusion")
    plt.show()


def main():
    X,Y = load_data("part3_data_train.csv")
    X_val,Y_val = load_data("part3_data_val.csv")
    X_test,Y_test = load_data("part3_data_test.csv")

    print("1",X.shape)
    print("2",Y.shape)

    red = NeuralNetwork()
    red.type = 'C'

    red.structure = [5,4,5]
    red.initialize_parameters()
    red.back_propagation2(X,Y,10,X_val,Y_val,0.05,3)
    
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

    stadistics(red.get_coded_y(predicts), red.get_coded_y(original))

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
    