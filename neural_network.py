
import json
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score


class neural_network:
    def __init__(self):
        self.learning_rate = 0.25
        self.epochs = 100
        self.epsilon = -1
        self.type = 'regression'
        self.structure = []
        self.flag = False
    
    def load_info(self,file_name):
        params = {}
        structure = []
        with open(file_name) as file:
            data = json.load(file)                    
            structure.append(data['entradas'])
            for i in range(len(data['capas'])):
                matrixtemp=[]
                biastemp=[]
                cont = 0
                for j in range(len(data['capas'][i]['neuronas'])):
                    peso = data['capas'][i]['neuronas'][j]['pesos']
                    biastemp.append(peso[0])
                    matrixtemp.append(peso[1:])
                    cont += 1
                structure.append(cont)
                params['W'+str(i+1)] = np.array(matrixtemp)
                params['b'+str(i+1)] = (np.array(biastemp)).reshape((len(biastemp),1))
        self.params = params
        self.structure = structure
        self.learning_rate = 0.075        

    def initialize_parameters(self):    
        parameters = {}
        structure = self.structure
        L = len(structure)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.uniform(low=-1, high=1, size=(structure[l],structure[l-1]))
            parameters['b' + str(l)] = np.random.uniform(low=-1, high=1, size=(structure[l],1))
        
        self.params = parameters     

    def linear_forward(self,A, W, b):    
        Z = np.dot(W,A)+b    
        cache = (A, W, b)    
        return Z, cache

    def linear_activation_forward(self,A_prev, W, b, _type):        
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        if _type != 'regression':
            A, activation_cache = self.sigmoid(Z)  
        else:
            A = Z
            activation_cache = linear_cache
        cache = (linear_cache, activation_cache)
        return A, cache
        
    def sigmoid(self,Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        return A, cache

    def L_model_forward(self,X):  
        parameters = self.params  
        caches = []
        A = X
        L = len(parameters) // 2
        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],"asd")
            caches.append(cache)

        if self.flag:        
            AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)],"regression")
        else:
            AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)],"sdasd")
        caches.append(cache)
                
        return AL, caches

    def linear_backward(self,dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1/m)*np.dot(dZ,np.transpose(A_prev))
        db = (1/m)*np.sum(dZ,axis=1,keepdims = True)
        dA_prev = np.dot(np.transpose(W),dZ)
        return dA_prev, dW, db

    def linear_activation_backward(self,dA, cache, type_):
        linear_cache, activation_cache = cache
        if type_ != 'regression_last':        
            dZ = self.sigmoid_backward(dA, activation_cache)
        else:
            dZ = dA
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        return dA_prev, dW, db

    def sigmoid_backward(self,dA, cache):
        Z = cache 
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        return dZ
    
    

    def L_model_backward(self,AL,Y,caches):
    
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape) 

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L-1]
        if self.flag:
            grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache,"regression_last")
        else:
            grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache,"asdasds")

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l+1)], current_cache,"asd")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads


    def update_parameters(self, grads):
        L = len(self.params) // 2 
        for l in range(L):
            self.params["W" + str(l+1)] = self.params["W" + str(l+1)] - self.learning_rate*grads["dW" + str(l+1)]
            self.params["b" + str(l+1)] = self.params["b" + str(l+1)] - self.learning_rate*grads["db" + str(l+1)]
        return self.params


    def L_layer_model(self,X,Y,epochs,X_val=[],Y_val=[],epsilon=-1,max_rounds=-1):
        epocas = {}
        epocas_val = {}
        actual_rounds = -1
        mse_actual = 0
        mse_prev = 100
        self.epochs = epochs
        for j in range(0, epochs):            
            predic = []
            original = []
            for i in range(len(X)):                
                AL, caches = self.L_model_forward(np.array(X[i]).reshape(len(X[i]),1))
                grads = self.L_model_backward(AL, np.array(Y[i]).reshape(len(Y[i]),1), caches)
                self.update_parameters(grads)
                predic.append(AL.reshape(len(AL)))
                original.append(Y[i])
            
            epocas[str(j)] = [predic,original]

            _predic = []
            _original = []
            for i in range(len(X_val)):
                AL,_ = self.L_model_forward(np.array(X_val[i]).reshape(len(X_val[i]),1))
                _predic.append(list(AL.reshape(len(AL))))
                _original.append(list(Y_val[i]))
            epocas_val[str(j)] = [_predic,_original]

            
            if epsilon!=-1 and max_rounds!=-1:
                mse_actual = mean_squared_error(original, predic)
                if abs(mse_actual-mse_prev) > epsilon:
                    actual_rounds+=1                
                else:
                    actual_rounds = 0
                mse_prev = mse_actual
                if actual_rounds == max_rounds:
                    break
                
       
        if self.type == 'regression':
            self.plot_mse(epocas)
            if len(X_val) != 0:
                self.plot_mse(epocas_val) 
        else:
            self.plot_classifcacion(epocas)
            if len(X_val) != 0:
                self.plot_classifcacion(epocas_val)
            

    
    def calculate_stats(self,Y_pred,Y_real):
        print("PRED",Y_pred)
        print("REAL",Y_real)

    def get_coded_y(self,Y):
        Y_ret = []

        for y in Y:
            max_y = y[0]
            pos = 0
            for i in range(1,len(y)):
                if y[i] > max_y:
                    max_y = y[i]
                    pos = i
            Y_ret.append(pos)
        return Y_ret
    

    def evaluar(self,X):        
        Y,_  = self.L_model_forward(X)
        return Y

    def plot_classifcacion(self,epocas):
        cont=-1
        x=[]
        yf1=[]
        yacc=[]
        for i in epocas:
            matrix1= self.get_coded_y(epocas[i][0])
            matrix2=self.get_coded_y(epocas[i][1])
            if cont ==-1:
                #f1
                results = f1_score(matrix2, matrix1,average=None)
                acum = 0
                total = len(results)
                for j in results:
                    acum = acum + j
                temp= acum / total

                yf1.append(temp)
                yacc.append(accuracy_score(matrix2,matrix1))
                x.append(i)
                cont=0

            if cont == 5:
                #f1
                results = f1_score(matrix2, matrix1, average=None)
                acum = 0
                total = len(results)
                for j in results:
                    acum = acum + j
                temp = acum / total

                yf1.append(temp)
                yacc.append(accuracy_score(matrix2, matrix1))
                x.append(i)
                cont = 0
            cont = cont + 1
        plt.plot(x, yf1, label="F1-score")
        plt.plot(x, yacc, label="Accuracy")
        plt.legend()
        plt.title("Estadistica")
        plt.show()


    def plot_mse(self,epocas):
        
        x=[]
        y=[]
        ymin=[]
        ymax=[]
        matriz1=[]
        matriz2=[]
        cont =-1
        

        for i in epocas:
            matriz1=epocas[i][0]
            matriz2=epocas[i][1]
            #Minimun and maximun
            
            list1 = []
            for j in range(len(matriz1)):
                for k in range(len(matriz1[0])):
                    #print(matriz1[j][k])
                    temp = mean_squared_error([matriz2[j][k]], [matriz1[j][k]])
                    list1.append(temp)



            if cont ==-1:
                y.append(mean_squared_error(matriz2, matriz1))
                x.append(i)
                ymin.append(min(list1))
                ymax.append(max(list1))

                cont=0


            if cont == 1:
                y.append(mean_squared_error(matriz2, matriz1))
                ymin.append(min(list1))
                ymax.append(max(list1))
                x.append(i)
                cont = 0
            cont = cont + 1
        plt.plot(x, y, label="MSE")
        plt.plot(x, ymin, label="MSE minumun")
        plt.plot(x, ymax, label="MSE maximun")
        #plt.ylim(0, 1)
        plt.legend()
        plt.title("MSE")

        plt.show()


