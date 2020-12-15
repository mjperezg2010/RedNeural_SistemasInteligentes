
import json
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


class neural_network:
    def __init__(self):
        pass
    
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
        self.epochs = 100

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

    def linear_activation_forward(self,A_prev, W, b):        
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        A, activation_cache = self.sigmoid(Z)  
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
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])
            caches.append(cache)
        
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])    
        caches.append(cache)
                
        return AL, caches

    def compute_cost(self,AL, Y):    
        m = Y.shape[1]
        cost = (-1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))    
        cost = np.squeeze(cost)        
        return cost

    def linear_backward(self,dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1/m)*np.dot(dZ,np.transpose(A_prev))
        db = (1/m)*np.sum(dZ,axis=1,keepdims = True)
        dA_prev = np.dot(np.transpose(W),dZ)
        return dA_prev, dW, db

    def linear_activation_backward(self,dA, cache):
        linear_cache, activation_cache = cache

        dZ = self.sigmoid_backward(dA, activation_cache)
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
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache)

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l+1)], current_cache)
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


    def L_layer_model(self,X, Y):                            
        epocas = {}
        for j in range(0, self.epochs):
            predic = []
            original = []
            for i in range(len(X)):                
                AL, caches = self.L_model_forward(np.array(X[i]).reshape(len(X[i]),1))                
                grads = self.L_model_backward(AL, np.array(Y[i]).reshape(len(Y[i]),1), caches)
                self.update_parameters(grads)
                predic.append(AL)
                original.append(X[i])
                #print ("Cost after iteration %i: %f" %(i, cost))            
                #costs.append(cost)
            epocas['mse'+str(j)] = [predic,original]
        self.plot_mse(epocas)
    
    def plot_mse(self,epocas):
        x=[]
        y=[]
        for i in epocas:
            matriz1=[[epocas[i][0][0][0][0],epocas[i][0][0][1][0]],
                      [epocas[i][0][1][0][0],epocas[i][0][1][1][0]],
                      [epocas[i][0][2][0][0],epocas[i][0][2][1][0]],
                      [epocas[i][0][3][0][0],epocas[i][0][3][1][0]]
                    ]
            matriz2 = [[epocas[i][1][0][0],epocas[i][1][0][1]],
                       [epocas[i][1][1][0], epocas[i][1][1][1]],
                       [epocas[i][1][2][0], epocas[i][1][2][1]],
                       [epocas[i][1][3][0], epocas[i][1][3][1]]
                     ]
            y.append(mean_squared_error(matriz2, matriz1))
            x.append(i)
        #print(epocas[0])
        plt.plot(x,y)
        plt.show()




