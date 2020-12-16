import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd
from sklearn.metrics import f1_score,accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import json

class NeuralNetwork:
    def __init__(self):
        self.learning_rate = 0.075        
        self.flag = 'P1'
        self.structure = []        
        self.parameters = {}
        self.type = "R"
        self.out = 'sigmoid'

    def sigmoid(self,Z):
        A = 1 / (1 + np.exp(-Z))
        cache = Z
        return A, cache

    def sigmoid_backward(self,dA,cache):
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ
    
    def initialize_parameters(self):
        structure = self.structure
        parameters = {}
        L = len(structure)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.uniform(low=-1, high=1, size=(structure[l], structure[l - 1]))
            parameters['b' + str(l)] = np.random.uniform(low=-1, high=1, size=(structure[l], 1))
        self.parameters = parameters



    def linear_forward(self,A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self,A_prev, W, b, activation):
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        elif activation == "linear":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A = Z
            activation_cache = linear_cache

        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self,X):
        parameters = self.parameters
        caches = []
        A = X
        L = len(parameters) // 2                

        for l in range(1, L):
            A_prev = A 
            A, linear_activation_cache = self.linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "sigmoid")
            caches.append(linear_activation_cache)

        AL, linear_activation_cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], self.out)
        caches.append(linear_activation_cache)

        return AL, caches

    def compute_cost(self,AL, Y):
        cost = mean_squared_error(Y, AL)
        return cost

    def linear_backward(self,dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis = 1, keepdims = True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def linear_activation_backward(self,dA, cache, activation):
        
        linear_cache, activation_cache = cache

        if activation == "linear":
            dZ = dA
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def L_model_backward(self,AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dA_prev, dW, db = self.linear_activation_backward(dAL, caches[L - 1], self.out)
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = dA_prev, dW, db

        for l in reversed(range(L-1)):
            dA = dA_prev
            dA_prev, dW, db = self.linear_activation_backward(dA, caches[l], "sigmoid")
            grads["dA" + str(l + 1)] = dA_prev
            grads["dW" + str(l + 1)] = dW
            grads["db" + str(l + 1)] = db

        return grads

    def update_parameters(self,grads):
        parameters = self.parameters
        learning_rate = self.learning_rate
        L = len(parameters) // 2 
        for l in range(L):
            parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
        self.parameters = parameters        

    def evaluar(self,X):
        probas, _ = self.L_model_forward(X)
        return probas
  
    def back_propagation3(self,X,Y,epochs,X_val,Y_val,epsilon,max_rounds):
        
        flag = (len(X_val) != 0)
        mse_prev = 9999
        rounds = -1
        epocas = {}             
        X_temp = np.transpose(X)
        Y_temp = np.transpose(Y)
        
        if flag:
            epocas_val = {}
            X_val_temp = np.transpose(X_val)
            Y_val_temp = np.transpose(Y_val)
        
        for i in range(epochs):
            predic = []
            original = []
            for j in range(len(X_temp)):
                X_eval = X_temp[j].reshape(len(X_temp[j]),1)
                Y_eval = Y_temp[j].reshape(len(Y_temp[j]),1)
                AL, caches = self.L_model_forward(X_eval)
                grads = self.L_model_backward(AL, Y_eval, caches)
                self.update_parameters(grads)
                predic.append(AL.reshape(len(AL)))
                original.append(Y_eval.reshape(len(Y_eval)))
            if flag:
                predic_val = []
                original_val = []
                for j in range(len(X_val_temp)):
                    X_eval_val = X_val_temp[j].reshape(len(X_val_temp[j]),1)
                    Y_eval_val = Y_val_temp[j].reshape(len(Y_val_temp[j]),1)
                    AL_val, _ = self.L_model_forward(X_eval_val)
                    predic_val.append(AL_val.reshape(len(AL_val)))                    
                    original_val.append(Y_eval_val.reshape(len(Y_eval_val)))
                    
                mse = mean_squared_error(original_val,predic_val)            
            else:
                mse = mean_squared_error(original,predic)


            epocas[str(i)] = [predic,original]
            if flag:
                epocas_val[str(i)] = [predic_val,original_val]
            epsilon =-1
            if epsilon != -1:
                if abs(mse - mse_prev) < epsilon:
                    rounds+=1
                else:
                    rounds = 0

                if rounds == max_rounds:
                    break
                mse_prev = mse
            
    
        self.plot_mse(epocas)
        if flag:
            self.plot_mse(epocas_val)

    def back_propagation2(self,X,Y,epochs,X_val,Y_val,epsilon,max_rounds):
        flag = (len(X_val) != 0)
        f1_prev = 9999
        rounds = -1
        epocas = {}             
        X_temp = np.transpose(X)
        Y_temp = np.transpose(Y)
        
        if flag:
            epocas_val = {}
            X_val_temp = np.transpose(X_val)
            Y_val_temp = np.transpose(Y_val)
        
        for i in range(epochs):
            predic = []
            original = []
            for j in range(len(X_temp)):
                X_eval = X_temp[j].reshape(len(X_temp[j]),1)
                Y_eval = Y_temp[j].reshape(len(Y_temp[j]),1)
                AL, caches = self.L_model_forward(X_eval)
                grads = self.L_model_backward(AL, Y_eval, caches)
                self.update_parameters(grads)
                predic.append( AL.reshape(len(AL)) )
                original.append(Y_eval.reshape(len(Y_eval)))
            if flag:
                predic_val = []
                original_val = []
                for j in range(len(X_val_temp)):
                    X_eval_val = X_val_temp[j].reshape(len(X_val_temp[j]),1)
                    Y_eval_val = Y_val_temp[j].reshape(len(Y_val_temp[j]),1)
                    AL_val, _ = self.L_model_forward(X_eval_val)
                    predic_val.append(AL_val.reshape(len(AL_val)))
                    original_val.append(Y_eval_val.reshape(len(Y_eval_val)))
                    
                f1 = f1_score(self.get_coded_y(original_val),self.get_coded_y(predic_val),average='macro')
            else:
                f1 = f1_score(self.get_coded_y(original),self.get_coded_y(predic),average='macro')

            epocas[str(i)] = [predic,original]
            if flag:
                epocas_val[str(i)] = [predic_val,original_val]

            if epsilon != -1:
                if abs(f1 - f1_prev) > epsilon:
                    rounds+=1
                else:
                    rounds = 0

                if rounds == max_rounds:
                    break
                f1_prev = f1
            
        self.plot_classifcacion(epocas)
        if flag:
            self.plot_classifcacion(epocas_val)

    def back_propagation1(self,X,Y,epochs,X_val,Y_val,epsilon,max_rounds):
        
        flag = (len(X_val) != 0)
        mse_prev = 9999
        rounds = -1
        epocas = {}             
        X_temp = np.transpose(X)
        Y_temp = np.transpose(Y)
        
        if flag:
            epocas_val = {}
            X_val_temp = np.transpose(X_val)
            Y_val_temp = np.transpose(Y_val)
        
        for i in range(epochs):
            predic = []
            original = []
            for j in range(len(X_temp)):
                X_eval = X_temp[j].reshape(len(X_temp[j]),1)
                Y_eval = Y_temp[j].reshape(len(Y_temp[j]),1)
                AL, caches = self.L_model_forward(X_eval)
                grads = self.L_model_backward(AL, Y_eval, caches)
                self.update_parameters(grads)
                predic.append(AL.reshape(len(AL)))
                original.append(Y_eval.reshape(len(Y_eval)))
            if flag:
                predic_val = []
                original_val = []
                for j in range(len(X_val_temp)):
                    X_eval_val = X_val_temp[j].reshape(len(X_val_temp[j]),1)
                    Y_eval_val = Y_val_temp[j].reshape(len(Y_val_temp[j]),1)
                    AL_val, _ = self.L_model_forward(X_eval_val)
                    predic_val.append(AL_val.reshape(len(AL_val)))                    
                    original_val.append(Y_eval_val.reshape(len(Y_eval_val)))
                    
                mse = mean_squared_error(original_val,predic_val)            
            else:
                mse = mean_squared_error(original,predic)


            epocas[str(i)] = [predic,original]
            if flag:
                epocas_val[str(i)] = [predic_val,original_val]

            if epsilon != -1:
                if abs(mse - mse_prev) < epsilon:
                    rounds+=1
                else:
                    rounds = 0

                if rounds == max_rounds:
                    break
                mse_prev = mse
            
    
        self.plot_mse(epocas)
        if flag:
            self.plot_mse(epocas_val)
    
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
    
    def plot_classifcacion(self,epocas):
        cont=-1
        x=[]
        yf1=[]
        yacc=[]
        for i in epocas:
            matrix1=self.get_coded_y(epocas[i][0])
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
        
        x_t=[]
        y_t=[]
        ymin=[]
        ymax=[]
        cont =-1
        

        for i in epocas:
            matriz1=epocas[i][0]
            matriz2=epocas[i][1]
            #Minimun and maximun

            list1 = []
            for j in range(len(matriz1)):
                #print(matriz1[j][k])
                temp = mean_squared_error(matriz2[j], matriz1[j])
                list1.append(temp)

            if cont == -1:
                y_t.append(mean_squared_error(matriz2, matriz1))
                x_t.append(i)
                ymin.append(min(list1))
                ymax.append(max(list1))

                cont = 0

            if cont == 1:
                y_t.append(mean_squared_error(matriz2, matriz1))
                ymin.append(min(list1))
                ymax.append(max(list1))
                x_t.append(i)
                cont = 0
            cont = cont+1
            
                        
            #y_t.append(mean_squared_error(matriz2, matriz1))
            #x_t.append(i)
            
        plt.plot(x_t, y_t, label="MSE")
        plt.plot(x_t, ymin, label="MSE minumun")
        plt.plot(x_t, ymax, label="MSE maximun")
        plt.legend()
        plt.title("MSE")

        plt.show()

    def load_data_P2(self,file):
        data=pd.read_csv(file)
        array1 = np.array([data['x1'].values, data['x2'].values])
        array2 = np.array([data['y1'].values, data['y2'].values])

        return array1,array2

    def load_params(self,file_name):
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
        self.parameters = params
         
    def load_data_P4(self,file):
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

            return np.transpose(scaled), np.transpose(Y.reshape(Y.shape[0],1))

    def load_data_P3(self,file_name):
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

