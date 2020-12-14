
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import nicolle as nicky



layers_dims = (2, 2, 2)

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 100, print_cost=False):#lr was 0.009

    np.random.seed(1)
    costs = []                             
    parameters = nicky.initialize_parameters_deep(layers_dims)    
        
    for i in range(0, num_iterations):
        AL, caches = nicky.L_model_forward(X, parameters)
        
        cost = nicky.compute_cost(AL, Y)
                    
        grads = nicky.L_model_backward(AL, Y, caches)
                 
        parameters = nicky.update_parameters(parameters, grads, learning_rate)
                                
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
                
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


parameters = L_layer_model(np.array([[0,0,1,1],[0,1,0,1]]), np.array([[0,0,0,1],[0,1,1,0]]), [2,2,2], num_iterations = 1000, print_cost = True)

print(parameters)

'''
parameters = {'W1': np.array([[ 0.5,  0.5],
       [-0.5, -0.5]]), 'b1': np.array([[0.5],
       [-0.5]]), 'W2': np.array([[0.25,  0.25],
       [-0.25, -0.25]]), 'b2': np.array([[0.25],
       [ -0.25]])}

'''
#{'W1': array([[ 0.5,  0.5],
#       [-0.5, -0.5]]), 'b1': array([ 0.5, -0.5]), 'W2': array([[ 0.25,  0.25],
#       [-0.25, -0.25]]), 'b2': array([ 0.25, -0.25])}


XA,_ = nicky.L_model_forward(np.array([[0],[0]]), parameters)

print(XA)





