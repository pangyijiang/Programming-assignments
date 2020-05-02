import numpy as np
import Forward_Propagation as FP

"""
Function: Cost_function
Description: 
Input: 
    W: para of NN
    x_set：shape(num of example，dimension of x)
    y_set：shape(num of example, dimension of result)
    layers: the number of units for each layer, example：[784, 25, 10]
Return: error
Others: 
"""
def Cost_function(W, x_set, y_set, layers):
    W_list = FP.Reshape_W(W, layers)
    a_last = FP.FP_result(x_set, W_list)
    error = -np.multiply(y_set, np.log(a_last)) - np.multiply((1 - y_set), np.log(1-a_last))
    m = len(y_set)
    return np.sum(error)/m

"""
Function: BP
Description: 
Input: 
    W: para of NN
    x_set：shape(num of example，dimension of x)
    y_set：shape(num of example, dimension of result)
    layers: the number of units for each layer, example：[784, 25, 10]
Return: 
    delta_list: gradient
Others: 
"""
def BP(W, x_set, y_set, layers):
    layer_num = len(layers) #num of layer
    m = len(x_set) #example num
    W_list = FP.Reshape_W(W, layers)
    error_list,delta_list = [0 for i in range(layer_num-1)],[0 for i in range(layer_num-1)] #误差，梯度
    #forward propagation
    z_list, a_list = FP.FP(x_set, W_list)   #a_list is result

    #compute "error" and futuremore "gradient"
    for i in [layer_num-2-i for i in range(layer_num-1)]:   #example: i = [3,2,1,0]
        if(i == (layer_num-2)):
            error_list[i] = a_list[i+1] - y_set    #compute "error"
        else:
            error_list[i] = (error_list[i+1] @ W_list[i+1][:,1:]) * FP.gFunction_D(a_list[i+1][:,1:], FP.activation_fucntion_hidden)   #compute "error"
        delta_list[i] = (error_list[i].T @ a_list[i])/m   #compute "gradient"


    #delta_list -> delta(1 row)
    delta = 0  
    for i in range(len(delta_list)):
        if(i == 0):
            delta = delta_list[i].reshape((delta_list[i].size, 1))
        else:
            delta = np.vstack((delta, delta_list[i].reshape((delta_list[i].size, 1))))
    return delta.reshape(delta.size,)   #(N,) -> (N,1)
