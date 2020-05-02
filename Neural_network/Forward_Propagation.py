import numpy as np

activation_fucntion_output = "sig" # sig\tanh\reLU
activation_fucntion_hidden = "sig" # sig\tanh\reLU

"""
Function: gFunction
Description: activation function
Input: z
Return: a
Others: 
"""
def gFunction(z, myFunction = 'sig'):
    if(myFunction == 'sig'):
        a = 1/(1 + np.exp(-z))  #sigmod function
    elif(myFunction == 'tanh'):
        a = (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))  #tanh function
    elif(myFunction == 'reLU'):
        a = (np.abs(z) + z)/2.0          #reLU function
    else:
        raise NameError('Activation fucnation name is wrong')
    return a
"""
Function: gFunction_D
Description: derivative of activation function
Input: z
Return: a
Others: 
"""
def gFunction_D(a, myFunction = 'sig'):
    if(myFunction == 'sig'):
        a = a*(1-a)  #sigmod function
    elif(myFunction == 'tanh'):
        a = 1-a*a  #tanh function
    elif(myFunction == 'reLU'):
        a = np.where(a > 0, 1, 0) #reLU function
    else:
        raise NameError('Activation fucnation name is wrong')
    return a

"""
Function: Init_W
Description: 
Input: 
    layers: the number of units for each layer, example：[784, 25, 10]
    w_range: example [-0.1, 0.1]
Return: 
    W: paras
Others: 
"""
def Init_W(layers,  w_range = [-0.12,0.12]):
    layer_num = len(layers)
    W,W_list = np.zeros((1,1)),[]
    for i in range(layer_num -1):
        W_simple = np.random.uniform(w_range[0],w_range[1],(layers[i+1], layers[i]+1))
        W_list.append(W_simple)
        if(W.size == 1):
            W = W_simple.reshape((W_simple.size,1))
        else:
            W = np.vstack((W, W_simple.reshape((W_simple.size,1))))

    return W
"""
Function: Reshape_W
Description: 
Input: 
    W: paras
    layers: the number of units for each layer, example：[784, 25, 10]
Return: 
    W_list: paras for each layer
Others: 
"""
def Reshape_W(W, layers):
    layer_num = len(layers) 
    W_num = [0]
    W_list = []
    for i in range(layer_num -1):
        row = layers[i+1]
        column = layers[i]+1
        W_num.append(row * column)
        start  =  np.sum(W_num[:i+1])
        end = np.sum(W_num[:i+2])
        W_list.append(W[start:end].reshape((row,column)))
    return W_list

"""
Function: FP
Description: 
Input: 
    x_vector: 
    W_list: 
Return: 
    z_list: 
    a_list
Others: 
"""
def FP(x_vector, W_list):
    z_list = [0]
    a_list = [x_vector]

    for i in range(len(W_list)):    #times of forward propagation
        a_list[i] = np.insert(a_list[i], 0, 1, axis = 1)  #a0 = 1
        z = a_list[i] @ W_list[i].T
        if(i == (len(W_list)-1)):   
            a = gFunction(z, activation_fucntion_output)    #activation_fucntion of output layer ，sig
        else:
            a = gFunction(z, activation_fucntion_hidden)    #activation_fucntion of hidden layer ，sig
        z_list.append(z)
        a_list.append(a)

    return z_list, a_list

"""
Function: FP_result
Description: 
Input: 
    x_vector: 
    W_list: 
Return: 
Others: 
"""
def FP_result(x_vector, W_list):
    z_list, a_list = FP(x_vector, W_list)
    return a_list[len(W_list)]


if __name__ == "__main__":
    layers = [784, 25, 10] 
    W = Init_W(layers)
    W_list = Reshape_W(W, layers)