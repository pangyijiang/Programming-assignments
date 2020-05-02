import matplotlib.pyplot as plt
import Back_Propagation as BP


"""
Function: iteration to optimal W
Description: 
Input: 
    W: paras (a row)
    x_set：shape(num of example，dimension of x)
    y_set：shape(num of example, dimension of result)
    layer_units: the number of units for each layer, example：[784, 25, 10]
Return: W
Others: 
"""
def iter(W, x_Set, y_Set, layer_units):
    iter = 700  #max iterations
    rate = 0.5  #learning rate

    fig = plt.figure()
    splt = fig.add_subplot(111)
    splt.set_title("Max iteraion = %d" % iter)
    splt.set_xlabel("iteration")
    splt.set_ylabel("Loss")
    plt.ion()   #turn on plot interation
    
    for i in range(iter):
        D = BP.BP(W,x_Set,y_Set,layer_units)
        error = BP.Cost_function(W,x_Set,y_Set,layer_units)
        D = D.reshape(D.size, 1)*rate   #to make shape(n,) -> shape(n,1)
        W = W - D*rate 
        splt.scatter(i,error,c = 'blue')
        plt.pause(0.01)

    plt.ioff() 
    return W