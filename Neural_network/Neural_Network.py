# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import Back_Propagation as BP
import Forward_Propagation as FP
import Import_Data as ID
import Iter_Fig_Method
import random


def Predict_accuracy(W, x_test, y_test, layer_units):
    W_list = FP.Reshape_W(W, layer_units)
    y = FP.FP_result(x_test, W_list)   #forward propagation to get result
    y_predict = np.zeros((y.shape[0],1))
    for i in range(y.shape[0]):
        y_predict[i] = np.unravel_index(np.argmax(y[i,:]), y[i,:].shape)[0] #
    #Count the correct result
    count = 0 
    for i in range(y_test.shape[0]):
        if y_predict[i] == y_test[i]:
            count = count +1
    print('accuracy = %.2f%%' % (count/(y_test.size)*100))  
    #show result
    f, ax = plt.subplots(3, 10)
    f.suptitle('accuracy = %.2f%%' % (count/(y_test.size)*100))
    ax = ax.flatten()
    for i in range(30):
        num = random.randrange(0, 1000, 1)
        font_c = "Greys" if y_predict[num] == y_test[num] else "Reds"
        ax[i].imshow(x_test[num].reshape(28, 28), cmap = font_c)
        ax[i].set_title("r = %d" % y_predict[num])
    plt.show()

if __name__ == "__main__":
    #define the number of units for each layer
    layer_units = [784, 32, 10] 

    #load the data: train data and test data
    x_Set, y_Set_multi, x_test, y_test_multi =  ID.Init_set(train_images_num = 4000, test_images_num = 1000)
    y_Set = np.eye(10)[y_Set_multi.reshape(-1)]  #example: transform 2 to [0,0,1,0,0,0,0,0,0,0]

    #randomly init parameters
    W = FP.Init_W(layer_units)

    #Iterative method
    W_best = Iter_Fig_Method.iter(W, x_Set, y_Set, layer_units)

    #Cal accuracy
    Predict_accuracy(W_best, x_test, y_test_multi, layer_units)



    