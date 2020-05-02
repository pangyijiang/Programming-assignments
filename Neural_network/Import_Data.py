import ReadIDXFile
import numpy as np


def ReturnXList(file = 'train-images.idx3-ubyte', examplesNum = 100):
    #load data
    examples_num = examplesNum
    examples_file = file
    examples_List = ReadIDXFile.ReturnIdx3ImageList(idx3image_file = examples_file, requestImageNum = examples_num)
    #feature scaling
    for i in range(len(examples_List)):
        for j in range(len(examples_List[i])):
            examples_List[i][j] = float(examples_List[i][j])
            examples_List[i][j] = (examples_List[i][j] - 128)/256   #feature scaling: -0.5~0.5, x = (x-average)/range

    return examples_List


def ReturnYList(file = 'train-labels.idx1-ubyte', examplesNum = 100):
    #load data
    labels_num = examplesNum
    labels_file = file
    labels_List = ReadIDXFile.ReturnIdx3ImageListLabels(idx3imagelabels_file = labels_file, requestImageNum = labels_num)

    return labels_List
"""
Function: Init_set
Description: 
Input: 
Return: 
    x_Set: shape(num of example，dimension of x)
    y_Set: shape(num of example, dimension of result)
    x_test: shape(num of example，dimension of x)
    y_test: shape(num of example, dimension of result)
Others: 
"""
def Init_set(train_images_num = 3000, test_images_num = 1000):
    #train_images_num = 3000
    x_Set = np.array(ReturnXList(file = 'train-images.idx3-ubyte', examplesNum = train_images_num))
    y_Set = np.array(ReturnYList(file = 'train-labels.idx1-ubyte', examplesNum = train_images_num)) 

    #train_images_num = 1000
    x_test = np.array(ReturnXList(file = 't10k-images.idx3-ubyte', examplesNum = test_images_num))
    y_test = np.array(ReturnYList(file = 't10k-labels.idx1-ubyte', examplesNum = test_images_num)) 

    print("shape of x_Set:", x_Set.shape, "shape of y_Set:", y_Set.shape)
    print("shape of x_test:", x_test.shape, "shape of y_test:", y_test.shape)
    return x_Set, y_Set, x_test, y_test