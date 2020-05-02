
import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img # process img

 

class K_Means():
    """
        parameter:  x_set: point set, 
                    num_K: number of cluster, 
                    max_iter: max number of iteration, 
                    accuracy: quit loop when done
    """
    def __init__(self, x_set, num_K = 0):
        assert num_K!=0
        self.num_K = num_K
        self.x_set = x_set
        x_set_temp = x_set.reshape(-1,x_set.shape[-1])
        self.ccenter_K = np.array(random.sample(list(x_set_temp), self.num_K))  #randomly choose cluster center
        self.cset_K = [[] for i in range(self.num_K)]   #init cluster set
        #self.max_iter = max_iter
        #self.accuracy = accuracy
    def step(self):
        ccenter_K_last = copy.deepcopy(self.ccenter_K)             #last time value of ccenter_K，to calculate (ccenter_K_last - ccenter_K)
        distance = [-1 for i in range(self.num_K)]    #init distance of each point to K centers, temp parameters
        #start clustering
        #for iteration in range(self.max_iter):
            #empty each cluster of self.cset_K
        for i in range(self.num_K):
            self.cset_K[i].clear()
        #cluster points based on the distance between point and self.ccenter_K
        # for x_row in range(self.x_set.shape[0]):
        #     for x_list in range(self.x_set[x_row].shape[0]):
        for x in self.x_set:
            for num,K in enumerate(self.ccenter_K):
                distance[num] = np.sum(np.square(x - K))
            index_min = distance.index(min(distance))
            self.cset_K[index_min].append(x)

        #average points in self.cset_K to get self.ccenter_K
        for i in range(self.ccenter_K.shape[0]):
            self.ccenter_K[i] = np.mean(self.cset_K[i], axis=0)    #axis = 0, average of row

        #calculate the error of (ccenter_K_last - self.ccenter_K)
        # if(np.mean(ccenter_K_last - self.ccenter_K) < self.accuracy):
        #     print("The accuracy larger than %f" % self.accuracy)
        #     #print("Cluster Center points: ", self.ccenter_K)
        #     #print("Cluster Center points (last time): ", ccenter_K_last)
        #     #return self.ccenter_K, self.cset_K
        # else:
        ccenter_K_last = copy.deepcopy(self.ccenter_K)
        #print("iter = ", iteration)
        #print("iteration dene, quit")
        return self.ccenter_K, self.cset_K

if  __name__ == "__main__":
    #load data
    num_K = 3
    data_size = 50
    k_set = [[np.random.uniform(0, 10),np.random.uniform(0, 10)] for i in range(num_K)]
    x_set = [[np.random.uniform(0, 1) + k_set[j][0], np.random.uniform(0, 1) + k_set[j][1]] for i in range(data_size) for j in range(num_K)]
    x_set = np.array(x_set)

    #Init figure
    fig = plt.figure()
    fig_original = fig.add_subplot(121)
    plt.xlabel("X1"), plt.ylabel("X2")
    for x in x_set:
        fig_original.scatter(x[0],x[1], c = 'blue', marker = 'o')
    fig_result = fig.add_subplot(122)
    plt.xlabel("X1"), plt.ylabel("X2")
    plt.ion() 
    color_set = ["red","yellow","green","blue"]

    #Init k_means
    max_iter = 10
    k_means = K_Means(x_set, num_K)
    for i in range(max_iter):
        ccenter_K, cset_K = k_means.step()
        print("iter = %d" % i)
        #show result
        fig_result.cla()
        for i in range(ccenter_K.shape[0]):
            for x in cset_K[i]:
                fig_result.scatter(x[0], x[1], c = color_set[i], marker = 'o')
            fig_result.scatter(ccenter_K[i][0],ccenter_K[i][1], c = "black", marker = 'x')

        plt.pause(1.0)

    plt.show()
