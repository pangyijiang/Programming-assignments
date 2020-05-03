import random
import numpy as np

class AntColony:
    global_p = 0.002 # pheromone evaporation
    ants = []

    def __init__(self, map, num_ants = 200):
        self.map = map
        for i in range(num_ants):
            self.ants.append(Ant(self.map, self.map.Ants_Nest, i))

    def step(self, matrix):
        for ant in self.ants:
            degree = ant.step(matrix)
            self.map.show_ant(ant.pos[0], ant.pos[1], degree, 127)
        self._pheromone_evaporation(matrix)

    def _pheromone_evaporation(self, matrix):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i][j] > 0.0:	
                    matrix[i][j] *= (1 - self.global_p)

class Ant:
    alpha = 2 # weight of feromone
    beta = 3 # weight of path len
    Q = 300 # relase feromone density
    
    def __init__(self, map, init_pos, id):
        self.map = map
        self.id = id
        self.dir = np.array([[0, -1],[1, -1],[1, 0],[1, 1],[0, 1],[-1, 1],[-1, 0],[-1, -1],])
        self.init_pos = init_pos
        self._init_paras(init_pos)
        
    def _init_paras(self, pos):
        self.pos = pos
        self.back_nest = False
        self.path_len = 0
        self.life = 1000
        self.traveled_road = []


    def get_feromone(self, dir, matrix):

        X = self.pos[0] + dir[0]
        Y = self.pos[1] + dir[1]

        if X <= (self.map.map_matrix.shape[0]-1) and Y <= (self.map.map_matrix.shape[1]-1) and X >= 0 and Y >= 0:
            if matrix[X][Y] > 0.0:	
                if [X, Y] not in self.traveled_road:		
                    feromone =  matrix[X][Y]
                else:
                    feromone = matrix[X][Y]/8 #np.min(np.abs(matrix))/2
            elif matrix[X][Y] == self.map.map_mark["obstacle"]:
                feromone =  0.0
            elif matrix[X][Y] == self.map.map_mark["spawn"]:
                feromone =  0.0
            elif matrix[X][Y] == self.map.map_mark["food"]:
                feromone =  np.max(matrix)*10 #self.init_feromone*100 
            else:
                raise Exception("Value Error: matrix[%d][%d] = %.2f" % (X, Y, matrix[X][Y]))
            
        else:
            feromone =  0.0 #out of map
        return feromone

    def get_inverse_distance(self, dir):
        if np.abs(np.sum(dir)) == 1:
            return 1.0 
        else:
            return (1 / 1.414) 

    #die or found food
    def respawn(self, cause):
        self._init_paras(self.init_pos)
        print("Ant ID %03d respawn: %s" % (self.id, cause))

    def move(self, dir):
        if [self.pos[0], self.pos[1]] not in self.traveled_road:
            self.traveled_road.append([self.pos[0], self.pos[1]])

        self.pos = self.pos + dir

        if [self.pos[0], self.pos[1]] not in self.traveled_road:
            self.traveled_road.append([self.pos[0], self.pos[1]])

        if np.any(dir == 0):
            self.path_len += 1
        else:
            self.path_len += 1.414

	#Ant take one step
    def step(self, matrix):
        pos_prv = self.pos
        if not self.back_nest:
            if self.life < 0:
                #can't find food, be dead
                self.respawn("die of no food")
            else:
                self.life -= 1
                summ = 0
                probabilities = []
                for dir in self.dir:
                    summ += self.get_inverse_distance(dir) ** self.beta * self.get_feromone(dir, matrix) ** self.alpha
                for dir in self.dir:
                    probabilities.append(self.get_inverse_distance(dir) ** self.beta * self.get_feromone(dir, matrix) ** self.alpha / summ )

                dir_num = np.random.choice([0,1,2,3,4,5,6,7], p = probabilities)
                self.move(self.dir[dir_num])

                if (matrix[self.pos[0]][self.pos[1]] == self.map.map_mark["food"]):
                    print("Ant ID %03d found food in (%.1f, %.1f)" % (self.id, self.pos[0], self.pos[1]))
                    self.back_nest = True
                
        else: #go back to neast, put feromone on path

            for i,road in enumerate(self.traveled_road):
                if matrix[road[0]][road[1]] > 0.0:
                    matrix[road[0]][road[1]] = matrix[road[0]][road[1]] + (self.Q / self.path_len)#*(i + 1)/(steps)
            self.respawn("successfully relase feromone and bring back food")

        dir = self.pos - pos_prv
        return np.arctan2(dir[1], dir[0])/(2*np.pi)*360