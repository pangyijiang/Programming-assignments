import pygame as pg
import numpy as np
import random
from PIL import Image

class Map():
    init_feromone = 1.0 #zero is hard for ant to select next step (sum = 0)
    map_mark = {"spawn":-100, "food":-101, "obstacle":-102}
    Ants_Nest = np.array([0, 0])
    color_nest = (0, 0, 255)
    scale = 6
    img_map = "map.jpg"
    img_ant = "ant.png"
    
    def __init__(self):
        self.map_matrix = self._gen_map_matrix(self.img_map)
        self._init_pgscreen()

    def _init_pgscreen(self):
        # Initialise screen
        pg.init()
        self.screen = pg.display.set_mode([int(self.map_matrix.shape[0]*self.scale), int(self.map_matrix.shape[1]*self.scale)])
        pg.display.set_caption("Ant Colony Algorithm")
        #load elements
        self.ant_img = pg.image.load(self.img_ant)
        self.ant_img = self.ant_img.convert_alpha()
        self.bg_img = pg.image.load(self.img_map)
        self.bg_img = self.bg_img.convert()

        self.clock = pg.time.Clock()

    def _gen_map_matrix(self, img_path, init_feromone = 1.0):
        img = Image.open(img_path)
        width = int(img.size[0]/self.scale)
        height = int(img.size[1]/self.scale)
        img = img.resize((width, height))
        width = img.size[0]
        height = img.size[1]

        map_matrix = np.ones((width, height))*init_feromone

        acc = 10
        for w in range(width):
            for h in range(height):
                pixel = img.getpixel((w, h))   
                #[255, 255, 255]white is empty space
                if pixel[0] > 255 - acc and pixel[1] > 255 - acc and pixel[2] > 255 - acc:
                    map_matrix[w][h] = init_feromone
                # #[255, 0, 0]red is food
                elif pixel[0] > 255 - acc and pixel[1] < acc and pixel[2] < acc:
                    map_matrix[w][h] = self.map_mark["food"]
                # #[0, 0, 0]black is obstacle
                elif pixel[0] < acc and pixel[1] < acc and pixel[2] < acc:
                    map_matrix[w][h] = self.map_mark["obstacle"]

        map_matrix[self.Ants_Nest[0]][self.Ants_Nest[1]] = self.map_mark["spawn"]
        return map_matrix

    #show the nest and feromone level
    def show_point(self, pos_x, pos_y, color):
        s = pg.Surface((self.scale, self.scale))  # the size of your rect
        s.fill(color)           # this fills the entire surface
        self.screen.blit(s, (pos_x * self.scale, pos_y * self.scale))    # (0,0) are the top-left coordinates

    #show ants
    def show_ant(self, pos_x, pos_y, degree = 0, alpha = 255):
        ant_img = pg.transform.rotate(self.ant_img, degree)
        self.screen.blit(ant_img, (pos_x * self.scale, pos_y * self.scale))    # (0,0) are the top-left coordinates

    def show_background(self):
        self.screen.blit(self.bg_img, (0, 0))
        for i in range(self.map_matrix.shape[0]):
            for j in range(self.map_matrix.shape[1]):
                # if self.map_matrix[i][j] == self.map_mark["spawn"]:
                #     self.show_point(i, j, self.color_nest)
                #     pass
                # elif self.map_matrix[i][j] == self.map_mark["food"]:
                #     pass
                # elif self.map_matrix[i][j] == self.map_mark["obstacle"]:
                #     pass
                # else:
                
                if self.map_matrix[i][j] not in [self.map_mark["spawn"], self.map_mark["food"], self.map_mark["obstacle"]]:
                    level = 255 - (self.map_matrix[i][j] - self.init_feromone)
                    if level > 255: level = 255
                    if level < 0: level = 0
                    self.show_point(i, j, (255, 255, level))
        #show ant nest
        self.show_point(self.Ants_Nest[0], self.Ants_Nest[1], self.color_nest)

    def _pg_update(self):
        pg.display.flip()
        self.clock.tick(30)
        
    def event(self):
        flag_running = True
        events =  pg.event.get()

        for event in events:
            if event.type == pg.QUIT:
                flag_running = False
            #keyboard event
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                    self.flag_best_path = ~self.flag_best_path
                    print("P is pressed")

        return flag_running

    def quit(self):
        pg.quit()