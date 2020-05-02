import pygame as pg
from Map import Map as MAP
from AntColony import AntColony as AC

if __name__ == "__main__":
	map = MAP()
	ac = AC(map, num_ants = 500)
	
	flag_running = True
	while flag_running:
		flag_running = map.event()
		map.show_background()
		ac.step(map.map_matrix)
		map._pg_update()
	map.quit()