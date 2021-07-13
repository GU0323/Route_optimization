import cv2
import numpy as np

image = cv2.imread('./resources/mapImage/worldmap.png', cv2.IMREAD_COLOR)

map = np.zeros((1800, 3600), dtype = image.dtype)

map_height, map_width = 2, 2

for j in range(1800):
    for i in range(3600):
        x = i*map_width
        y = j*map_height

        pixel = image[y:y+map_height, x:x+map_width]
        map[j,i] = cv2.mean(pixel)[0]

map = np.where(map<255,int(1),map)
map = np.where(map>=255,int(0),map)

np.save('maparray', map)