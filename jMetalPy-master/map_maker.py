from PIL import Image
import cv2
import os
import numpy as np
import shapefile
import pngcanvas

shapefile_name = 'map.shp'

r = shapefile.Reader(shapefile_name)
#1737
xdist = r. bbox[2] - r. bbox[0]
ydist = r. bbox[3] - r. bbox[1]
print(xdist)
print(ydist)
xyratio = xdist/ydist
image_max_dimension = 14400
if (xyratio >= 1):
    iwidth  = image_max_dimension
    iheight = int(image_max_dimension/xyratio)
else:
    iwidth  = int(image_max_dimension*xyratio)
    iheight = image_max_dimension

xratio = iwidth/xdist
yratio = iheight/ydist
pixels = []
c = pngcanvas.PNGCanvas(iwidth,iheight)
for shape in r.shapes():
    for x,y in shape.points:
        px = int(iwidth - ((r.bbox[2] - x) * xratio))
        py = int((r.bbox[3] - y) * yratio)
        pixels.append([px,py])
    c.polyline(pixels)
    pixels = []


f = open("123.png", "wb")
f.write(c.dump())
f.close()

image = cv2.imread('123.png', cv2.IMREAD_COLOR)
os.remove('123.png')
mapImage = np.zeros((3472, 7200), dtype = image.dtype)
newImage = np.zeros((3472, 3600), dtype = image.dtype)
map2Image = np.zeros((128, 10800), dtype = image.dtype)


map_height, map_width = 2, 2

for j in range(3472):
    for i in range(7200):
        x = i*map_width
        y = j*map_height

        pixel = image[y:y+map_height, x:x+map_width]
        mapImage[j,i] = cv2.mean(pixel)[0]

for j in range(3472):
    for i in range(3600):
        x = i*map_width
        y = j*map_height

        pixel = image[y:y+map_height, x:x+map_width]
        newImage[j,i] = cv2.mean(pixel)[0]

mapImage = np.where(mapImage<255,int(0),mapImage)
mapImage = np.where(mapImage>=255,int(255),mapImage)

newImage = np.where(newImage<255,int(0),newImage)
newImage = np.where(newImage>=255,int(255),newImage)

MapImage = np.c_[mapImage, newImage]

final_map = np.r_[map2Image, MapImage]

pil_image = Image.fromarray(final_map)
pil_image.save('map_F.png')