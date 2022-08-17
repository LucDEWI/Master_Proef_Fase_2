import cv2
import numpy as np
import cv2 as cv
from src import Functions as func
import os
# deze functie toont de beelden van de kleur en diepte camera
# dit geeft ook een aanduiding van de bounding box in het beeld om de kalibratie te controleren

#resolutie en fps van de beelden
cres = (1920, 1080)
dres = (1280, 720)
fps = 30


#connectie starten
connection, align = func.start(cres, dres, fps)

datapath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\data'
box = np.load(datapath+'\e_box.npy')
x = box[0]
y = box[1]
w = box[2]
h = box[3]
print(x, y, w, h)
#loop voor beelden
while True:
    #frames ophalen
    color, depth = func.getframes(connection, align)
    cv2.rectangle(color, (x+50, y+50), (x + w-50, y + h-75), (0, 255, 0), 2)
    #color = color[box[1]:box[0]+box[2], box[1]:box[1]+box[3]]
    #color = cv2.resize(color,(300, 300), interpolation= cv2.INTER_LINEAR)
    #dieptebeeld inkleuren
    depth_color = cv.applyColorMap(cv.convertScaleAbs(depth, alpha=0.03), cv.COLORMAP_JET)


    #beelden naast elkaar zetten
    #img = np.hstack((color, depth_color))
    cv2.imshow('', color)
    if cv.waitKey(1) == ord('q'):
        break



