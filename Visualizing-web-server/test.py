import cv2
import os
import glob
import numpy as np
import json

image = cv2.imread('frames/001/0001.jpg')
shape = (image.shape[1], image.shape[0])
out = cv2.VideoWriter(f'test.webm', cv2.VideoWriter_fourcc(*'vp80'), 30, shape)
for path in os.listdir('frames/001'):
    image = cv2.imread(f'frames/001/{path}')
    out.write(image)
out.release

