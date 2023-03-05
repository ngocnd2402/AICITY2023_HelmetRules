from ultralytics import YOLO
from IPython.display import display, Image
import os
import cv2
TEST_IMAGE_PATH = '/workspace/web-model/yolov8/test/images'
INFER_IMAGE_PATH = '/workspace/web-model/yolov8/infer'
model = YOLO(r"/workspace/model-web/yolov8/runs/detect/train/weights/best.pt")
results = model.predict(source = r'/workspace/model-web/deep_sort/data/test/096/img', conf=0.7, save = True)


