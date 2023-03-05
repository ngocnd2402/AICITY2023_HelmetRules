from ultralytics import YOLO
from PIL import Image
import cv2
import os 
from strongsort import StrongSORT
from pathlib import Path
import numpy as np 
import torch

VIDEO_PATH = 'aicity2023_track5_test/videos'

model = YOLO("runs/detect/train2/weights/best.pt")
tracker = StrongSORT(model_weights=Path('resnet50_fc512_dukemtmcreid.pt'), device='cuda', fp16=False)

for video in sorted(os.listdir(VIDEO_PATH)):
    print(f'Processing video {video}')
    video_id = int(video.split('.')[0])
    video_path = os.path.join(VIDEO_PATH, video)
    vidcap = cv2.VideoCapture(video_path)
    success, img = vidcap.read()
    frame_idx = 1

    f = open('result.txt', 'a')

    while success:
        # Yolo
        results = model.predict(source=img)
        results = results[0].cpu()
        boxes = results.boxes.xyxy
        conf = results.boxes.conf.view(-1, 1)
        cls = results.boxes.cls.view(-1, 1)
        detections = torch.cat((boxes, conf, cls), -1)
        # Tracker
        tracks = tracker.update(detections, img, frame_idx) # detection in format (xyxy, conf, cls)
        for track in tracks:
            frame_idx, track_id, x, y, w, h, conf, class_id = track
            f.write(f'{video_id},{frame_idx},{x},{y},{w},{h},{class_id},{conf}\n')
        # Read new img from video
        frame_idx += 1
        success, img = vidcap.read()

    f.close()