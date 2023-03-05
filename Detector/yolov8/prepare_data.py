# 1: motorbike
# 2: helmet
# 3: no helmet

import cv2
import os
import glob
import numpy as np
import json

# FRAME_PATH = r'/workspace/web/visualizing-web/frames'
# VIDEO_PATH = r'/workspace/data/videos'
# GT_PATH = r'/workspace/data/gt.txt'
# GT_YOLOV8_PATH = r'/workspace/model-web/yolov8/gt_yolov8.txt'
# TEST_IMAGE_PATH = r'/workspace/model-web/yolov8/test/images'
# TEST_LABEL_PATH= r'/workspace/model-web/yolov8/test/labels'
# TRAIN_IMAGE_PATH = r'/workspace/model-web/yolov8/train/images'
# TRAIN_LABEL_PATH= r'/workspace/model-web/yolov8/train/labels'

TRAIN_IMAGE_PATH = r'/mlcv/WorkingSpace/Personals/sangdn/AIC2023/yolov8/train/images'
TRAIN_LABEL_PATH = '/mlcv/WorkingSpace/Personals/sangdn/AIC2023/yolov8/train/labels' 
TEST_IMAGE_PATH = r'/mlcv/WorkingSpace/Personals/sangdn/AIC2023/yolov8/test/images'
TEST_LABEL_PATH = r'/mlcv/WorkingSpace/Personals/sangdn/AIC2023/yolov8/test/labels'
VAL_IMAGE_PATH = r'/mlcv/WorkingSpace/Personals/sangdn/AIC2023/yolov8/valid/images'
VAL_LABEL_PATH = r'/mlcv/WorkingSpace/Personals/sangdn/AIC2023/yolov8/valid/labels'
VIDEO_PATH = r'/mlcv/WorkingSpace/Personals/sangdn/AIC2023/videos'
GT_PATH = '/mlcv/Databases/NvidiaAIC2023/aicity2023_track5/gt.txt'

helmet_ids = [2, 4, 6]
no_helmet_ids = [3, 5, 7]
motorbike_id = 1

with open(GT_PATH) as f:
    lines = f.readlines()

gt = {}
for line in lines:
    vid_id, frame_id, track_id, x, y, w, h, cls = [int(x) for x in line.split(',')]

    if vid_id not in gt:
        gt[vid_id] = {}
    
    if frame_id not in gt[vid_id]:
        gt[vid_id][frame_id] = []


    gt[vid_id][frame_id].append(
        {
            'x_left': x,
            'y_top': y,
            'w': w,
            'h': h,
            'class_id': cls
        }
    )

videos = os.listdir(VIDEO_PATH)
count = 1
for video in sorted(videos):
    video_name = video.split('.')[0]
    video_id = int(video_name)
    
    vidcap = cv2.VideoCapture(f'{VIDEO_PATH}/{video}')
    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    w_frame = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h_frame = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print(f'video_id: {vid_id}\tn_frames: {int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1}')

    for frame_id in range(1, n_frames + 1):
        ret, frame = vidcap.read()

        # Open file to write
        if video_id <= 70:
            f = open(f'{TRAIN_LABEL_PATH}/{count:05}.txt', 'w')
        elif video_id>70 and video_id<=90:
            f = open(f'{VAL_LABEL_PATH}/{count:05}.txt', 'w')
        else: 
            f = open(f'{TEST_LABEL_PATH}/{count:05}.txt', 'w')

        # If a frame has no bboxes *.txt will be empty
        if video_id in gt and frame_id in gt[video_id]:
            bboxes = gt[video_id][frame_id]
            for bbox in bboxes:
                x_left = bbox['x_left']
                y_top = bbox['y_top']
                w_bb = bbox['w']
                h_bb = bbox['h']
                class_id = bbox['class_id'] 

                x_center_norm = (x_left + w_bb / 2) / w_frame
                y_center_norm = (y_top + h_bb / 2) / h_frame
                w_norm = w_bb / w_frame 
                h_norm = h_bb / h_frame
                if class_id in helmet_ids: # class_id = 1 is motorbike
                    class_id = 1
                elif class_id in no_helmet_ids:
                    class_id = 2
                elif class_id == motorbike_id:
                    class_id = 0

                f.write(f'{class_id} {x_center_norm} {y_center_norm} {w_norm} {h_norm}\n')
            f.close()

        # Cut frame
        if video_id<=70:
            cv2.imwrite(f'{TRAIN_IMAGE_PATH}/{count:05}.jpg',frame)
        elif video_id>70 and video_id<=90:
            cv2.imwrite(f'{VAL_IMAGE_PATH}/{count:05}.jpg',frame)
        else:
            cv2.imwrite(f'{TEST_IMAGE_PATH}/{count:05}.jpg',frame)
        count+=1
        print(f'Successfully captured {count} images')
    
    