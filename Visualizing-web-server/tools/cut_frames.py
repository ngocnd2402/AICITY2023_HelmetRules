import cv2
import os
import glob
import numpy as np
import json

FRAME_PATH = '/workspace/web/visualizing-web/frames'
VIDEO_PATH = '/workspace/data/videos'
GT_PATH = '/workspace/data/gt.txt'
MP4_BBOX_VIDEO_PATH = '/workspace/web/visualizing-web/static/data/bbox-video/mp4'
WEBM_BBOX_VIDEO_PATH = '/workspace/web/visualizing-web/static/data/bbox-video/webm'

with open(GT_PATH) as f:
    lines = f.readlines()

gt = {}
for line in lines:
    vid_id, frame_id, track_id, x, y, w, h, cls = [int(x) for x in line.split(',')]

    if vid_id not in gt:
        gt[vid_id] = {}
    
    if frame_id not in gt[vid_id]:
        gt[vid_id][frame_id] = []
    
    start_point = (x, y)
    end_point = (x + w, y + h)

    gt[vid_id][frame_id].append(
        {
            'start_point': start_point,
            'end_point': end_point,
            'class': cls 
        }
    )


# 1, motorbike
# 2, DHelmet
# 3, DNoHelmet
# 4, P1Helmet
# 5, P1NoHelmet
# 6, P2Helmet
# 7, P2NoHelmet
# BGR
colors = [
    (233, 27, 233), # Pink
    (233, 41, 27), # Blue
    (25, 32, 218), # Red
    (196, 149, 77), # Light blue
    (3, 103, 191), # Orange
    (20, 236, 20), # Green
    (0, 247, 255) # Yellow
    ] 

thickness = 2

label_names = [
    'motorbike',
    'DHelmet',
    'DNoHelmet',
    'P1Helmet',
    'P1NoHelmet',
    'P2Helmet',
    'P2NoHelmet'
]

# open gt.json:
# gt_j = open('./gt.json','r')
# gt_j = json.load(gt_j)
# print(gt_j)

# Cut frames and draw bounding boxes
videos = os.listdir(VIDEO_PATH)

# for video in sorted(videos):
#     video_name = video.split('.')[0]
#     frame_path = f'{FRAME_PATH}/{video_name}'
#     if not os.path.exists(frame_path):
#         os.mkdir(frame_path)

#     vidcap = cv2.VideoCapture(f'{VIDEO_PATH}/{video}')
#     success,image = vidcap.read()
#     success,image = vidcap.read() # Drop first frame
#     count = 1
#     vid_id = int(video.split('.')[0])
#     img_array = []

#     while success:
        
#         bboxes = gt[vid_id][count]

#         for bbox in bboxes:
#             start_point = bbox['start_point']
#             end_point = bbox['end_point']
#             cls = bbox['class']

#             image = cv2.rectangle(image, start_point, end_point, colors[cls-1], thickness)   

#         cv2.imwrite(f'{frame_path}/{count:04}.jpg', image)     # save frame as JPEG file   
#         img_array.append(image)

#         success,image = vidcap.read()
#         print('Read a new frame: ', success)
#         count += 1
    
#     shape = (img_array[0].shape[1], img_array[0].shape[0])
#     out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, shape)
#     print(len(img_array))
#     for i in range(len(img_array)):
#         out.write(img_array[i])
#     out.release()

#     if count + 1 % 5 == 0:
#         print(count)
#     count += 1

#     break

for video in sorted(videos):
    video_name = video.split('.')[0]

    frame_path = f'{FRAME_PATH}/{video_name}'
    if not os.path.exists(frame_path):
        os.mkdir(frame_path)

    vidcap = cv2.VideoCapture(f'{VIDEO_PATH}/{video}')
    success1,image1 = vidcap.read()
    success2,image2 = vidcap.read()
    count = 1
    vid_id = int(video.split('.')[0])
    img_array = []
    print(f'video_id: {vid_id}\tn_frames: {int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1}')
    
    while success2:
        if count in gt[vid_id]:
            bboxes = gt[vid_id][count]
            for bbox in bboxes:
                start_point = bbox['start_point']
                end_point = bbox['end_point']
                cls = bbox['class']

                image1 = cv2.rectangle(image1, start_point, end_point, colors[cls-1], thickness)   

        cv2.imwrite(f'{frame_path}/{count:04}.jpg', image1)     # save frame as JPEG file   
        img_array.append(image1)

        success1 = success2
        image1 = image2
        success2,image2 = vidcap.read()
        # print('Read a new frame: ', success2)
        count += 1
    
    shape = (img_array[0].shape[1], img_array[0].shape[0])
    
    out = cv2.VideoWriter(f'{MP4_BBOX_VIDEO_PATH}/{video}', cv2.VideoWriter_fourcc(*'mp4v'), 20, shape)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    webm_video = video.split('.')[0] + '.webm'
    os.system(f'/usr/bin/ffmpeg  -i {MP4_BBOX_VIDEO_PATH}/{video} -b:v 0  -crf 30  -pass 1  -an -f webm -y /dev/null')
    os.system(f'/usr/bin/ffmpeg  -i {MP4_BBOX_VIDEO_PATH}/{video} -b:v 0  -crf 30  -pass 2  {WEBM_BBOX_VIDEO_PATH}/{webm_video}')


