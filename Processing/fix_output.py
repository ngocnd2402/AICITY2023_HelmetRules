# This file to drop some output which has boundingbox extending outside the image

img_size = (1920, 1080) # (width, height)

rf = open('result.txt')
wf= open('new-result.txt', 'w')
lines = rf.readlines()
for line in lines:
    video_id, frame_id, tl_x, tl_y, w, h, cls, conf = [float(i) for i in line.split(',')] 
    video_id = int(video_id)
    frame_id = int(frame_id)
    cls = int(cls)
    br_x = tl_x + w
    br_y = tl_y + h
    tl_x = min(img_size[0], max(0, tl_x))
    br_x = min(img_size[0], max(0, br_x))    
    tl_y = min(img_size[1], max(0, tl_y))
    br_y = min(img_size[1], max(0, br_y))
    w = br_x - tl_x 
    h = br_y - tl_y 
    # Skip if w, h < 1
    if w < 1 or h < 1:
        continue 
    wf.write(f'{video_id},{frame_id},{tl_x},{tl_y},{w},{h},{cls},{conf}\n')