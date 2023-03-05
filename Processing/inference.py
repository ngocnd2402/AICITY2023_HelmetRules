rf = open('result.txt')
wf= open('new-result.txt', 'w')
lines = rf.readlines()
for line in lines:
    video_id, frame_id, x, y, w, h, cls, conf = [float(i) for i in line.split(',')] 
    video_id = int(video_id)
    frame_id = int(frame_id)
    if x == -0:
        x = 0

    if x < 0:
        x = 0
        w = w + x
    if y < 0:
        y = 0
        h = h + y

    if x > 1920:
        continue

    if y > 1080:
        continue

    if 1920 - x < 1:
        continue 
    if 1080 - y < 1:
        continue 
        
    cls = int(cls)
    
    wf.write(f'{video_id},{frame_id},{min(max(x, 0),1920)},{min(max(y, 0),1080)},{w},{h},{cls},{conf}\n')