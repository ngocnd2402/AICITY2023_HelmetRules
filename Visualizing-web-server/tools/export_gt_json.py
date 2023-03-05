import json

FRAME_PATH = './frames'
VIDEO_PATH = '/workspace/data/videos'
GT_PATH = '/workspace/data/gt.txt'

# gt: ground truth
with open(GT_PATH) as f:
        lines = f.readlines()

gt = {}
for line in lines:
    vid_id, frame_id, track_id, x, y, w, h, cls = [int(x) for x in line.split(',')]
    if vid_id == 3:
        break

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
gt_json = json.dumps(gt, indent=4)
with open("./gt.json", "w") as outfile:
    outfile.write(gt_json)