# General
This is the code we used to participate [AI CITY CHALLANGE 2023](https://www.aicitychallenge.org/).
We used a YOLOv8 for detector and strongSORT for tracker. 
# Requirement 
- For visualizing-web: ```pip install -r visualizing-web/requirements.txt```
- For yolov8, refer this repo: https://github.com/ultralytics/ultralytics
- For strongSORT, refer this repo: https://github.com/kadirnar/strongsort-pip
# Visualizing data
- If you want to visualize video, move videos to 'visualizing-web/static/data/bbox-video'
- Then, run the following command:
```python visualizing-web/app.py```
- With folder frames, you can cut the video into frames and save them by the format: 001, 002, 003
- With folder static/data, you put your videos in.
# Training
- To train yolov8, refer this repo: https://github.com/ultralytics/ultralytics'
- To train feature extractor of strongsort, refer this repo: https://github.com/kadirnar/strongsort-pip
