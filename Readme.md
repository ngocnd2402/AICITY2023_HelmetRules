# General
This is the code we used to participate [AI CITY CHALLANGE 2023](https://www.aicitychallenge.org/).
We used a YOLOv8 for detector and strongsort for tracker. 
# Requirement 
- For visualizing-web: ```pip install -r visualizing-web/requirements.txt```
- For yolov8, refer this repo: https://github.com/ultralytics/ultralytics
- For strongsort, refer this repo: https://github.com/kadirnar/strongsort-pip
# Visualizing data
- If you want to visualize video, move videos to 'visualizing-web/static/data/bbox-video'
- Then, run the following command:
```python visualizing-web/app.py```
- In frames, you can cut the video into frames and save them by the format: 001, 002, 003
- Download data in this link: 
- After that, put in 
# Training
- To train yolov8, refer this repo: https://github.com/ultralytics/ultralytics
- To train feature extractor of strongsort, refer this repo: https://github.com/kadirnar/strongsort-pip
