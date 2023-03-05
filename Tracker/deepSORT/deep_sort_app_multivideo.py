from deep_sort_app import *
import os 

SET_DIR = 'data/test' # 'data/train'
DET_DIR = 'detections/AIC2023'
OUTPUT_DIR = 'result'

video_names = sorted(os.listdir(SET_DIR))
det_file_names = sorted(os.listdir(DET_DIR))

min_confidence = 0.3
nms_max_overlap = 1.0
min_detection_height = 0
max_cosine_distance = 0.2
nn_budget = 100
display = False

if __name__ == '__main__':
    for i in range(len(video_names)):
        print(f'Processing video {video_names[i]}')
        sequence_dir = os.path.join(SET_DIR, video_names[i])
        detection_file = os.path.join(DET_DIR, det_file_names[i])
        output_file = os.path.join(OUTPUT_DIR, f'{video_names[i]}.txt')

        run(
            sequence_dir, detection_file, output_file,
            min_confidence, nms_max_overlap, min_detection_height,
            max_cosine_distance, nn_budget, display)