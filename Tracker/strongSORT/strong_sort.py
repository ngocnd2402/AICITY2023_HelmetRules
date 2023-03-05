import numpy as np
import torch

from strongsort.reid_multibackend import ReIDDetectMultiBackend
from strongsort.sort.detection import Detection
from strongsort.sort.nn_matching import NearestNeighborDistanceMetric
from strongsort.sort.tracker import Tracker

# Id converted
MOTOR = 0
HELMET = 1
NO_HELMET = 2

# Real ID
MOTORBIKE = 1
D_HELMET = 2
D_NO_HELMET = 3
P1_HELMET = 4
P1_NO_HELMET = 5
P2_HELMET = 6
P2_NO_HELMET = 7

NOT_RELAVANT = -2

def find_inter_area(box1, box2):
    x_min_1 = box1[0]
    y_min_1 = box1[1]
    x_max_1 = box1[0] + box1[2]
    y_max_1 = box1[1] + box1[3]

    x_min_2 = box2[0]
    y_min_2 = box2[1]
    x_max_2 = box2[0] + box2[2]
    y_max_2 = box2[1] + box2[3]

    dx = min(x_max_1, x_max_2) - max(x_min_1, x_min_2)
    dy = min(y_max_1, y_max_2) - max(y_min_1, y_min_2)

    if (dx>=0) and (dy>=0):
        return dx*dy
    else: 
        return 0

def get_real_class_id(class_id, pos=None):
    if class_id == MOTOR:
        if pos == 'X':
            return -1 # This is for debug
        else:
            return MOTORBIKE
    elif class_id == HELMET:
        if pos == 'D':
            return D_HELMET
        elif pos == 'P1':
            return P1_HELMET
        elif pos == 'P2':
            return P2_HELMET
        else:
            return -1 # This is for debug
    else:
        if pos == 'D':
            return D_NO_HELMET
        elif pos == 'P1':
            return P1_NO_HELMET
        elif pos == 'P2':
            return P2_NO_HELMET
        else:
            return -1 # This is for debug

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product(p1, p2):
    return p1[0] * p2[0] + p1[1] + p2[1]

def get_argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def get_argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

class StrongSORT(object):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        max_dist=0.2,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
    ):

        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)

        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric("cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, dets, ori_img, frame_idx):
        xyxys = dets[:, :4]
        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]

        classes = clss.numpy()
        xywhs = xyxy2xywh(xyxys.numpy())
        confs = confs.numpy()
        self.height, self.width = ori_img.shape[:2]

        # generate detections
        features = self._get_features(xywhs, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confs)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, clss, confs)

        #  update frames and bboxes
        motor_ids = []
        people_ids = []
        map_pt_id_2_idx = {}

        for i, track in enumerate(self.tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            track.add_info['frames'].append(frame_idx)
            track.add_info['bboxes'].append(bbox)
            if track.class_id == MOTOR:
                motor_ids.append(i)
                track.add_info['people'].append([])
            else:
                people_ids.append(i)

            map_pt_id_2_idx[track.track_id] = i

        # Assign people to motors (update people of motor) and update is_clustered (attr of person)
        # Can use is_clustered for another purpose?
        # Use both area of intersection and motion vector
        # Chua xet truong hpo person chua duoc phan loai
        # Chua loai tuong hop track bi mat
        for person_id in people_ids:
            areas = []
            cos_sims = []

            person_frames = self.tracker.tracks[person_id].add_info['frames']
            person_track_id = self.tracker.tracks[person_id].track_id

            # If person is hard driver, skip
            hard_motor = self.tracker.tracks[person_id].add_info['hard_motor']
            if hard_motor is not None:
                if hard_motor in map_pt_id_2_idx:
                    motor_idx = map_pt_id_2_idx[hard_motor]
                    self.tracker.tracks[motor_idx].add_info['people'][-1].append(hard_motor)
                continue 
            
            for motor_id in motor_ids:
                current_person_bbox = self.tracker.tracks[person_id].to_tlwh()
                current_motor_bbox = self.tracker.tracks[motor_id].to_tlwh()
                areas.append(find_inter_area(current_person_bbox, current_motor_bbox))

            if areas.count(0) == len(areas):
                self.tracker.tracks[person_id].add_info['is_clustered'] = False
                continue
            else:
                self.tracker.tracks[person_id].add_info['is_clustered'] = True

            if len(person_frames) <= 1:
                max_idx = get_argmax(areas)
                best_motor_id = motor_ids[max_idx]
                self.tracker.tracks[best_motor_id].add_info['people'][-1].append(person_track_id)
            else:
                for i, motor_id in enumerate(motor_ids):
                    if areas[i] == 0:
                        cos_sims.append(NOT_RELAVANT)
                    else: # Chua xet truong hop motor chi moi co 1 frame
                        cur_person_bbox = self.tracker.tracks[person_id].add_info['bboxes'][-1]
                        prev_person_bbox = self.tracker.tracks[person_id].add_info['bboxes'][-min(3, len(person_frames))]
                        cur_person_center = np.array((
                            cur_person_bbox[0] + cur_person_bbox[2] / 2,
                            cur_person_bbox[1] + cur_person_bbox[3] / 2
                        ))
                        prev_person_center = np.array((
                            prev_person_bbox[0] + prev_person_bbox[2] / 2,
                            prev_person_bbox[0] + prev_person_bbox[3] / 2
                        ))
                        person_vector = cur_person_center - prev_person_center

                        motor_frames = self.tracker.tracks[motor_id].add_info['frames']
                        cur_motor_bbox = self.tracker.tracks[motor_id].add_info['bboxes'][-1]
                        prev_motor_bbox = self.tracker.tracks[motor_id].add_info['bboxes'][-min(3, len(motor_frames))]
                        cur_motor_center = np.array((
                            cur_motor_bbox[0] + cur_motor_bbox[2] / 2,
                            cur_motor_bbox[1] + cur_motor_bbox[3] / 2
                        ))
                        prev_motor_center = np.array((
                            prev_motor_bbox[0] + prev_motor_bbox[2] / 2,
                            prev_motor_bbox[0] + prev_motor_bbox[3] / 2
                        ))
                        motor_vector = cur_motor_center - prev_motor_center
                        
                        cos = cosine_similarity(motor_vector, person_vector)
                        if cos >= 0.7:
                            cos_sims.append(cos)
                        else:
                            cos_sims.append(NOT_RELAVANT)
                            
                        if self.tracker.tracks[person_id].track_id in [8] and self.tracker.tracks[motor_id].track_id in [6, 7]:
                            print(f'- motor_id: {self.tracker.tracks[motor_id].track_id}, {cosine_similarity(motor_vector, person_vector)}')
                            # print(f'--- vecotr: {person_id}')
                best_motor_id = motor_ids[get_argmax(cos_sims)]
                self.tracker.tracks[best_motor_id].add_info['people'][-1].append(person_track_id) 

        # for track in self.tracker.tracks:
        #     if track.class_id == MOTOR:
        #         print(track.add_info['people'])
        # # Write to file
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            # person not clusted will be processed seperately
            # person clustered will be processed with motor
            # if track.class_id in (HELMET, NO_HELMET):
            #     print(track.class_id)
            #     for i, info in track.add_info.items():
            #         print('-- ', i, ': ', info)

            if track.class_id in (HELMET, NO_HELMET) and not track.add_info['is_clustered']:
                track.add_info['pos'] = 'D'
            elif track.class_id == MOTOR:
                people = track.add_info['people'][-1]
                n_people = len(people)
                if n_people == 0:
                    continue
                elif len(track.add_info['frames']) < 3 or n_people == 1:
                    for i, person_track_id in enumerate(people):
                        track_idx = map_pt_id_2_idx[person_track_id]
                        self.tracker.tracks[track_idx].add_info['pos'] = 'D'
                        # Need to check
                        self.tracker.tracks[track_idx].add_info['hard_motor'] = track.track_id
                elif n_people == 2:
                    # Get info of 2 person
                    id1 = map_pt_id_2_idx[people[0]]
                    bbox1 = self.tracker.tracks[id1].to_tlbr()
                    id2 = map_pt_id_2_idx[people[1]]
                    bbox2 = self.tracker.tracks[id1].to_tlbr()

                    # Find vector person1 -> person2
                    center1 = np.array((bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2))
                    center2 = np.array((bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2))
                    v12 = center2 - center1
                    
                    # Find motion vector of motorbike
                    cur_motor_bbox = track.add_info['bboxes'][-1]
                    prev_motor_bbox = track.add_info['bboxes'][-3]

                    cur_center = np.array((
                        cur_motor_bbox[0] + cur_motor_bbox[2] / 2, 
                        cur_motor_bbox[1] + cur_motor_bbox[3] / 2
                    ))
                    prev_center = np.array((
                        prev_motor_bbox[0] + prev_motor_bbox[2] / 2, 
                        prev_motor_bbox[1] + prev_motor_bbox[3] / 2
                    ))
                    v = cur_center - prev_center
                    
                    # Find dot product
                    if v.dot(v12) >= 0:
                        self.tracker.tracks[id1].add_info['pos'] = 'P1'
                        self.tracker.tracks[id2].add_info['pos'] = 'D'
                        self.tracker.tracks[id2].add_info['hard_motor'] = track.track_id
                    else:
                        self.tracker.tracks[id1].add_info['pos'] = 'D'
                        self.tracker.tracks[id2].add_info['pos'] = 'P1'
                        self.tracker.tracks[id1].add_info['hard_motor'] = track.track_id

                elif n_people == 3:
                    # Get info of 3 person
                    id0 = map_pt_id_2_idx[people[0]]
                    bbox0 = self.tracker.tracks[id0].to_tlbr()
                    id1 = map_pt_id_2_idx[people[1]]
                    bbox1 = self.tracker.tracks[id1].to_tlbr()
                    id2 = map_pt_id_2_idx[people[2]]
                    bbox2 = self.tracker.tracks[id2].to_tlbr()
                    pos = [id0, id1, id2]
                    # Find vector person1 -> person2, 1 -> 3, 2 -> 3
                    center0 = np.array((bbox0[0] + bbox0[2] / 2, bbox0[1] + bbox0[3] / 2))
                    center1 = np.array((bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2))
                    center2 = np.array((bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2))
                    v01 = center1 - center0
                    v02 = center2 - center0
                    v12 = center2 - center1

                    # Find motion vector of motorbike
                    # Find motion vector of motorbike
                    cur_motor_bbox = track.add_info['bboxes'][-1]
                    prev_motor_bbox = track.add_info['bboxes'][-3]

                    cur_center = np.array((
                        cur_motor_bbox[0] + cur_motor_bbox[2] / 2, 
                        cur_motor_bbox[1] + cur_motor_bbox[3] / 2
                    ))
                    prev_center = np.array((
                        prev_motor_bbox[0] + prev_motor_bbox[2] / 2, 
                        prev_motor_bbox[1] + prev_motor_bbox[3] / 2
                    ))
                    v = cur_center - prev_center

                    # print('last:', last_center, 'prev: ', prev_center)
                    # print('v: ', v)
                    # Find dot product. If dot product with v01 >= 0, person1 will be in front
                    # of the person 0. Therefore, count[1] += 1 
                    count = np.array([0] * 3)
                    
                    if v.dot(v01) >= 0:
                        count[1] += 1
                    else:
                        count[0] += 1
                    
                    if v.dot(v02) >= 0:
                        count[2] += 1
                    else: 
                        count[0] += 1
                    
                    if v.dot(v12) >= 0:
                        count[2] += 1
                    else: 
                        count[1] += 1
                    
                    argmax = np.argmax(count)
                    argmin = np.argmin(count)
                    
                    if argmax == 0 and argmin == 1:
                        argme = 2
                        self.tracker.tracks[id0].add_info['hard_motor'] = track.track_id
                    elif argmax == 1 and argmin == 0:
                        argme = 2
                        self.tracker.tracks[id1].add_info['hard_motor'] = track.track_id
                    elif argmax == 0 and argmin == 2:
                        argme = 1
                        self.tracker.tracks[id0].add_info['hard_motor'] = track.track_id
                    elif argmax == 2 and argmin == 0:
                        argme = 1
                        self.tracker.tracks[id2].add_info['hard_motor'] = track.track_id
                    elif argmax == 1 and argmin == 2:
                        argme = 0
                        self.tracker.tracks[id1].add_info['hard_motor'] = track.track_id
                    else:
                        argme = 0
                        self.tracker.tracks[id2].add_info['hard_motor'] = track.track_id
                    
                    # Update pos__on_motor
                    self.tracker.tracks[pos[argmax]].add_info['pos'] = 'D'
                    self.tracker.tracks[pos[argme]].add_info['pos'] = 'P1'
                    self.tracker.tracks[pos[argmin]].add_info['pos'] = 'P2'
                else: 
                    print("Processing frame %05d" % frame_idx)
                    n = len(motor_self.tracker[track.track_id]['people'][-1])
                    print(f'More than 3 ({n}) people assigned to 1 motor: ', 
                    motor_self.tracker[track.track_id]['people'][-1])

                    for person_id in motor_self.tracker[track.track_id]['people'][-1]:
                        pos = people[person_id]['pos_in_self.tracker']
                        self.tracker.tracks[pos].add_info['pos'] = 'X'

        results = []
        for track in self.tracker.tracks:     
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, 
                round(bbox[0], 2), round(bbox[1], 2), round(bbox[2], 2), round(bbox[3], 2), 
                track.conf, get_real_class_id(track.class_id, track.add_info['pos'])])
        
        return results
    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features
