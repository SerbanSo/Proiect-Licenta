import cv2 as cv
import sys
import time

from constants import *
from copy import deepcopy
from math import sqrt

sys.path.insert(0, OBJ_DETECTOR_PATH)

from detector import Detector
from opts import opts


def similar_boxes(bbox1, bbox2):
    return sqrt(pow(bbox1[0] - bbox2[0], 2) + (pow(bbox1[1] - bbox2[1], 2))) \
        < SIMILARITY_ERROR or sqrt(pow(bbox1[2] - bbox2[2], 2) + \
        (pow(bbox1[3] -  bbox2[3], 2))) < SIMILARITY_ERROR


class Obj:    
    def __init__(self, crt_name, crt_id, conf, bbox):
        self.class_name = crt_name
        self.tracking_ids = [crt_id]
        self.confidence = conf
        self.bounding_box = bbox    # of type [x1, y1, x2, y2]
        self.nr_occurrences = 1

    def similar_to(self, other_obj):
        return self.class_name == other_obj.class_name and \
            (other_obj.tracking_ids[0] in self.tracking_ids or
            similar_boxes(self.bounding_box, other_obj.bounding_box))
    
    def merge(self, other_obj):
        self.nr_occurrences += 1
        self.bounding_box = other_obj.bounding_box

        if other_obj.confidence > self.confidence:
            self.confidence = other_obj.confidence
        if other_obj.tracking_ids[0] not in self.tracking_ids:
            self.tracking_ids.append(other_obj.tracking_ids[0])
    
    def display(self):
        print("{}, {}, {}, {}, {}".format(self.class_name, self.confidence,
            str(self.bounding_box), str(self.tracking_ids),
            str(self.nr_occurrences)))


def detect_objects(frames, verbose=1):
    # frames = deepcopy(frames)
    opt = opts().init()
    detector = Detector(opt)

    objects = []
    i = 0
    last_frame = None

    count = 0

    for frame in frames:
        i += 1
        if i % FRAME_STEP != 0:
            continue
        if verbose >= 2:
            last_frame = frame

        ret = detector.run(frame)['results']
        
        for obj_raw in ret:
            confidence = obj_raw['score']        
            if confidence > CONF_THRESHOLD:
                obj = Obj(CLASS_NAME[int(obj_raw['class']) - 1],
                    obj_raw['tracking_id'], obj_raw['score'], obj_raw['bbox'])
                
                if verbose >= 2:
                    cv.rectangle(last_frame, (int(obj.bounding_box[0]),
                        int(obj.bounding_box[1])), (int(obj.bounding_box[2]),
                        int(obj.bounding_box[3])), BLUE, 2)
                    cv.putText(last_frame, obj.class_name,
                        (int(obj.bounding_box[0]), int(obj.bounding_box[1]) - \
                         7), cv.FONT_HERSHEY_SIMPLEX, 0.4, BLUE, 1)
                
                existing = False
                for old_obj in objects:
                    if obj.tracking_ids[0] in old_obj.tracking_ids:
                        old_obj.merge(obj)
                        existing = True
                        break

                if not existing:
                    for old_obj in objects:
                        if old_obj.similar_to(obj):
                            old_obj.merge(obj)
                            existing = True
                            break
                
                if not existing:
                    objects.append(obj)

        if verbose >= 2:
            # cv.imwrite("../../3D-BoundingBox/3D-BoundingBox/eval/image_2/%#05d.jpg" % (count+1), frame)
            cv.imshow("Detected objects", last_frame)
            # cv.waitKey(0)
	    # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

            # cv.destroyAllWindows()
            # pass
	
        count += 1

    if verbose >= 1:
        for obj in objects:
            obj.display()

    return objects
