from argparse import Namespace
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

def init_detector():
    print("====================================================================================================================================================================================")
    # print(opt)
    opt = Namespace(K=100, add_05=False, amodel_offset_weight=1, arch='dla_34', aug_rot=0, backbone='dla34', batch_size=32, chunk_sizes=[32], custom_dataset_ann_path='', custom_dataset_img_path='', data_dir='object_detector/lib/../../data', dataset='coco', dataset_version='', debug=0, debug_dir='object_detector/lib/../../exp/tracking/default/debug', debugger_theme='white', demo='', dense_reg=1, dep_weight=1, depth_scale=1, dim_weight=1, dla_node='dcn', down_ratio=4, efficient_level=0, eval_val=False, exp_dir='object_detector/lib/../../exp/tracking', exp_id='default', fix_res=True, fix_short=-1, flip=0.5, flip_test=False, fp_disturb=0, gpus=[0], gpus_str='0', head_conv={'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256]}, head_kernel=3, heads={'hm': 80, 'reg': 2, 'wh': 2, 'tracking': 2}, hm_disturb=0, hm_hp_weight=1, hm_weight=1, hp_weight=1, hungarian=False, ignore_loaded_cats=[], input_h=512, input_res=512, input_w=512, keep_res=False, kitti_split='3dop', load_model='object_detector/coco_tracking.pth', load_results='', lost_disturb=0, lr=0.000125, lr_step=[60], ltrb=False, ltrb_amodal=False, ltrb_amodal_weight=0.1, ltrb_weight=0.1, map_argoverse_id=False, master_batch_size=32, max_age=-1, max_frame_dist=3, model_output_list=False, msra_outchannel=256, neck='dlaup', new_thresh=0.3, nms=False, no_color_aug=False, no_pause=False, no_pre_img=False, non_block_test=False, not_cuda_benchmark=False, not_idaup=False, not_max_crop=False, not_prefetch_test=False, not_rand_crop=False, not_set_cuda_env=False, not_show_bbox=False, not_show_number=False, not_show_txt=False, num_classes=80, num_epochs=70, num_head_conv=1, num_iters=-1, num_layers=101, num_stacks=1, num_workers=4, nuscenes_att=False, nuscenes_att_weight=1, off_weight=1, only_show_dots=False, optim='adam', out_thresh=0.3, output_h=128, output_res=128, output_w=128, pad=31, pre_hm=False, pre_img=True, pre_thresh=0.3, print_iter=0, prior_bias=-4.6, public_det=False, qualitative=False, reg_loss='l1', reset_hm=False, resize_video=False, resume=False, reuse_hm=False, root_dir='object_detector/lib/../..', rot_weight=1, rotate=0, same_aug_pre=False, save_all=False, save_dir='object_detector/lib/../../exp/tracking/default', save_framerate=30, save_img_suffix='', save_imgs=[], save_point=[90], save_results=False, save_video=False, scale=0, seed=317, shift=0, show_trace=False, show_track_color=False, skip_first=-1, tango_color=False, task='tracking', test=False, test_dataset='coco', test_focal_length=-1, test_scales=[1.0], track_thresh=0.3, tracking=True, tracking_weight=1, trainval=False, transpose_video=False, use_kpt_center=False, use_loaded_results=False, val_intervals=10000, velocity=False, velocity_weight=1, video_h=512, video_w=512, vis_gt_bev='', vis_thresh=0.3, weights={'hm': 1, 'reg': 1, 'wh': 0.1, 'tracking': 1}, wh_weight=0.1, zero_pre_hm=False, zero_tracking=False)
    print("====================================================================================================================================================================================")
    detector = Detector(opt)

    return detector

def detect_objects(frames, detector, verbose=2):
    # frames = deepcopy(frames)
    # opt = opts().init()

    objects = []
    i = 0
    last_frame = None

    count = 0

    time_mean = 0.0

    for frame in frames:
        # print('Here')
        i += 1
        # if i % FRAME_STEP != 0:
            # continue
        if verbose >= 2:
            last_frame = frame
	    
        if i > 500:
            print('Mean time:' + str(time_mean / 500.0))
            break

        start = time.time() # time record
        ret = detector.run(frame)['results']
        
        print(ret)

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

        # print('Here')
        if verbose >= 2:
            # cv.imwrite("../../3D-BoundingBox/3D-BoundingBox/eval/image_2/%#05d.jpg" % (count+1), frame)
            # print('Here')
            # cv.imshow("AAA Detected objects", last_frame)
            return last_frame
            # cv.waitKey(0)
	    # Press Q on keyboard to  exit
        #     if cv.waitKey(25) & 0xFF == ord('q'):
        #         break

            # cv.destroyAllWindows()
            # pass
	
        count += 1

    if verbose >= 1:
        for obj in objects:
            obj.display()

    return objects
