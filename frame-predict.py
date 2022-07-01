import os
import cv2
import random
import numpy as np
from PIL import Image
from distutils.version import LooseVersion

from sacred import Experiment
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
import torchvision.transforms as tf

from models.baseline_same import Baseline as UNet
from utils.disp import tensor_to_image
from utils.disp import colors_256 as colors
from bin_mean_shift import Bin_Mean_Shift
from modules import get_coordinate_map
from utils.loss import Q_loss
from instance_parameter_loss import InstanceParameterLoss
from utils.metric import black_and_white_img, color_to_white_img, eval_plane_and_pixel_recall_normal, pixel_accuracy, zoom_to_plane
import time

import mediapipe as mp

from objects_detector import detect_objects, init_detector


DEMO_IMG = 2

ex = Experiment()


@ex.main
def predict(_run, _log):
    cfg = edict(_run.config)


    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = init_detector()

    # build network
    network = UNet(cfg.model)

    if not (cfg.resume_dir == 'None'):
        model_dict = torch.load(cfg.resume_dir, map_location=lambda storage, loc: storage)
        network.load_state_dict(model_dict)

    # load nets into gpu
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        network = torch.nn.DataParallel(network)
    network.to(device)
    network.eval()

    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    bin_mean_shift = Bin_Mean_Shift(device=device)
    k_inv_dot_xy1 = get_coordinate_map(device)
    instance_parameter_loss = InstanceParameterLoss(k_inv_dot_xy1)

    h, w = 192, 256

    # First filter
    floor_mask_1 = np.zeros((h, w))

    for i in range(int(h/2), h):
        floor_mask_1[i][:] = 1

    floor_mask_1 = floor_mask_1.astype(np.int64)

    # Second filter
    floor_mask_2 = np.zeros((h, w))

    for i in range(int(h/2), h):
        # floor_mask_2[i][0:int(w/4)] = 1
        # floor_mask_2[i][int(3*w/4):] = 1
        floor_mask_2[i][int(w/4):int(3*w/4)] = 2

    floor_mask_2_print = np.zeros((h, w))
    for i in range(int(h/2), h):
        floor_mask_2_print[i][0:int(w/4)] = 1
        floor_mask_2_print[i][int(3*w/4):] = 1
        floor_mask_2_print[i][int(w/4):int(3*w/4)] = 2

    floor_mask_2 = floor_mask_2.astype(np.int64)
    floor_mask_2_print = floor_mask_2_print.astype(np.int64)
    
    # Selected the used filter
    floor_mask = floor_mask_1

    filter = 'filter_1'

    # objectron
    mp_objectron = mp.solutions.objectron
    mp_drawing = mp.solutions.drawing_utils

    onlyframes = [f for f in os.listdir(cfg.images_path) if os.path.isfile(os.path.join(cfg.images_path, f))]
    noFrames = len(onlyframes)
    count = 0

    frame_array = []
    size = (0, 0)

    frame_threshold = 10000
    noFrames = frame_threshold if frame_threshold < noFrames else noFrames

    time_mean = 0.0

    framePath = os.path.join(cfg.images_path, onlyframes[0])  
    tmp_image = cv2.imread(framePath)
    tmp_image = cv2.resize(tmp_image, (w, h))
    tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
    tmp_image = Image.fromarray(tmp_image)
    tmp_image = transforms(tmp_image)
    tmp_image = tmp_image.to(device).unsqueeze(0) 
    cv2.imshow("AAADetected objects", tensor_to_image(tmp_image.cpu()[0]))
    cv2.waitKey(0)

    # print(onlyframes)
    for frame in onlyframes:
        print('Progress' + str(count / noFrames * 100))

        if count > noFrames - 1:
                break

        if count > 500:
                break
                
        framePath = os.path.join(cfg.images_path, frame)

        # start = time.time() # time record
        # _ = detect_objects(frame, verbose=1)

        # end = time.time()
        # print('Time elapsed: ', end - start)
        # time_mean += end - start



        with torch.no_grad():
                # start = time.time() # time record
                image = cv2.imread(framePath)
                cp_img = image.copy()
                # the network is trained with 192*256 and the intrinsic parameter is set as ScanNet
                image = cv2.resize(image, (w, h))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = transforms(image)
                image = image.to(device).unsqueeze(0)
                # forward pass
                logit, embedding, _, _, param = network(image)

                prob = torch.sigmoid(logit[0])

                # infer per pixel depth using per pixel plane parameter, currently Q_loss need a dummy gt_depth as input
                _, _, per_pixel_depth = Q_loss(param, k_inv_dot_xy1, torch.ones_like(logit))

                # fast mean shift
                segmentation, sampled_segmentation, sample_param = bin_mean_shift.test_forward(
                prob, embedding[0], param, mask_threshold=0.1)

                # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned, 
                # we thus use avg_pool_2d to smooth the segmentation results
                b = segmentation.t().view(1, -1, h, w)

                predict_segmentation_not_smooth = segmentation.view(-1, h*w).t().cpu().numpy().argmax(axis=1)

                pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
                b = pooling_b.view(-1, h*w).t()
                segmentation = b

                # infer instance depth
                instance_loss, instance_depth, instance_abs_disntace, instance_parameter = instance_parameter_loss(
                segmentation, sampled_segmentation, sample_param, torch.ones_like(logit), torch.ones_like(logit), False)

                # return cluster results
                predict_segmentation = segmentation.cpu().numpy().argmax(axis=1)

                # mask out non planar region
                predict_segmentation[prob.cpu().numpy().reshape(-1) <= 0.1] = 20
                predict_segmentation = predict_segmentation.reshape(h, w)
    
                predict_floor = np.multiply(floor_mask_1, predict_segmentation + 1)
                if filter == 'filter_2':
                    filtered_floor = np.append(predict_floor.flatten(), np.multiply(floor_mask_2, predict_segmentation + 1).flatten())
                else:
                    filtered_floor = predict_floor
                floor_value = np.argmax(np.bincount(filtered_floor.flatten())[1:]) + 1
                predict_floor[(predict_segmentation+1) != floor_value] = 20
                predict_floor[(predict_segmentation+1) == floor_value] = 0
                # predict_segmentation = predict_floor

                # visualization and evaluation
                image = tensor_to_image(image.cpu()[0])
                mask = (prob > 0.1).float().cpu().numpy().reshape(h, w)
                depth = instance_depth.cpu().numpy()[0, 0].reshape(h, w)
                per_pixel_depth = per_pixel_depth.cpu().numpy()[0, 0].reshape(h, w)

                # use per pixel depth for non planar region
                depth = depth * (predict_segmentation != 20) + per_pixel_depth * (predict_segmentation == 20)

                # change non planar to zero, so non planar region use the black color
                predict_segmentation += 1
                predict_segmentation[predict_segmentation == 21] = 0
                predict_floor += 1
                predict_floor[predict_floor == 21] = 0
                predict_floor[predict_floor != 0] = 2

                pred_seg = cv2.resize(np.stack([colors[predict_segmentation, 0],
                                                colors[predict_segmentation, 1],
                                                colors[predict_segmentation, 2]], axis=2), (w, h))

                pred_flo = cv2.resize(np.stack([colors[predict_floor, 0],
                                                colors[predict_floor, 1],
                                                colors[predict_floor, 2]], axis=2), (w, h))

                floor_mask_1_img = cv2.resize(np.stack([colors[floor_mask_1, 0],
                                                	colors[floor_mask_1, 1],
                                                	colors[floor_mask_1, 2]], axis=2), (w, h))

                floor_mask_2_img = cv2.resize(np.stack([colors[floor_mask_2_print, 0],
                                                	colors[floor_mask_2_print, 1],
                                                	colors[floor_mask_2_print, 2]], axis=2), (w, h))

                # blend image
                blend_pred = (pred_seg * 0.7 + image * 0.3).astype(np.uint8)

                # visualize depth map as PlaneNet
                depth = 255 - np.clip(depth / 5 * 255, 0, 255).astype(np.uint8)
                depth = cv2.cvtColor(cv2.resize(depth, (w, h)), cv2.COLOR_GRAY2BGR)

                # end = time.time()
                # print('Time elapsed: ', end - start)
                # time_mean += end - start

                mask = cv2.resize((mask * 255).astype(np.uint8), (w, h))
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                orig_image = image
                # filter_1_img = np.concatenate((image, floor_mask_1_img, floor_mask_1_img * 0.7 + image * 0.3, pred_flo), axis=1)
                filter_1_img = np.concatenate((image, floor_mask_1_img, (floor_mask_1_img * 0.7 + image * 0.3).astype(np.uint8), pred_flo, pred_seg), axis=1)
                filter_2_img = np.concatenate((image, floor_mask_2_img, (floor_mask_2_img * 0.7 + image * 0.3).astype(np.uint8), pred_flo, pred_seg), axis=1)
                floor_exemple = np.concatenate((image, (pred_flo * 0.7 + image * 0.3).astype(np.uint8)), axis=1)
                image = np.concatenate((image, pred_flo, blend_pred, mask, depth), axis=1)

                # blend image
                # mask out non planar region
                # change non planar to zero, so non planar region use the black color
                predict_segmentation_not_smooth[prob.cpu().numpy().reshape(-1) <= 0.1] = 20
                predict_segmentation_not_smooth = predict_segmentation_not_smooth.reshape(h, w)
                predict_segmentation_not_smooth += 1
                predict_segmentation_not_smooth[predict_segmentation_not_smooth == 21] = 0
                predict_segmentation_not_smooth[predict_segmentation_not_smooth != 0] = 21
                pred_seg_not_smooth = cv2.resize(np.stack([colors[predict_segmentation_not_smooth, 0],
                                                colors[predict_segmentation_not_smooth, 1],
                                                colors[predict_segmentation_not_smooth, 2]], axis=2), (w, h))
                # blend_pred_not_smooth = (pred_seg_not_smooth * 0.7 + image * 0.3).astype(np.uint8)
                # blend_pred_not_smooth = (pred_seg_not_smooth * 0.7 + orig_image * 0.3).astype(np.uint8)
                # img_smooth = np.concatenate((pred_seg_not_smooth, pred_seg), axis=1)
                # img_not_smooth = blend_pred_not_smooth

                line, column = zoom_to_plane(pred_flo)
                print(str(line) + ' ' + str(column))
                zoomed_image = orig_image.copy()
                zoomed_image = zoomed_image[line:, column:, :]
                zoomed_image_diff = orig_image.copy()
                zoomed_image_diff[:line, :, :] = 0
                zoomed_image_diff[:, :column, :] = 0

                # cv2.imwrite('tmp.png', image)
                # cv2.imwrite('./demo/results/floor_mask_1.png', floor_mask_1_img)
                # cv2.imwrite('./demo/results/floor_mask_2.png', floor_mask_2_img)
                # cv2.imwrite('./demo/results/img_smooth.png', img_not_smooth)
                # cv2.imwrite('./demo/results/full.png', image)
                # frame_array.append(image)
                # cv2.imshow("Detected objects", pred_flo)
                # cv2.imshow("Detected objects", orig_image)
                # cv2.waitKey(0)
                # cv2.imshow("Detected objects", zoomed_image)
                # cv2.imshow("Detected objects", filter_2_img)
                # cv2.imshow("Detected objects", floor_exemple)
	            # Press Q on keyboard to  exit
                # cv2.waitKey(0)
                
                # resize_zoomed_img = cv2.resize(zoomed_image, (640, 360), interpolation = cv2.INTER_AREA)
                # print(resize_zoomed_img.shape)
                # cv2.imshow("Detected objects", resize_zoomed_img)
                # cv2.imshow("Detected objects", zoomed_image)
                # cv2.waitKey(0)
                # detected_obj_img = detect_objects([orig_image.copy()], detector, verbose=2)
                detected_obj_img = detect_objects([zoomed_image], detector, verbose=2)
                # print('Before detect_objects')
                # detect_objects([zoomed_image], verbose=3)
                # print('After detect_objects')
                # cv2.imshow("Detected objects", detected_obj_img)
                # cv2.waitKey(0)

                # if cv2.waitKey(25) & 0xFF == ord('q'):
                    # break

                
                # image = cv2.imread('./demo/own_imgs/img_5_gt.jpg')
                # # the network is trained with 192*256 and the intrinsic parameter is set as ScanNet
                # image = cv2.resize(image, (w, h))
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = Image.fromarray(image)
                # image = transforms(image)
                # image = image.to(device).unsqueeze(0)
                # image = tensor_to_image(image.cpu()[0])

                # image = black_and_white_img(image)
                # pred_flo = color_to_white_img(pred_flo)

                annotated_image = cp_img.copy() 
                annotated_image = objectron_object(mp_objectron, mp_drawing, annotated_image, object='Shoe')
                annotated_image = objectron_object(mp_objectron, mp_drawing, annotated_image, object='Cup')
                annotated_image = cv2.resize(annotated_image, (w, h))
                # annotated_image = objectron_object(mp_objectron, mp_drawing, annotated_image, object='Shoe')

                # demo_img = np.concatenate((zoomed_image, detected_obj_img), axis=1)
                demo_img = np.concatenate((orig_image, zoomed_image_diff), axis=1)
                # demo_img = np.concatenate((orig_image, zoomed_image_diff), axis=1)
                # cv2.imshow("Detected objects", demo_img)
                # cv2.waitKey(0)
                # demo_img = np.concatenate((orig_image, annotated_image), axis=1)

                # cv2.imshow("Detected objects", annotated_image)
                # cv2.imshow("Detected objects", pred_flo)
                # cv2.waitKey(0)
                # cv2.imshow("Detected objects", image)
                # cv2.waitKey(0)
                # print(pixel_accuracy(pred_flo, image))

                if DEMO_IMG == 1:
                        cv2.imshow("Detected objects", filter_1_img)
                        cv2.waitKey(0)
                        cv2.imshow("Detected objects", demo_img)
                        cv2.waitKey(0)
                        cv2.imshow("Detected objects", detected_obj_img)
                        cv2.waitKey(0)
                        cv2.imshow("Detected objects", annotated_image)
                        cv2.waitKey(0)
                else:
                        demo_video = np.concatenate((orig_image, pred_flo, cv2.resize(detected_obj_img, (w, h)), annotated_image), axis=1)
                        cv2.imshow("Detected objects", demo_video)
                        # cv2.waitKey(0)



                height, width, _ = image.shape
                size = (width, height)

        count += 1

    print('Mean time: ' + str(time_mean / 500.0))
    # out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    # for i in range(len(frame_array)):
        # out.write(frame_array[i])
    # out.release()

def objectron_object(mp_objectron, mp_drawing, annotated_image, object='Shoe'):
    with mp_objectron.Objectron(
        static_image_mode=True,
        max_num_objects=5,
        min_detection_confidence=0.1,
        model_name=object) as objectron:
        # Run inference on shoe images.
        # Convert the BGR image to RGB and process it with MediaPipe Objectron.
        results = objectron.process(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

        # Draw box landmarks.
        if not results.detected_objects:
            print(f'No box landmarks detected on Shoe')
        else:
            print(f'Box landmarks of Shoe')
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(
                    annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(annotated_image, detected_object.rotation, detected_object.translation)
                # cv2.imshow('Annotated image', annotated_image)
                # cv2.waitKey(0)
        return annotated_image
    

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    ex.add_config('./configs/frame-predict.yaml')
    ex.run_commandline()
