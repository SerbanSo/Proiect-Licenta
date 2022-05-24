import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

from constants import GCP_KEY, HAR_MODEL, MHI_OUT_PATH, MHI_THRESHOLD
from google.cloud import automl_v1beta1

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_KEY


def compute_mhi(frames, mhi_duration):
    h, w = frames[0].shape[:2]
    prev_frame = frames[0].copy()
    motion_history = np.zeros((h, w), np.float32)

    for frame in frames[1:]:
        frame_diff = cv.absdiff(frame, prev_frame)
        gray_diff = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
        ret, motion_mask = \
            cv.threshold(gray_diff, MHI_THRESHOLD, 1, cv.THRESH_BINARY)
        timestamp = cv.getTickCount() / cv.getTickFrequency() * 1000
        cv.motempl.updateMotionHistory(
            motion_mask, motion_history, timestamp, mhi_duration)
        vis = np.uint8(np.clip((motion_history - (timestamp - \
            mhi_duration)) / mhi_duration, 0, 1) * 255)
        prev_frame = frame.copy()
    
    return vis


def generate_mhi_picture(frames, mhi_duration, verbose):
    mhi = compute_mhi(frames, mhi_duration)

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    ax.matshow(mhi, cmap=plt.cm.gray)
    plt.savefig(MHI_OUT_PATH)
    
    if verbose >= 2:
        plt.show()
        plt.pause(1)


def get_prediction(content):
    prediction_client = automl_v1beta1.PredictionServiceClient()
    request = prediction_client.predict(
        name=HAR_MODEL, payload={'image': {'image_bytes': content}}, params={})
    
    return request  # Waits until request is returned.


def detect_activity(frames, mhi_duration, objects, verbose):
    # Do not try to detect the activity if the object detection mechanism did
    # not detect any person.
    obj_names = [obj.class_name for obj in objects]
    if "person" not in obj_names:
        return "no person detected", 1

    generate_mhi_picture(frames, mhi_duration, verbose)

    with open(MHI_OUT_PATH, 'rb') as f:
        content = f.read()

    predictions = {}
    try:
        res = get_prediction(content).payload
        for pred in res:
            predictions[pred.display_name] = pred.classification.score
    except Exception as e:
        print(e)
        predictions["error when calling the activity recognition model"] = 1
    finally:
        os.remove(MHI_OUT_PATH)
    
    # Context fusion mechanism: adjust the confidence by adding common sense
    # knowledge and information from the object detection mechanism. An object
    # has more influence if it was detected in more frames. The only objects
    # that can influence the current activities are "cup" and "bottle".
    if objects:
        nr_occurrences = [obj.nr_occurrences for obj in objects]
        max_occ = max(nr_occurrences)
        min_occ = min(nr_occurrences)
    
        for obj, act in \
            [("cup", "drink from a mug"), ("bottle", "drink from a bottle")]:
            if obj in obj_names:
                crt_occ = [obj.nr_occurrences for obj in objects \
                    if obj.class_name == obj]
                crt_occ = crt_occ[0] if crt_occ else 0
                predictions[act] = predictions.setdefault(act, 0) + 0.3 * \
                    (max_occ - crt_occ) / (max_occ - min_occ)

    activity = max(predictions, key=predictions.get)
    confidence = 1.0 * max(predictions.values()) / sum(predictions.values())

    if verbose >= 1:
        for act, conf in predictions.items():
            print("{}, {}".format(act, str(conf)))
        print("Result: {}, {}".format(activity, str(confidence)))
    
    return activity, confidence

