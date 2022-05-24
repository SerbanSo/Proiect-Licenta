import cv2 as cv
import os

from gtts import gTTS

from activity_detector import detect_activity
from constants import AUDIO_FILE_PATH
from floor_detector import detect_floor
from objects_detector import detect_objects


def read_video(video_src):
    frames = []

    cam = cv.VideoCapture(video_src)
    if not cam.isOpened():
        print("Could not open video_src " + str(video_src))

    cam.set(cv.CAP_PROP_POS_AVI_RATIO, 1)
    mhi_duration = cam.get(cv.CAP_PROP_POS_MSEC) * 5

    cam = cv.VideoCapture(video_src)
    if not cam.isOpened():
        print("Could not open video_src " + str(video_src))

    ret, frame = cam.read()
    if ret == False:
        print("Could not read from " + str(video_src))
    frames.append(frame)

    
    frame_no = 0

    while cam.isOpened():
        print("Frames" + str(frame_no))
        frame_no += 1
        ret, frame = cam.read()
        if ret == False:
            break
        frames.append(frame)
    
    return frames, mhi_duration

def read_video_generator(video_src):
    frames = []

    cam = cv.VideoCapture(video_src)
    if not cam.isOpened():
        print("Could not open video_src " + str(video_src))

    cam.set(cv.CAP_PROP_POS_AVI_RATIO, 1)
    mhi_duration = cam.get(cv.CAP_PROP_POS_MSEC) * 5

    cam = cv.VideoCapture(video_src)
    if not cam.isOpened():
        print("Could not open video_src " + str(video_src))

    ret, frame = cam.read()
    if ret == False:
        print("Could not read from " + str(video_src))
    # frames.append(frame)

    
    frame_no = 0

    while cam.isOpened():
        print("Frames" + str(frame_no))
        frame_no += 1
        if ret == False:
            return
        yield frame
        ret, frame = cam.read()


def speak(message):
    myobj = gTTS(text=message, lang='en', slow=False)
    myobj.save(AUDIO_FILE_PATH)
    # os.system("mpg321 " + AUDIO_FILE_PATH)
    os.remove(AUDIO_FILE_PATH)


def main():
    # video_src = "09062021/turtlebot-demo1.avi"
    video_src = "vacuum-robot.mp4"
    verbose = 2
    results_to_speech = False
    
    # frames, mhi_duration = read_video_generator(video_src)
    frames = read_video_generator(video_src)

    print('Here')

    # The following two tasks can be run in parallel.
    # Trigger floor detection mechanism.
    # TODO (once running on CPU is fixed).

    # Trigger object detection mechanism.
    objects = detect_objects(frames, verbose)
    
    # Filter object list according to the detected floor.
    # TODO

    # Trigger activity detection mechanism.
    # activity, conf = detect_activity(frames, mhi_duration, objects, verbose)
    
    if results_to_speech:
        speak("Detected objects: " + " and ".join(
            [obj.class_name for obj in objects]))
        speak("Detected activity: " + activity)
    

if __name__ == '__main__':
    main()
