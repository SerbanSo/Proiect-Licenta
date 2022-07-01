OBJ_DETECTOR_PATH = "object_detector/lib/"

# Maximum distance allowed between two points (bounding box coordinates) such
# that they are considered the same point.
SIMILARITY_ERROR = 10

# Prediction confidence. Every prediction with a confidence below this thres-
# hold is discarded.
# CONF_THRESHOLD = 0.35
CONF_THRESHOLD = 0.25

BLUE = (0, 0, 255)

CLASS_NAME = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# In the object detection mechanism, due to hardware constraints (i.e. need to
# run the application on a machine without GPUs) and controlled environment
# (objects are considered to stay in the same position for long periods, the
# robot does not move or moves slowly), skip every `FRAME_STEP` - 1 frames.
FRAME_STEP = 7     # Considering 15fps.

# Constant that determines the intensity of the motion history images.
MHI_THRESHOLD = 40

MHI_OUT_PATH = "mhi.png"

# Google Cloud Platform configuration constants for calling the human activity
# recognition module.
GCP_KEY = "har-rgb-3cadab83ede7.json"
GCP_PROJECT_ID = "har-rgb"
GCP_MODEL_ID = "ICN8870376055897672796"
HAR_MODEL = 'projects/{}/locations/us-central1/models/{}'.format(
    GCP_PROJECT_ID, GCP_MODEL_ID)
    
AUDIO_FILE_PATH = "out.mp3"

