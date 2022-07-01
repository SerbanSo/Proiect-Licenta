# Delete the contents of this file after test code is done
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
from utils.metric import eval_plane_and_pixel_recall_normal

h, w = 192, 256

floor = np.loadtxt('test2.csv', delimiter=',').astype(np.int64)

print(np.bincount(floor.astype(np.int64).flatten()))
print(np.argmax(np.bincount(floor.astype(np.int64).flatten())[1:]) + 1)

floor_value = np.argmax(np.bincount(floor.astype(np.int64).flatten())[1:]) + 1
floor[floor != floor_value] = 20

print(floor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with torch.no_grad():
	image = cv2.imread('todo')
	image = cv2.resize(image, (w, h))
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(image)
	image = transforms(image)
	image = image.to(device).unsqueeze(0)
	
	logit, embedding, _, _, param = network(image)

	prob = torch.sigmoid(logit[0])

	# infer per pixel depth using per pixel plane parameter, currently Q_loss need a dummy gt_depth as input
	_, _, per_pixel_depth = Q_loss(param, k_inv_dot_xy1, torch.ones_like(logit))

	# fast mean shift
	segmentation, sampled_segmentation, sample_param = bin_mean_shift.test_forward(
	prob, embedding[0], param, mask_threshold=0.1)


