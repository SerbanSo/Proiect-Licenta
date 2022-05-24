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

h, w = 192, 256

floor = np.loadtxt('test2.csv', delimiter=',').astype(np.int64)

print(np.bincount(floor.astype(np.int64).flatten()))
print(np.argmax(np.bincount(floor.astype(np.int64).flatten())[1:]) + 1)

floor_value = np.argmax(np.bincount(floor.astype(np.int64).flatten())[1:]) + 1
floor[floor != floor_value] = 20

print(floor)