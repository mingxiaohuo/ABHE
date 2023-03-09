import os
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from models import H_estimator, disjoint_augment_image_pair
from loss_functions import intensity_loss
from utils import load, save, DataLoader
import constant
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
gpus=[0,1]