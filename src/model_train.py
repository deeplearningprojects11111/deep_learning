import numpy as np
import os  # Fixed 'impordt OS' to 'import os'
import matplotlib.pyplot as plt  # Fixed 'pylab' to 'pyplot'

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from keras import models  # Fixed 'from keras.model import models' to 'from keras import models'
from glob import glob

import archspec  # Assuming you actually need this
from get_data import get_data, read_params  # Fixed 'get_dta' to 'get_data'


from tensorflow.keras.applications import VGG16
import tensorflow as tf

def train_model(config_file):
    config = get_data(config_file)
    train = config['model']['trainable']
    if train == True:
        img_size = config['model']['img_size']
        train_set = config['model']['train_path']
        test_set = config['model']['test_path']
        num_cls = config['load_data']['num_classes']
        rescale = config['img_augment']['rescale']
        shear_range = config['img_augment']['shear_range']
        zoom_range = config['img_augment']['zoom_range']
        horizontal_flip = config['img_augment']['horizontal_flip']
        vertifal_flip = config['img_augment']['vertifal_flip']
        class_mode = config['img_augment']['class_mode']
        batch = config['img_augment']['batch_size']
        loss = config['model']['loss']
        optimizer = config['model']['optimizer']
        metrics = config['model']['metrics']
        epochs = config['model']['epochs']
        model_path = config['model']['sav_dir']
