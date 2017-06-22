from __future__ import print_function

import os
import math
import json
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime

def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def rank(array):
    return len(array.shape)

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)

# images 데이터 형식: NCHW
# image 를 context 생성용 주변부와
# context 를 이용해서 생성해야 하는 recon 영역으로 나눠서 리턴함.
def context_encoder_loader(images, mask):
    # NCHW => CHW
    return context_encoder_fill_unmasked_area(images, mask)


# shape should be NCHW
# H, W 는 짝수여야 함.
def context_encoder_image_mask(shape):
    print('context_encoder_image_mask: ', shape)
    hiding_size = 64
    overlap_size = 7
    #hiding_size = 6
    #overlap_size = 1
    mask_center = np.pad(
            np.ones([hiding_size - 2*overlap_size,   # 1: (64-7)x(64-7)
                     hiding_size - 2*overlap_size]),
            [[overlap_size, overlap_size],            # 0: 7두께로둘러쌈.
             [overlap_size, overlap_size]],
            'constant', constant_values=[[0,0],[0,0]])
    mask_overlap = 1 - mask_center

           

    # 이미지의 크기와 동일하게 mask 크기를 맞춤(0으로 padding)
    pad = (np.array(shape[2:])- np.array(mask_center.shape)) / 2
    pad = pad.astype(int)
    mask_center   = np.pad(mask_center,
                           [[pad[0],pad[0]],[pad[1],pad[1]]],
                           'constant', constant_values=((0,0),(0,0)))
    mask_overlap  = np.pad(mask_overlap,
                           [[pad[0],pad[0]],[pad[1],pad[1]]],
                           'constant', constant_values=[[0,0],[0,0]])

    # 3채널로 만들기
    mask_center   = np.reshape(mask_center,   [1, shape[2], shape[3]])
    mask_overlap  = np.reshape(mask_overlap,  [1, shape[2], shape[3]])
    mask_center   = np.concatenate([mask_center]*3,  0)
    mask_overlap  = np.concatenate([mask_overlap]*3, 0)

    mask_outside = np.ones(shape)
    mask_outside = mask_outside - mask_center - mask_overlap

    return mask_outside, mask_center, mask_overlap

# image 의 mask 외부 영역을 특별한 값으로 채움
# 값의 의미는 모름.
def context_encoder_fill_unmasked_area(image, mask_outside):
    v = np.zeros(mask_outside.shape)
    v[:,0,:,:] = 2*117. / 255. - 1
    v[:,1,:,:] = 2*104. / 255. - 1
    v[:,2,:,:] = 2*123. / 255. - 1

    print('image: ', image.shape)
    print('mask_outside: ', mask_outside.shape)
    return (image * mask_outside) + (v * (1 - mask_outside))
#
#
#def int_shape(tensor):
#    shape = tensor.get_shape().as_list()
#    return [num if num is not None else -1 for num in shape]
#
#def get_conv_shape(tensor, data_format):
#    shape = int_shape(tensor)
#    # always return [N, H, W, C]
#    if data_format == 'NCHW':
#        return [shape[0], shape[2], shape[3], shape[1]]
#    elif data_format == 'NHWC':
#        return shape
#
