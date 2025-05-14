#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import cv2 as cv
import math

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calc_bbox(mask, margin=0):
    # [left right)  [top down)
    height, width = mask.shape
    mask_sum0 = np.sum(mask, axis=0) > 0
    mask_sum1 = np.sum(mask, axis=1) > 0
    left = np.argmax(mask_sum0)
    right = width - np.argmax(mask_sum0[::-1])
    top = np.argmax(mask_sum1)
    down = height - np.argmax(mask_sum1[::-1])

    if margin != 0:
        left = max(0, left - margin)
        right = min(width, right + margin)
        top = max(0, top - margin)
        down = min(height, down + margin)
    bbox = np.array([left, top, right, down], dtype=int)
    return bbox

try:
    from nvjpeg import NvJpeg
    nj = NvJpeg()
except:
    pass 

def encode_bytes(image, image_encode_method=''):
    if image_encode_method == 'cpu':
        image_byte = cv.imencode('.jpg', image)
    elif image_encode_method == 'gpu':
        image_byte = nj.encode(image, 90)
    else:
        image_byte = image.tobytes()
    return image_byte

def normal_to_color(nor):
    ''' x: right [-1,1] -> 0-255   
    y: up [-1,1] -> 0-255   
    z: towards to camrea [0-1] -> 128-255
    out: [0-1]
    '''
    if isinstance(nor, np.ndarray):
        nor = np.clip(nor, -1, 1)
    elif isinstance(nor, torch.Tensor):
        nor = torch.clip(nor, -1, 1)
    nor = (nor + 1) * 0.5
    return nor

def linear_to_srgb(x):
    return torch.clamp_min(x, 1e-8)**(1/2.2)

    srgb0 = 323 / 25 * x
    srgb1 = (211 * torch.maximum(torch.tensor(1e-8), x)**(5 / 12) - 11) / 200
    return torch.where(x <= 0.0031308, srgb0, srgb1)

def envmap_flip(envmap):
    He, We = envmap.shape[:2] # 16, 32
    envmap[:,0:We//2] = np.flip(envmap[:,0:We//2], axis=1) 
    envmap[:,We//2:] = np.flip(envmap[:,We//2:], axis=1)
    return envmap

def exr_to_envmap(image, envmap_shape=(16, 32)):
    H, W = image.shape[:2]
    He, We = envmap_shape
    envmap = np.zeros((He, We, 3), dtype=np.float32)

    for y in range(He):
        for x in range(We):
            yt, yb = math.floor(H/He*y), math.floor(H/He*(y+1))
            xl, xr = math.floor(W/We*x), math.floor(W/We*(x+1))
            envmap[y,x] = np.mean(image[yt:yb,xl:xr], axis=(0,1))
            
    envmap = envmap_flip(envmap)

    return envmap 