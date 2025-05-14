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
from torch import nn
import numpy as np
from utils.graphics_utils import getProjectionMatrix_refine, getProjectionMatrix 

class Camera():
    uid = 0
    world_vertex = None
    smpl_R = None
    smpl_T = None
    frame_id = None
    cam_id = None
    mask = None

    Rh = None
    Th = None
    poses = None
    shapes = None
    def __init__(self, image, image_name, T, R, fovx, fovy, K=None):
        '''matrix is in normal form'''
        self.image_name = image_name
        self.original_image = image 
        self.image_width = image.shape[1]
        self.image_height = image.shape[0]

        self.T = T
        self.R = R

        self.FoVx = fovx
        self.FoVy = fovy

        self.zfar = 100.0
        self.znear = 0.01

        _C2W = np.eye(4, dtype=np.float32)
        _C2W[:3,:3] = R
        _C2W[:3,3] = T
        self.world_view_transform = np.linalg.inv(_C2W)
        if K is not None:
            self.projection_matrix = getProjectionMatrix_refine(K, self.image_height, self.image_width, self.znear, self.zfar)
        else:
            self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, fovx, fovy)
        self.full_proj_transform = self.projection_matrix @ self.world_view_transform
        self.camera_center = T

    def to_tensor(self, device='cpu'):
        '''matrix is in glm form, data is ready for training.
        '''
        dtype = torch.float32
        if self.Rh is not None: self.Rh = torch.tensor(self.Rh, dtype=dtype, device=device)
        if self.Th is not None: self.Th = torch.tensor(self.Th, dtype=dtype, device=device)
        if self.poses is not None: self.poses = torch.tensor(self.poses, dtype=dtype, device=device)
        if self.shapes is not None: self.shapes = torch.tensor(self.shapes, dtype=dtype, device=device)

        assert len(self.Rh.shape) == 2 and len(self.Th.shape) == 1
        assert len(self.poses.shape) == 1 and len(self.shapes.shape) == 1

        self.world_view_transform = torch.tensor(self.world_view_transform.T, dtype=dtype, device=device)
        self.full_proj_transform =  torch.tensor(self.full_proj_transform.T, dtype=dtype, device=device)
        self.camera_center =  torch.tensor(self.camera_center, dtype=dtype, device=device)

        self.original_image = torch.tensor(self.original_image.transpose(2,0,1), dtype=dtype, device=device)
        self.original_image = torch.clamp(self.original_image, 0, 1)
        self.mask = torch.tensor(self.mask.astype(np.float32), device=device)[None,...] # 1HW
        return self 
    
    def to_cuda(self):
        self.Rh = self.Rh.cuda(non_blocking=True)
        self.Th = self.Th.cuda(non_blocking=True)
        self.poses = self.poses.cuda(non_blocking=True)
        self.shapes = self.shapes.cuda(non_blocking=True)

        self.world_view_transform = self.world_view_transform.cuda(non_blocking=True)
        self.full_proj_transform = self.full_proj_transform.cuda(non_blocking=True)
        self.camera_center = self.camera_center.cuda(non_blocking=True)
        self.original_image = self.original_image.cuda(non_blocking=True)
        self.mask = self.mask.cuda(non_blocking=True)
        return self

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

