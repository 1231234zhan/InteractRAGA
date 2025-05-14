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

from scene.cameras import Camera
import numpy as np
import torch
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):

    resolution = cam_info.image.size
    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, pose_id=cam_info.pose_id, R=cam_info.R, T=cam_info.T, K=cam_info.K, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, bkgd_mask=cam_info.bkgd_mask, 
                  bound_mask=cam_info.bound_mask, smpl_param=cam_info.smpl_param, 
                  world_vertex=cam_info.world_vertex, world_bound=cam_info.world_bound, 
                  data_device=args.data_device)


def camera_to_JSON(id, camera : Camera):
    pos = camera.T
    rot = camera.R
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.image_width,
        'height' : camera.image_height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FoVy, camera.image_height),
        'fx' : fov2focal(camera.FoVx, camera.image_width)
    }
    return camera_entry

import json
def load_json_camera(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    cam_info_list =[]
    for cam_data in data:
        cam_data['rotation'] = np.array(cam_data['rotation'], dtype=np.float32)
        cam_data['position'] = np.array(cam_data['position'], dtype=np.float32)
        cam_info_list.append(cam_data)

    return cam_info_list

def pose_to_JSON(camera : Camera):
    camera_entry = {
        'frame_id' : int(camera.frame_id),
        'poses': camera.poses.tolist(),
        'Th': camera.Th.tolist(),
        'Rh': [x.tolist() for x in camera.Rh],
    }
    return camera_entry

def load_json_pose(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    pose_info_list =[]
    for pose_data in data:
        pose_data['poses'] = np.array(pose_data['poses'], dtype=np.float32)
        pose_info_list.append(pose_data)

    return pose_info_list 