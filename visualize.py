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

IMAGE_ENCODE = 'gpu'

import os
from os import path
import time
import torch
import torch.nn.functional as F
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import encode_bytes, normal_to_color, linear_to_srgb
from utils.camera_utils import load_json_camera, load_json_pose
from argparse import ArgumentParser
import numpy as np
import pickle
from scipy.spatial.transform import Rotation
import scene.smpl as SMPL
from scene.edit import optimize_edit

from utils.net_utils import net_init, wait_connection, is_connected, receive, send

def check_poses_change(pose1, pose2, Rh1, Rh2):
    if not torch.allclose(pose1, pose2, atol=1e-4): return True
    if not torch.allclose(Rh1, Rh2, atol=1e-4): return True
    return False

def get_render_choice(gaussians: GaussianModel, client_data):
    render_type, cam = client_data['render_type'], client_data['camera']
    override_color, is_opacity = None, True
    if render_type == 'albedo':
        override_color, is_opacity = gaussians.get_albedo, True
        override_color = linear_to_srgb(override_color)
    if render_type == 'normal':
        is_opacity = True
        nor = gaussians.get_normal
        nor = torch.einsum('ij,nj->ni', cam.world_view_transform[:3,:3].T, nor) 
        nor[:,1] *= -1
        nor[:,2] *= -1
        color = normal_to_color(nor)
        override_color = color                 
    if render_type == 'roughness':
        is_opacity = False
        rough = gaussians.get_roughness
        override_color = torch.tile(rough, [1,3])
    if render_type == 'specularTint':
        is_opacity = False
        tint = gaussians.get_specularTint
        override_color = torch.tile(tint, [1,3])
    if render_type == 'visibility':
        is_opacity = True
        visib = gaussians.get_visiblity
        albedo = torch.full_like(gaussians.get_albedo, 0.5)
        wo = F.normalize(cam.camera_center - gaussians.get_xyz)
        nor = gaussians.get_normal
        visib = gaussians.get_visiblity
        override_color = gaussians.envmap.env_integral_diffuse(wo, nor, albedo, visibility=visib)
        override_color = linear_to_srgb(override_color)

    return override_color, is_opacity 

def visualizing(dataset, opt, pipe, model_path):
    first_iter = 0
    SMPL.init(dataset.smpl_pkl_path)
    gaussians = GaussianModel()

    # find ckpt with largest iter number
    ckpt_path = path.join(model_path, sorted([pth for pth in os.listdir(model_path) if 'chkpnt' in pth and '.pth' in pth], key=lambda s: (len(s), s))[-1])

    load_data = torch.load(ckpt_path, weights_only=False)
    first_iter = load_data['iteration']
    print(f'Loading checkpoint from ITER {first_iter}')
    gaussians.restore(load_data)

    gaussians.is_specular, gaussians.is_visibility, gaussians.is_dxyz = False, False, True
    gaussians.is_train = False
    gaussians.is_novel_pose = True

    background = dataset.background if isinstance(dataset.background, list) else [dataset.background] * 3
    background_init = torch.tensor(background, dtype=torch.float32, device='cuda')
    
    # load training camera
    camera_path = path.join(model_path, 'cameras.json')
    cam_info_list = load_json_camera(camera_path)

    # load poses
    poses_path = path.join(model_path, 'poses.json')
    pose_info_list = load_json_pose(poses_path)

    # initial data contains: cameras, poses
    is_send_initial_data = True

    envmap_init = gaussians.envmap.envmap
    _envmap_init = gaussians.envmap._envmap
    init_Rh = torch.tensor(Rotation.from_euler('x', np.pi/2).as_matrix()).float().cuda()
    init_Th = torch.tensor([0,0,1.1]).float().cuda()

    while True:
        status = wait_connection()
        if status == 'connected':
            is_send_initial_data = True

        if not is_connected(): continue

        data = {}

        net_image_bytes = None
        client_data = receive()
        if client_data != None:
            if 'edit_texture' in client_data:
                texture = torch.tensor(client_data['edit_texture']).cuda()
                tex_pos = torch.tensor(client_data['edit_texture_position']).cuda()
                gaussians.envmap._envmap = _envmap_init
                optimize_edit(
                    gaussians=gaussians,
                    cam=client_data['camera'],
                    edit_texture=texture,
                    edit_position=tex_pos,
                    optim_args=opt,
                    pipe_args=pipe,
                    out_dir='/tmp/model',
                    model_dir=model_path,
                    pose=torch.tensor(client_data['poses']).float().cuda(),
                    Th=torch.tensor(client_data['Th']).float().cuda(),
                    Rh=torch.tensor(client_data['Rh']).float().cuda(),
                    background=torch.tensor(client_data['background']).float().cuda(),
                )
                # while wait_connection() != 'connected': pass

            envmap = torch.clone(envmap_init)
            background = torch.clone(background_init)

            if 'load_envmap' in client_data and client_data['load_envmap'] is not None:
                envmap = torch.tensor(client_data['load_envmap'], device='cuda')

            if 'env_rot' in client_data:
                env_rot = np.floor(client_data['env_rot'] * 32).astype(int)
                envmap = torch.roll(envmap, env_rot, dims=1)

            if 'env_brightness' in client_data:
                envmap *= client_data['env_brightness']

            poses = torch.tensor(SMPL.big_poses).float().cuda(non_blocking=True)
            if 'poses' in client_data:
                poses = torch.tensor(client_data['poses']).float().cuda(non_blocking=True)

            Rh = init_Rh
            if 'Rh' in client_data:
                Rh = torch.tensor(client_data['Rh']).float().cuda(non_blocking=True)

            Th = init_Th
            if 'Th' in client_data:
                Th = torch.tensor(client_data['Th']).float().cuda(non_blocking=True)

            if 'specular' in client_data:
                gaussians.is_specular = client_data['specular']

            if 'visibility' in client_data:
                gaussians.is_visibility = client_data['visibility']

            if 'is_novel_pose' in client_data:
                gaussians.is_novel_pose = client_data['is_novel_pose']

            if 'dxyz' in client_data:
                gaussians.is_dxyz = client_data['dxyz']

            if 'background' in client_data:
                background = torch.tensor(client_data['background']).float().cuda(non_blocking=True)

            if 'get_visibility' in gaussians.cache_dict and not check_poses_change(gaussians.smpl_poses, poses, gaussians.Rh, Rh):
                cache_visib = gaussians.cache_dict['get_visibility']
            else:
                cache_visib = None

            gaussians.envmap.envmap = envmap
            gaussians.Rh, gaussians.Th = Rh, Th
            gaussians.smpl_poses = poses

            if cache_visib is not None: gaussians.cache_dict['get_visibility'] = cache_visib

            with torch.set_grad_enabled(False):
                override_color, is_opacity = get_render_choice(gaussians, client_data)

                net_image = render(client_data['camera'], gaussians, pipe, background, client_data['scaling_modifier'],\
                                    override_color=override_color, is_opacity=is_opacity)["render"]
            net_image = (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            net_image_bytes = encode_bytes(net_image, IMAGE_ENCODE)
            data['image_bytes'] = net_image_bytes

            data['gaussian_num'] = len(gaussians._xyz)
            
            if is_send_initial_data:
                is_send_initial_data = False
                data['cameras'] = cam_info_list
                data['poses'] = pose_info_list

            if 'edit_texture' in client_data:
                data['is_optimize_edit_finish'] = True

            send(pickle.dumps(data))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Visualizing script parameters")
    from arguments import load_yaml, set_param_attribute
    parser.add_argument('-m', '--model_path', type=str, default='')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=23456)

    pargs = parser.parse_args(sys.argv[1:])

    pargs.config_path = path.join(pargs.model_path, 'config.yaml')
    args_dict = load_yaml(pargs.config_path)
    args_dict['ip'], args_dict['port'] = pargs.ip, pargs.port

    args = set_param_attribute(args_dict)

    print("Visualizing " + pargs.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    net_init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    visualizing(
        dataset=args.model, 
        opt=args.optimization, 
        pipe=args.pipeline, 
        model_path=pargs.model_path,
    )
