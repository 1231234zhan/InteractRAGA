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

import os
from os import path
import random
import json
import numpy as np
import torch
from scene.dataset_readers import readZJUMoCapRefineInfo, readBlenderInfo
from scene.gaussian_model import GaussianModel
from utils.camera_utils import camera_to_JSON, pose_to_JSON

from typing import List
from scene.cameras import Camera

def get_vert_normal(vert, face):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vert)
    mesh.triangles = o3d.utility.Vector3iVector(face)
    mesh.compute_vertex_normals()
    vert_normal = np.asarray(mesh.vertex_normals)
    return vert_normal

def get_point_weight(vert, face, weights, pts):
    import trimesh
    mesh = trimesh.Trimesh(vertices=vert, faces=face)
    close_pts, _ , tri_id = trimesh.proximity.closest_point(mesh, pts)
    closetri_id = face[tri_id]
    closetri_pts = vert[closetri_id] 
    bc_crood = trimesh.triangles.points_to_barycentric(closetri_pts, close_pts)

    closetri_weights = weights[closetri_id] 
    close_weights = np.einsum('ni,nic->nc', bc_crood, closetri_weights)
    return close_weights

def get_sample_points_poisson_disk(vert, face, pts_num):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vert)
    mesh.triangles = o3d.utility.Vector3iVector(face)
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=pts_num, init_factor=15)
    point_cloud = np.asarray(point_cloud.points)
    return point_cloud

def get_decimation_mesh(vert, face, faces_num):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vert)
    mesh.triangles = o3d.utility.Vector3iVector(face)
    mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, faces_num)    
    vert = np.asarray(mesh.vertices).astype(np.float32)
    face = np.asarray(mesh.triangles).astype(int)
    return vert, face

def storePlyMesh(out_path, vert, face):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vert)
    mesh.triangles = o3d.utility.Vector3iVector(face)
    o3d.io.write_triangle_mesh(out_path, mesh)    

def precompute_visibility(gaussians: GaussianModel, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    gaussians.compute_knn()
    with torch.set_grad_enabled(False):
        for key in gaussians.all_poses:
            frame_id = int(key)
            out_path = path.join(out_dir, str(frame_id)+'.pt')
            if path.exists(out_path): 
                # read from filesystem in each iteration is SLOW, so we load the visibility to cpu memory
                vis_map = torch.load(out_path, weights_only=True)
            else:
                gaussians.smpl_poses = gaussians.all_poses[str(frame_id)]
                gaussians.Th, gaussians.Rh = gaussians.all_Th[str(frame_id)], gaussians.all_Rh[str(frame_id)]

                vis_map = torch.clamp(gaussians.get_vb_visibility() * 255, 0, 255).type(torch.uint8).cpu()
                torch.save(vis_map, out_path)

            gaussians.all_vertvb_visib[str(frame_id)] = vis_map

    gaussians.cache_dict = {}

class Scene:

    gaussians : GaussianModel

    def __init__(self, args, gaussians : GaussianModel, shuffle=True):
        self.model_path = args.model_path
        self.gaussians = gaussians

        self.train_cameras = []
        self.test_cameras = []

        dataset_type = None
        if path.exists(path.join(args.source_path, 'params.npy')):
            dataset_type = 'blender'
        elif os.path.exists(os.path.join(args.source_path, "annots.npy")):
            print("Found annots.npy, ZJUMoCap dataset")
            dataset_type = 'zjumocap'
        else:
            assert False, "Could not recognize scene type!"

        frame_ids = np.arange(args.begin_ith_frame, args.frame_interval*args.num_train_frame, args.frame_interval).tolist()
        cam_ids = np.array(args.train_cam_ids).tolist()
        image_scaling = args.image_scaling

        if dataset_type == 'zjumocap':
            scene_info = readZJUMoCapRefineInfo(args.source_path, np.array(args.background), frame_ids, cam_ids, image_scaling)
        if dataset_type == 'blender':
            scene_info = readBlenderInfo(args.source_path, np.array(args.background), frame_ids, cam_ids, image_scaling)

        # dump camera to json for further visualization
        if True:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras: camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras: camlist.extend(scene_info.train_cameras)
            dump_frame_id = -1
            for id, cam in enumerate(camlist):
                if dump_frame_id == -1: dump_frame_id = cam.frame_id
                if cam.frame_id != dump_frame_id: continue
                json_cams.append(camera_to_JSON(cam.cam_id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

            # dump poses to json for further visualization
            dump_cam_id = -1
            json_poses = []
            for id, cam in enumerate(camlist):
                if dump_cam_id == -1: dump_cam_id = cam.cam_id
                if cam.cam_id != dump_cam_id: continue
                json_poses.append(pose_to_JSON(cam))
            with open(os.path.join(self.model_path, "poses.json"), 'w') as file:
                json.dump(json_poses, file)            
        print(f'Training images: {len(scene_info.train_cameras)}')

        if shuffle:
            random.shuffle(scene_info.train_cameras)  
            random.shuffle(scene_info.test_cameras)  

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # collect all the poses, Th, Rh
        dump_cam_id = -1
        all_poses, all_Th, all_Rh = {}, {}, {}
        for id, cam in enumerate(camlist):
            if dump_cam_id == -1: dump_cam_id = cam.cam_id
            if cam.cam_id != dump_cam_id: continue
            frame_id = cam.frame_id
            all_poses[str(frame_id)] = cam.poses
            all_Th[str(frame_id)] = cam.Th
            all_Rh[str(frame_id)] = cam.Rh

        print("Loading Cameras")
        self.train_cameras = [cam.to_tensor() for cam in scene_info.train_cameras]
        self.test_cameras = [cam.to_tensor() for cam in scene_info.test_cameras]

        if args.data_device == 'cuda':
            self.train_cameras = [cam.to_cuda() for cam in self.train_cameras]
            self.test_cameras = [cam.to_cuda() for cam in self.test_cameras]

        if dataset_type == 'blender' or dataset_type == 'zjumocap':
            bparam = scene_info.body_param
            mesh_xyz = bparam['mesh_xyz']
            t_joints = bparam['t_joints']

            vert_normal_path = path.join(args.source_path, 'gaussian/vert_normal.npy')
            if path.exists(vert_normal_path): vert_normal = np.load(vert_normal_path)
            else: 
                vert_normal = get_vert_normal(bparam['mesh_xyz'], bparam['mesh_f'])
                np.save(vert_normal_path, vert_normal)

            vert_weights_path = path.join(args.source_path, 'gaussian/vert_weights.npy')
            if path.exists(vert_weights_path):
                vert_weights = np.load(vert_weights_path)
            else:
                vert_weights = get_point_weight(bparam['b_xyz'], bparam['b_f'], bparam['weights'], mesh_xyz)
                np.save(vert_weights_path, vert_weights)   

            vert_f = bparam['mesh_f'].astype(int)

        self.gaussians.create_from_pcd(
            pcd=scene_info.point_cloud, 
            spatial_lr_scale=self.cameras_extent, 
            vert=mesh_xyz,
            knn_K=args.knn_K,
            vert_normal=vert_normal,
            vert_weights=vert_weights,
            vert_f=vert_f,
            t_joints=t_joints,
            all_poses=all_poses,
            all_Th=all_Th,
            all_Rh=all_Rh,
        )
        self.gaussians.is_specular = args.is_specular
        self.gaussians.is_visibility = args.is_visibility
        self.gaussians.is_dxyz = args.is_dxyz
        print(f'Use specular: {args.is_specular}    use visibilty: {args.is_visibility}')

        # If visibility is used in training, we pre-compute it
        if args.is_visibility:
            print('Pre-compute visibility, it may take some time...')
            precompute_visibility(self.gaussians, path.join(args.source_path, 'gaussian/visibility'))            

        self.gaussians.compute_knn()

    def getTrainCameras(self) -> List[Camera]:
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras