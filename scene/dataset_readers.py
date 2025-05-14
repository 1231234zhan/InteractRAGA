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
from typing import NamedTuple
from utils.graphics_utils import focal2fov
import numpy as np
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud
from scene.cameras import Camera
from copy import deepcopy
from os import path
import cv2 as cv
from scipy.spatial.transform import Rotation

import scene.smpl as SMPL

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    body_param: dict
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        cam_centers.append(cam.T[...,None])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


##################################   ZJUMoCapRefine   ##################################

def readCamerasZJUMoCapRefine(data_dir, annots, cam_id_list, frame_id, background=0, image_scaling=1):
    cam_infos = []

    param = np.load(path.join(data_dir, f'params/{frame_id}.npy'), allow_pickle=True).item()
    poses, shapes, Th, Rh = np.squeeze(param['poses']), np.squeeze(param['shapes']), np.squeeze(param['Th']), np.squeeze(param['Rh'])
    Rh = Rotation.from_rotvec(Rh).as_matrix()

    for cam_id in cam_id_list:
        # Load camera
        T, R, K, D = annots['cams']['T'][cam_id], annots['cams']['R'][cam_id], annots['cams']['K'][cam_id], annots['cams']['D'][cam_id]
        # T: 31  R: 33  K: 33  D: 15
        T, R, K, D = np.squeeze(T.copy()) / 1000, np.squeeze(R.copy()), np.squeeze(K.copy()), np.squeeze(D.copy())
        # zju camera: W2C  down y  transform to : C2W   down y
        R, T = R.T, -R.T @ T
        
        # Load image
        image_path = path.join(data_dir, f'images/{cam_id:02d}/{frame_id:06d}.jpg')
        image_name = path.basename(image_path)
        image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)  # RGB 3C u8

        msk_path = image_path.replace('images', 'mask').replace('jpg', 'png')
        msk = cv.imread(msk_path)   # 1C u8

        image = cv.undistort(image, K, D[None,...])
        msk = cv.undistort(msk, K, D[None,...])

        # Reduce the image resolution by ratio, then remove the back ground
        if image_scaling != 1:
            H, W = int(image.shape[0] * image_scaling), int(image.shape[1] * image_scaling)
            image = cv.resize(image, (W, H), interpolation=cv.INTER_AREA)
            msk = cv.resize(msk, (W, H), interpolation=cv.INTER_NEAREST)
            K[:2] = K[:2] * image_scaling

        image = image.astype(np.float32) / 255
        msk = msk[:,:,0] > 30 if len(msk.shape) == 3 else msk > 30
        # msk = sm.binary_erosion(msk)
        image[~msk] = background

        focalX = K[0,0]
        focalY = K[1,1]
        FovX = focal2fov(focalX, image.shape[1])
        FovY = focal2fov(focalY, image.shape[0])

        cam_info = Camera(
            image=image,
            image_name=image_name,
            fovx=FovX,
            fovy=FovY,
            T=T,
            R=R,
            K=K,
        )
        cam_info.mask = msk
        cam_info.frame_id = frame_id
        cam_info.cam_id = cam_id
        cam_info.Th = Th
        cam_info.Rh = Rh
        cam_info.poses = poses
        cam_info.shapes = shapes
        cam_infos.append(cam_info)
    return cam_infos

def generateInitRandPoint(vert, face, pts_num=7000):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vert)
    mesh.triangles = o3d.utility.Vector3iVector(face)
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=pts_num, init_factor=5)
    point_cloud = np.asarray(point_cloud.points)
    return point_cloud

def readZJUMoCapRefineInfo(data_dir, background, frame_id_list, cam_id_list, image_scaling=1):

    annots = np.load(path.join(data_dir, 'annots.npy'), allow_pickle=True).item()

    train_cam_infos = []
    test_cam_infos = []

    for frame_id in frame_id_list:
        train_cam_infos.extend(readCamerasZJUMoCapRefine(data_dir, annots, cam_id_list, frame_id, background, image_scaling))

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(cam_id_list) == 1: nerf_normalization['radius'] = 1

    # read global info
    # read SDF mesh
    plydata = PlyData.read(path.join(data_dir, 'lbs/bigpose_mesh.ply'))
    mesh_xyz = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']), axis=-1)
    mesh_face = np.array(plydata['face']['vertex_indices'].tolist())

    # initial gaussian generation
    ply_path = path.join(data_dir, f'gaussian/points3D.ply')
    if not path.exists(ply_path):
        print(f"Generating random point cloud ...")
        xyz = generateInitRandPoint(mesh_xyz, mesh_face)
        rgb = np.full((xyz.shape[0], 3), 128, dtype=np.float32)

        os.makedirs(path.join(data_dir, f'gaussian'), exist_ok=True)
        storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)

    # load smpl params
    shapes = train_cam_infos[0].shapes
    bf = SMPL.f
    b_joints, bxyz = SMPL.get_joint_vert(SMPL.big_poses, shapes)
    t_joints, _ = SMPL.get_joint_vert(SMPL.t_poses, shapes)
    weights = SMPL.weights

    body_param = {
        'shapes': shapes,
        'mesh_xyz': mesh_xyz,
        'mesh_f': mesh_face,
        'b_xyz': bxyz,
        'b_f': bf,
        'weights': weights,
        't_joints': t_joints,
        'b_joints': b_joints,
    }

    scene_info = SceneInfo(point_cloud=pcd,
                           body_param=body_param,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,)
    return scene_info

##################################   ZJUMoCapRefine   ##################################

##################################   Blender zju format (with bigpose mesh) ##################################

def readCamerasBlender(data_dir, annots, cam_id_list, frame_id, background=0, image_scaling=1):
    cam_infos = []
    cameras = annots['camera']

    # smpl param
    param = np.load(path.join(data_dir, f'params/{frame_id}.npy'), allow_pickle=True).item()
    poses, shapes, Th, Rh = np.squeeze(param['poses']), np.squeeze(param['shapes']), np.squeeze(param['Th']), np.squeeze(param['R'])

    for cam_id in cam_id_list:
        cam = deepcopy(cameras[cam_id])

        # cameras
        T, R, fovx, fovy = cam['T'], cam['R'], cam['fovx'], cam['fovy']
        R[:3,1:3] *= -1   # blender camera: up y     # gaussian camera: down y
        
        # image and mask
        image = cv.imread(path.join(data_dir, f'images/{cam_id:02d}/{frame_id:06d}.jpg'))
        mask = cv.imread(path.join(data_dir, f'mask/{cam_id:02d}/{frame_id:06d}.png'))

        if image_scaling != 1:
            H, W = int(image.shape[0] * image_scaling), int(image.shape[1] * image_scaling)
            image = cv.resize(image, (W, H), interpolation=cv.INTER_AREA)
            mask = cv.resize(mask, (W, H), interpolation=cv.INTER_NEAREST)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32) / 255
        mask = mask[:,:,0] > 30 if len(mask.shape) == 3 else mask > 30
        image[~mask] = background

        cam_info = Camera(
            image=image,
            image_name=f'{frame_id:06d}.jpg',
            T=T,
            R=R,
            fovx=fovx,
            fovy=fovy,
        )
        cam_info.mask = mask
        cam_info.frame_id = frame_id
        cam_info.cam_id = cam_id
        cam_info.Th = Th
        cam_info.Rh = Rh
        cam_info.poses = poses
        cam_info.shapes = shapes
        cam_infos.append(cam_info)
    return cam_infos

def readBlenderInfo(data_dir, background, frame_id_list, cam_id_list, image_scaling=1):
    annots = np.load(path.join(data_dir, 'params.npy'), allow_pickle=True).item()

    train_cam_infos = []
    test_cam_infos = []

    for frame_id in frame_id_list:
        train_cam_infos.extend(readCamerasBlender(data_dir, annots, cam_id_list, frame_id, background, image_scaling))

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if len(cam_id_list) == 1: nerf_normalization['radius'] = 1

    # read global info
    # read SDF mesh
    plydata = PlyData.read(path.join(data_dir, 'lbs/bigpose_mesh.ply'))
    mesh_xyz = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']), axis=-1)
    mesh_face = np.array(plydata['face']['vertex_indices'].tolist())

    # initial gaussian generation
    ply_path = path.join(data_dir, f'gaussian/points3D.ply')
    if not path.exists(ply_path):
        print(f"Generating random point cloud ...")
        xyz = generateInitRandPoint(mesh_xyz, mesh_face)
        rgb = np.full((xyz.shape[0], 3), 128, dtype=np.float32)

        os.makedirs(path.join(data_dir, f'gaussian'), exist_ok=True)
        storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)

    # load smpl params
    shapes = train_cam_infos[0].shapes
    bf = SMPL.f
    b_joints, bxyz = SMPL.get_joint_vert(SMPL.big_poses, shapes)
    t_joints, _ = SMPL.get_joint_vert(SMPL.t_poses, shapes)
    weights = SMPL.weights

    body_param = {
        'shapes': shapes,
        'mesh_xyz': mesh_xyz,
        'mesh_f': mesh_face,
        'b_xyz': bxyz,
        'b_f': bf,
        'weights': weights,
        't_joints': t_joints,
        'b_joints': b_joints,
    }

    scene_info = SceneInfo(point_cloud=pcd,
                           body_param=body_param,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,)
    return scene_info


##################################   Blender zju format (with bigpose mesh) ##################################