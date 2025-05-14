import os
from os import path
import numpy as np
import cv2 as cv
import imageio.v3 as iio
import skimage.morphology as sm
import copy
import tqdm
import torch
import shutil

import nvdiffrast.torch as dr
from pytorch3d.transforms import quaternion_to_matrix
import torch.nn.functional as F

from scene import GaussianModel, Camera
from gaussian_renderer import render
from utils.image_utils import linear_to_srgb
from utils.loss_utils import l1_loss, ssim

class Raster():
    f=None
    vert_enum=None
    vert_Gn=None
    def __init__(self, device='cuda'):
        self.glctx = dr.RasterizeGLContext(output_db=False)
        self.f = None

    def update(self, f):
        f = f.type(torch.int64)
        self.f = f

    def get_rasterization(self, xyz, cam: Camera):
        assert self.f is not None
        V_num = len(xyz)

        full_proj_transform = cam.full_proj_transform.T

        # world to ndc
        xyz = torch.einsum('ij,vj->vi', full_proj_transform, F.pad(xyz, [0,1], value=1))
        xyz = xyz[None].contiguous()

        resolution = [cam.image_height, cam.image_width]
        rast, _ = dr.rasterize(self.glctx, xyz, self.f.type(torch.int32), resolution=resolution)
        rast = rast[...,3].type(torch.int64)[0]
        
        rast = rast - 1
        return rast


def gaussian_scale_loss(gs: GaussianModel, mask):
    scaling = gs.scaling_activation(gs.vert_scaling)
    loss = F.relu(scaling[mask] - 0.005).mean()
    return loss

def gaussian_normal_scale_loss(gs: GaussianModel, mask):
    normal = gs.get_normal[mask]
    rotation = gs.get_rotation[mask]
    scaling = gs.get_scaling[mask]
    vec = torch.einsum('nji,nj->ni', quaternion_to_matrix(rotation), normal)
    scaling_nor = torch.norm(scaling * vec, dim=-1)

    loss = F.relu(scaling_nor - 0.001).mean()
    return loss

def gaussian_mesh_dist_loss(gs: GaussianModel, mask):
    dist = torch.norm((gs.vert_xyz[gs.knn_idx] - gs._xyz[:,None]), dim=-1)
    dist = torch.where(gs.knn_idx<0, torch.tensor(0), dist)
    return torch.mean(dist[mask])

def set_gaussian_status(gaussians: GaussianModel, status='edit'):
    gaussians.is_train = True
    gaussians.is_dxyz = True

    gaussians.vert_albedo.requires_grad_(True)
    gaussians._xyz.requires_grad_(True)
    gaussians.vert_scaling.requires_grad_(True)
    gaussians.vert_rotation.requires_grad_(True)

    gaussians.vert_roughness.requires_grad_(False)
    gaussians.vert_specularTint.requires_grad_(False)
    gaussians.all_poses.requires_grad_(False)
    gaussians.mlp_dxyz.requires_grad_(False)

def set_optim_status(opt):
    opt.iterations = 3000
    opt.position_lr_init = 0.0000016
    opt.position_lr_final = 0.0000016
    opt.position_lr_delay_mult = 1.0
    opt.position_lr_max_steps = 3000

    opt.densify_from_iter = 0
    opt.densify_until_iter = 1601
    opt.densify_grad_threshold = 0.0001
    opt.densification_interval = 100

    opt.albedo_lr = 0.05


def copy_model_files(model_dir, out_dir):
    shutil.copy(
        src=path.join(model_dir, 'cameras.json'),
        dst=path.join(out_dir, 'cameras.json'),
    )
    shutil.copy(
        src=path.join(model_dir, 'poses.json'),
        dst=path.join(out_dir, 'poses.json'),
    )
    shutil.copy(
        src=path.join(model_dir, 'config.yaml'),
        dst=path.join(out_dir, 'config.yaml'),
    )

def optimize_edit(gaussians: GaussianModel, cam: Camera, edit_texture, edit_position, optim_args, pipe_args, out_dir, model_dir,
                  pose, Th, Rh, background):

    texture, tex_pos = edit_texture, edit_position
    tex_H, tex_W = texture.shape[:2]
    H, W = cam.image_height, cam.image_width
    mask_tex = texture[...,3] > 0.1

    os.makedirs(out_dir, exist_ok=True)
    copy_model_files(model_dir, out_dir)

    gaussians.compute_knn()
    gaussians.smpl_poses =  torch.cat((pose[:3].detach(), pose[3:]), dim=-1) 
    gaussians.Th, gaussians.Rh = Th, Rh

    # render sample image
    with torch.set_grad_enabled(False):
        bg = background
        override_color = linear_to_srgb(gaussians.get_albedo)
        image_gt = render(viewpoint_camera=cam, pc=gaussians, pipe=pipe_args, bg_color=bg, override_color=override_color, is_opacity=True)['render']
        image_gt = image_gt.permute(1,2,0)

    # edit image and mask
    mask_gt = torch.zeros((H, W), dtype=bool).cuda()
    bbox = torch.tensor([max(tex_pos[0], 0), max(tex_pos[1], 0), min(tex_pos[0]+tex_W, W), min(tex_pos[1]+tex_H, H)])
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]: 
        return False
    bbox_t = bbox - torch.tensor([tex_pos[0], tex_pos[1], tex_pos[0], tex_pos[1]])
    texture_crop = texture[bbox_t[1]:bbox_t[3],bbox_t[0]:bbox_t[2]]
    
    image_gt_crop = image_gt[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    image_gt[bbox[1]:bbox[3],bbox[0]:bbox[2]] = ( texture_crop[...,:3] - image_gt_crop) * texture_crop[...,3:] + image_gt_crop
    mask_gt[bbox[1]:bbox[3],bbox[0]:bbox[2]] = mask_tex[bbox_t[1]:bbox_t[3],bbox_t[0]:bbox_t[2]]
    mask_gt = sm.binary_dilation(mask_gt.cpu().numpy(), sm.disk(4))
    mask_gt = torch.tensor(mask_gt).cuda()

    # get trainable vertex mask
    rast = Raster()
    rast.update(gaussians.vert_f)
    visib = rast.get_rasterization(gaussians.get_vb_xyz, cam)
    
    mask_flat = mask_gt.reshape(-1)
    visib_fids = torch.unique(visib.reshape(-1)[mask_flat])
    if visib_fids[0] < 0: visib_fids = visib_fids[1:]
    visib_vids = torch.unique(gaussians.vert_f[visib_fids])
    visib_vmask = torch.zeros(len(gaussians.vert_xyz), dtype=bool).cuda()
    visib_vmask[visib_vids] = True
    visib_vmask = torch.median(visib_vmask[gaussians.neighbor_idx].type(torch.long), dim=-1)[0].type(torch.bool)

    # set gaussian model status
    set_gaussian_status(gaussians)
    opt = copy.deepcopy(optim_args)
    set_optim_status(opt)
    checkpoint_iterations = [2999]
    gaussians.training_setup(opt)

    # training
    progress_bar = tqdm.tqdm(range(0, opt.iterations), initial=0, desc="TP")
    ema_loss_for_log = 0.0
    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        gaussians.compute_knn()
        visib_gmask = torch.sum(visib_vmask[gaussians.knn_idx], dim=-1) > 0

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        override_color = linear_to_srgb(gaussians.get_albedo)
        render_pkg = render(viewpoint_camera=cam, pc=gaussians, pipe=pipe_args, bg_color=bg, override_color=override_color, is_opacity=True)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = image_gt.permute(2,0,1).float().cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        scaling_loss = gaussian_scale_loss(gaussians, visib_vmask) * opt.lambda_scaling
        gs_mesh_dist_loss = gaussian_mesh_dist_loss(gaussians, visib_gmask) * opt.lambda_mesh_dist
        normal_scale_loss = gaussian_normal_scale_loss(gaussians, visib_gmask) * 10
        all_loss = loss + scaling_loss + gs_mesh_dist_loss + normal_scale_loss
        all_loss.backward()
        
        # clear untrainable vertex gradient
        gaussians.vert_albedo.grad[~visib_vmask] = 0
        gaussians.vert_scaling.grad[~visib_vmask] = 0
        gaussians.vert_rotation.grad[~visib_vmask] = 0
        gaussians._xyz.grad[~visib_gmask] = 0
        visibility_filter[~visib_gmask] = False

        # applying 3DGS's densification
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"L": f"{ema_loss_for_log:.{4}f}" ,'gs': f'{len(gaussians._xyz)}'})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification
            if iteration < opt.densify_until_iter:
                visibility_filter = visibility_filter & (gaussians.get_opacity(cam.camera_center) > 0.005)[:,0]
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 4, size_threshold)
                
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                save_data = gaussians.capture()
                save_data['iteration'] = iteration
                torch.save(save_data, path.join(out_dir, "chkpnt" + str(iteration) + ".pth"))

    gaussians.is_train = False
