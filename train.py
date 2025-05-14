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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, mlp_dxyz_loss, gaussian_scale_loss, gaussian_mesh_dist_loss, roughness_smooth_loss, albedo_smooth_loss, specularTint_smooth_loss
from gaussian_renderer import render
from utils.net_utils import receive, send, net_init, is_connected
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, encode_bytes
from argparse import ArgumentParser, Namespace
import numpy as np
from copy import deepcopy
import pickle
from tensorboardX import SummaryWriter
import scene.smpl as SMPL
from scene.cameras import Camera

def update_gs(gaussians: GaussianModel, cam: Camera):
    gaussians.compute_knn()
    
    frame_id = cam.frame_id
    pose =  gaussians.all_poses[str(frame_id)]
    gaussians.smpl_poses =  torch.cat((pose[:3].detach(), pose[3:]), dim=-1) 
    gaussians.Th, gaussians.Rh = gaussians.all_Th[str(frame_id)], gaussians.all_Rh[str(frame_id)]

    if gaussians.is_visibility:
        visib = gaussians.all_vertvb_visib[str(frame_id)]
        visib = visib.cuda().type(torch.float32) / 255
        gaussians.vertvb_visib = visib

def training(dataset, opt, pipe, testing_iterations, checkpoint_iterations):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    SMPL.init(dataset.smpl_pkl_path)

    gaussians = GaussianModel()
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    gaussians.set_freeze_params(False)

    background = dataset.background if isinstance(dataset.background, list) else [dataset.background] * 3
    background = torch.tensor(background, dtype=torch.float32, device='cuda')

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(0, opt.iterations), initial=first_iter, desc="TP")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if is_connected() and iteration % 15 == 0: 
            net_image_bytes = None
            client_data = receive()
            if client_data != None:
                gaussians.reset_pose()
                is_visibility, gaussians.is_visibility = gaussians.is_visibility, False
                net_image = render(client_data['camera'], gaussians, pipe, background, client_data['scaling_modifier'])["render"]
                gaussians.is_visibility = is_visibility
                net_image = (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                net_image_bytes = encode_bytes(net_image, IMAGE_ENCODE)
                data = {'image_bytes':  net_image_bytes}
                send(pickle.dumps(data))
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        if dataset.data_device != 'cuda':
            viewpoint_cam = deepcopy(viewpoint_cam).to_cuda()

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        update_gs(gaussians, viewpoint_cam)
        render_pkg = render(
            viewpoint_camera=viewpoint_cam,
            pc=gaussians,
            pipe=pipe,
            bg_color=bg,
            )
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        dxyz_loss = mlp_dxyz_loss(gaussians) * opt.lambda_mlp_dxyz
        scaling_loss = gaussian_scale_loss(gaussians) * opt.lambda_scaling
        gs_mesh_dist_loss = gaussian_mesh_dist_loss(gaussians) * opt.lambda_mesh_dist
        rough_loss = roughness_smooth_loss(gaussians) * opt.lambda_roughness_smooth
        albedo_loss = albedo_smooth_loss(gaussians) * opt.lambda_albedo_smooth
        specularTint_loss = specularTint_smooth_loss(gaussians) * opt.lambda_specularTint_smooth
        all_loss = loss + scaling_loss + gs_mesh_dist_loss + dxyz_loss + rough_loss + albedo_loss + specularTint_loss
        all_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"L": f"{ema_loss_for_log:.{4}f}", 'dxyz': f'{dxyz_loss:.6f}' ,'gs': f'{len(gaussians._xyz)}'})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            kwargs = dict(pipe=pipe, bg_color=background)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, kwargs)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                visibility_filter = visibility_filter & (gaussians.get_opacity(viewpoint_cam.camera_center) > 0.005)[:,0]
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            if iteration > opt.densify_density_from_iter and iteration < opt.densify_density_until_iter and iteration % opt.densification_interval == 0:
                gaussians.densify_by_density()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                save_data = gaussians.capture()
                save_data['iteration'] = iteration
                torch.save(save_data, scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                gaussians.dump_poses_json(os.path.join(scene.model_path, 'poses.json'))

def prepare_output_and_logger(args):    
       
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(args.model_path)
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    viewpoint = deepcopy(viewpoint).to_cuda()
                    update_gs(scene.gaussians, viewpoint)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, **renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # if tb_writer and (idx < 5):
                    if tb_writer and (idx < 20):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.cam_id), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.cam_id), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    if True:
        from arguments import load_yaml, set_param_attribute, save_yaml
        parser.add_argument('-c', '--config_path', type=str, default='config/default.yaml')
        parser.add_argument('-s', '--source_path', type=str, default='')
        parser.add_argument('-m', '--model_path', type=str, default='')
        parser.add_argument('--ip', type=str, default='127.0.0.1')
        parser.add_argument('--port', type=int, default=6009)
        pargs = parser.parse_args(sys.argv[1:])

        args_dict = load_yaml(pargs.config_path)
        args_dict['model']['source_path'] = pargs.source_path
        args_dict['model']['model_path'] = pargs.model_path
        args_dict['ip'], args_dict['port'] = pargs.ip, pargs.port
        os.makedirs(pargs.model_path, exist_ok=True)

        args_dict['model_path'] = pargs.model_path
        args_dict['iterations'] = args_dict['optimization']['iterations']

        save_yaml(os.path.join(pargs.model_path, 'config.yaml'), args_dict)
        args = set_param_attribute(args_dict)

    data_dir = args.model.source_path
    args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    net_init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args.model, args.optimization, args.pipeline, args.test_iterations, args.checkpoint_iterations)

    # All done
    print("\nTraining complete.")
