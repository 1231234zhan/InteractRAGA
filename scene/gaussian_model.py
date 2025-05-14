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
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
# from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, compute_knn
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.image_utils import linear_to_srgb

import scene.smpl as SMPL
from scene.mlp import MLP, Embedder
from scene.env_map import EnvMap, Visib
import torch.nn.functional as F

from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self):
        self._xyz = torch.empty(0)
        self._opacity = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # vertex feature
        self.vert_xyz = torch.empty(0)  
        self.vert_normal = torch.empty(0) 
        self.vert_albedo = torch.empty(0) 
        self.vert_roughness = torch.empty(0) 
        self.vert_specularTint = torch.empty(0) 
        self.vert_scaling = torch.empty(0) 
        self.vert_rotation = torch.empty(0) 
        self.vert_weights = torch.empty(0) 
        self.vert_f = torch.empty(0)
        self.vert_dxyz_avg = None 

        # KNN point index
        self.knn_idx = torch.empty(0)
        self.knn_weight = torch.empty(0)
        self.knn_K = None
        self.knn_grid = None
        self.neighbor_idx = torch.empty(0)

        # env map
        self.envmap = EnvMap()

        # pose
        self.Rh = torch.empty(0)
        self.Th = torch.empty(0)
        self.Ac_inv = torch.empty(0)
        self._smpl_poses = torch.empty(0)
        self.t_joints = torch.empty(0)

        self.all_poses = nn.ParameterDict()
        self.all_Th = {}
        self.all_Rh = {}

        # displacement
        self.mlp_dxyz = MLP(64+72, 3, 256, 4)
        self.embedder_dxyz = Embedder(3, 64, 3) 

        # visibility
        self.vb_knn_idx = torch.empty(0)
        self.vb_knn_weight = torch.empty(0)

        self.vertvb_visib = None
        self.all_vertvb_visib = {}

        self.visib = Visib()

        # output switch
        self.is_specular = False
        self.is_visibility = False
        self.is_dxyz = True

        # cache
        self.cache_dict = {}
        self.is_train = True
        self.is_novel_pose = False 

        self.setup_functions()

    def capture(self):
        data = {
            '_xyz': self._xyz,
            '_opacity': self._opacity,
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'denom': self.denom,
            'optimizer.state_dict': self.optimizer.state_dict(),
            'spatial_lr_scale': self.spatial_lr_scale,

            'vert_xyz': self.vert_xyz,
            'vert_normal': self.vert_normal,
            'vert_albedo': self.vert_albedo,
            'vert_roughness': self.vert_roughness,
            'vert_scaling': self.vert_scaling,
            'vert_rotation': self.vert_rotation,
            'vert_specularTint': self.vert_specularTint,
            'vert_weights': self.vert_weights,
            'vert_f': self.vert_f,

            'envmap._envmap': self.envmap._envmap,

            'knn_K': self.knn_K,

            't_joints': self.t_joints,
            'all_poses': self.all_poses,
            'all_Th': self.all_Th,
            'all_Rh': self.all_Rh,

            'mlp_dxyz': self.mlp_dxyz.state_dict(),
            'embedder_dxyz': self.embedder_dxyz.state_dict(),
        }
        return data
    
    def restore(self, data):
        def loader(s):
            if s in data: return data[s]
            else: print(f'NO DATA {s}!')
            return None

        self._xyz = data['_xyz']
        self._opacity = loader('_opacity')
        self.max_radii2D = data['max_radii2D']
        self.xyz_gradient_accum = data['xyz_gradient_accum']
        self.denom = data['denom']
        self.spatial_lr_scale = data['spatial_lr_scale']

        self.vert_xyz = data['vert_xyz']
        self.vert_normal = data['vert_normal']
        self.vert_albedo = data['vert_albedo']
        self.vert_roughness = data['vert_roughness']
        self.vert_scaling = data['vert_scaling']
        self.vert_rotation = data['vert_rotation']
        self.vert_specularTint = loader('vert_specularTint')
        self.vert_weights = loader('vert_weights')
        self.vert_f = loader('vert_f')

        self.envmap._envmap = data['envmap._envmap']

        self.knn_K = data['knn_K']

        self.t_joints = loader('t_joints')
        self.all_poses = loader('all_poses')
        self.all_Th = loader('all_Th')
        self.all_Rh = loader('all_Rh')

        self.mlp_dxyz.load_state_dict(loader('mlp_dxyz'))
        self.embedder_dxyz.load_state_dict(loader('embedder_dxyz'))
        self.mlp_dxyz = self.mlp_dxyz.cuda()
        self.embedder_dxyz = self.embedder_dxyz.cuda()

        self.init()

    def init(self):
        self.init_knn()
        self.compute_knn()
        self.init_body()
        self.visib.update(self.envmap.sphe_vec.reshape(-1,3), self.vert_f)

    def dump_poses_json(self, json_path):
        frame_ids = sorted([int(k) for k in self.all_poses])
        data_list = []
        for frame_id in frame_ids:
            data = {
                'frame_id' : frame_id,
                'poses': self.all_poses[str(frame_id)].data.cpu().detach().numpy().tolist(),
                'Th': self.all_Th[str(frame_id)].cpu().numpy().tolist(),
                'Rh': [x.tolist() for x in self.all_Rh[str(frame_id)].cpu().numpy()],                
            }
            data_list.append(data)
        with open(json_path, 'w') as file:
            json.dump(data_list, file)       

    @property
    def get_scaling(self):
        if 'get_scaling' in self.cache_dict: return self.cache_dict['get_scaling']
        knn_scaling = self.scaling_activation(self.vert_scaling)[self.knn_idx]
        _scaling = torch.sum(knn_scaling * self.knn_weight[...,None], dim=1)
        self.cache_dict['get_scaling'] = _scaling            
        return _scaling
    
    @property
    def get_weights(self):
        knn_weights = self.vert_weights[self.knn_idx]
        _weights = torch.sum(knn_weights * self.knn_weight[...,None], dim=1)
        return _weights

    @property
    def get_Gweights(self):
        if 'get_Gweights' in self.cache_dict: return self.cache_dict['get_Gweights']
        R_pose = SMPL.rotvec2mat(self.smpl_poses.reshape(-1,3))
        A = SMPL.transform_matrix(self.t_joints, R_pose)
        G = torch.matmul(A, self.Ac_inv)
        G_weight = torch.einsum('vp,pij->vij', self.get_weights, G)
        self.cache_dict['get_Gweights'] = G_weight
        return G_weight

    @property
    def get_rotation(self):
        '''
        use Nlerp to get the weighted rotation
        https://splines.readthedocs.io/en/latest/rotation/slerp.html
        '''
        if 'get_rotation' in self.cache_dict: return self.cache_dict['get_rotation']
        knn_rotation = self.rotation_activation(self.vert_rotation[self.knn_idx])
        _rotation = torch.sum(knn_rotation * self.knn_weight[...,None], dim=1) 
        _rotation = self.rotation_activation(_rotation)

        Rq_n = matrix_to_quaternion(self.get_Gweights[:,:3,:3])
        Rhq = matrix_to_quaternion(self.Rh)
        _rotation = quaternion_multiply(Rhq, quaternion_multiply(Rq_n, _rotation))
        self.cache_dict['get_rotation'] = _rotation            
        return _rotation

    @property
    def get_cano_xyz(self):
        if 'get_cano_xyz' in self.cache_dict: return self.cache_dict['get_cano_xyz']
        _xyz = self._xyz
        if self.is_dxyz:
            if not self.is_novel_pose:
                smpl_poses = torch.clone(self.smpl_poses).detach()
                smpl_poses[:3] = 0
                _mlp_in = torch.cat([self.embedder_dxyz(self.vert_xyz), torch.tile(smpl_poses, [self.vert_xyz.shape[0],1])], dim=-1)
                _mlp_dvertxyz = self.mlp_dxyz(_mlp_in)
            else:
                if self.vert_dxyz_avg is None: self.calculate_vert_avg_dxyz()
                _mlp_dvertxyz = self.vert_dxyz_avg

            knn_dxyz = _mlp_dvertxyz[self.knn_idx]
            _dxyz = torch.sum(knn_dxyz * self.knn_weight[...,None], dim=1)
            self.cache_dict['mlp_dxyz'] = _dxyz
            _xyz = _xyz + _dxyz
        else:
            self.cache_dict['mlp_dxyz'] = torch.zeros_like(_xyz)

        self.cache_dict['get_cano_xyz'] = _xyz
        return _xyz

    @property
    def get_xyz(self):
        if 'get_xyz' in self.cache_dict: return self.cache_dict['get_xyz']
        _xyz = self.get_cano_xyz
        _xyz = torch.einsum('vij,vj->vi', self.get_Gweights, F.pad(_xyz,(0,1),value=1))[:,:3]
        _xyz = torch.einsum('ij,vj->vi', self.Rh, _xyz) + self.Th
        self.cache_dict['get_xyz'] = _xyz
        return _xyz

    def get_opacity(self, cam_pose=None):
        _opacity = self._opacity
        if cam_pose is not None:
            _front = (torch.sum((cam_pose - self.get_xyz.detach()) * self.get_normal.detach(), dim=-1, keepdim=True) > 0)
            _opacity = torch.where(_front, _opacity, 0)
        return _opacity

    @property
    def get_normal(self):
        if 'get_normal' in self.cache_dict: return self.cache_dict['get_normal']
        knn_normal = self.vert_normal[self.knn_idx]
        _normal = torch.sum(knn_normal * self.knn_weight[...,None], dim=1)
        _normal = F.normalize(_normal)

        _normal = torch.einsum('nij,nj->ni', self.get_Gweights[:,:3,:3], _normal)
        _normal = torch.einsum('ij,nj->ni', self.Rh, _normal)
        self.cache_dict['get_normal'] = _normal
        return _normal

    @property
    def get_albedo(self):
        if 'get_albedo' in self.cache_dict: return self.cache_dict['get_albedo']
        knn_albedo = F.sigmoid(self.vert_albedo)[self.knn_idx]
        _albedo = torch.sum(knn_albedo * self.knn_weight[...,None], dim=1)
        self.cache_dict['get_albedo'] = _albedo
        return _albedo

    @property
    def get_roughness(self):
        if 'get_roughness' in self.cache_dict: return self.cache_dict['get_roughness']
        knn_roughness = F.sigmoid(self.vert_roughness)[self.knn_idx]
        _roughness = torch.sum(knn_roughness * self.knn_weight[...,None], dim=1)
        self.cache_dict['get_roughness'] = _roughness
        return _roughness

    @property
    def get_specularTint(self):
        if 'get_specularTint' in self.cache_dict: return self.cache_dict['get_specularTint']
        knn_specularTint = F.sigmoid(self.vert_specularTint)[self.knn_idx]
        _specularTint = torch.sum(knn_specularTint * self.knn_weight[...,None], dim=1)
        self.cache_dict['get_specularTint'] = _specularTint
        return _specularTint        

    @property
    def get_visiblity(self):
        if 'get_visibility' in self.cache_dict: return self.cache_dict['get_visibility']
        if self.is_train:
            vert_visib = self.vertvb_visib
            assert vert_visib is not None
        else: 
            vert_visib =  self.get_vb_visibility(self.envmap.envmap.reshape(-1,3))
        vb_knn_visib = vert_visib[self.vb_knn_idx]
        _visib = torch.sum(vb_knn_visib * self.vb_knn_weight[...,None], dim=1)
        self.cache_dict['get_visibility'] = _visib
        return _visib

    def get_color(self, cam_pos):
        wo = F.normalize(cam_pos - self.get_xyz)
        nor = self.get_normal
        
        if self.is_visibility:
            visib = self.get_visiblity
        else:
            visib = None
        
        c_d = self.envmap.env_integral_diffuse(wo, nor, self.get_albedo, visibility=visib)
        
        if self.is_specular:
            c_l = self.envmap.env_integral_specular(wo, nor, self.get_roughness, visibility=visib)
            c = c_d + c_l * self.get_specularTint
        else:
            c = c_d

        c = linear_to_srgb(c)
        return c

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, vert, vert_normal, \
                        t_joints, knn_K, all_poses, all_Th, all_Rh, vert_weights, vert_f):
        # init vert_xyz with body world vertices
        self.vert_xyz = torch.tensor(vert).float().cuda()

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist = compute_knn(fused_point_cloud, fused_point_cloud, K=2, r=0.5)[1][:,1]
        dist2 = torch.mean(torch.clamp_min(dist, 0.0000001)) * 0.5
        scales = torch.full((self.vert_xyz.shape[0], 3), torch.log(torch.sqrt(dist2))).float().cuda()
        rots = torch.zeros((self.vert_xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        albedo = torch.full((self.vert_xyz.shape[0], 3), inverse_sigmoid(torch.tensor(0.2)), dtype=torch.float32, device='cuda')
        roughness = torch.full((self.vert_xyz.shape[0], 1), inverse_sigmoid(torch.tensor(0.95)), dtype=torch.float32, device='cuda')
        specularTint = torch.full((self.vert_xyz.shape[0], 1), inverse_sigmoid(torch.tensor(0.95)), dtype=torch.float32, device='cuda')

        self.vert_normal = torch.tensor(vert_normal).float().cuda()
        self.vert_weights = torch.tensor(vert_weights).float().cuda()

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._opacity = torch.full((self._xyz.shape[0], 1), 1, dtype=torch.float32, device='cuda')
        self.vert_scaling = nn.Parameter(scales.requires_grad_(True))
        self.vert_rotation = nn.Parameter(rots.requires_grad_(True))

        self.vert_albedo = nn.Parameter(albedo.requires_grad_(True))
        self.vert_roughness = nn.Parameter(roughness.requires_grad_(True))
        self.vert_specularTint = nn.Parameter(specularTint.requires_grad_(True))

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.knn_K = knn_K
        self.t_joints = torch.tensor(t_joints).float().cuda()

        for key in all_poses: all_poses[key] = nn.Parameter(torch.tensor(all_poses[key], dtype=torch.float32, device='cuda').requires_grad_(False))
        for key in all_Th: all_Th[key] = torch.tensor(all_Th[key], dtype=torch.float32, device='cuda')
        for key in all_Rh: all_Rh[key] = torch.tensor(all_Rh[key], dtype=torch.float32, device='cuda')
        self.all_poses = nn.ParameterDict(all_poses)
        self.all_Rh = all_Rh
        self.all_Th = all_Th

        self.mlp_dxyz._modules['layers'][-1].weight.data.fill_(0)
        self.mlp_dxyz._modules['layers'][-1].bias.data.fill_(0)
        self.mlp_dxyz = self.mlp_dxyz.cuda()
        self.embedder_dxyz = self.embedder_dxyz.cuda()

        self.vert_f = torch.tensor(vert_f).cuda()

        self.init()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self.vert_scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self.vert_rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

            {'params': [self.vert_albedo], 'lr': training_args.albedo_lr, "name": "albedo"},
            {'params': [self.vert_roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
            {'params': [self.vert_specularTint], 'lr': training_args.specularTint_lr, "name": "specularTint"},
            {'params': [self.envmap._envmap], 'lr': training_args.envmap_lr, "name": "envmap"},
            {'params': self.all_poses.parameters(), 'lr': training_args.all_poses_lr_init, "name": "all_poses"},
            {'params': self.mlp_dxyz.parameters(), 'lr': training_args.mlp_dxyz_lr_init, "name": "mlp_dxyz"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.pose_scheduler_args = get_expon_lr_func(lr_init=training_args.all_poses_lr_init,
                                                    lr_final=training_args.all_poses_lr_final,
                                                    max_steps=training_args.position_lr_max_steps,
                                                    begin_steps=training_args.warm_up_iters)
        self.mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_dxyz_lr_init,
                                                    lr_final=training_args.mlp_dxyz_lr_final,
                                                    max_steps=training_args.position_lr_max_steps,
                                                    begin_steps=1000)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group['name'] == 'all_poses':
                lr = self.pose_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group['name'] == 'mlp_dxyz':
                lr = self.mlp_scheduler_args(iteration)
                param_group['lr'] = lr

##################### body part ########################
        
    def init_body(self):
        R_pose = SMPL.rotvec2mat(torch.tensor(SMPL.big_poses.reshape(-1,3)).cuda())
        Ac = SMPL.transform_matrix(self.t_joints, R_pose)
        self.Ac_inv = torch.linalg.inv(Ac)
        self.reset_pose()

    def reset_pose(self):
        self.Rh = torch.eye(3, dtype=torch.float32, device='cuda')
        self.Th = torch.zeros(3, dtype=torch.float32, device='cuda')
        self.smpl_poses = torch.tensor(SMPL.t_poses, dtype=torch.float32, device='cuda')

    @property
    def smpl_poses(self):
        return self._smpl_poses
    
    @smpl_poses.setter
    def smpl_poses(self, value):
        self.cache_dict = {}
        self.vertvb_visib = None
        self._smpl_poses = value

    def set_freeze_params(self, status):
        for k, v in self.all_poses.items():
            self.all_poses[k].requires_grad_(not status)
        self.mlp_dxyz.requires_grad_(not status)

    def calculate_vert_avg_dxyz(self):
        cnt = 0
        vert_dxyz = torch.zeros_like(self.vert_xyz)
        for key in self.all_poses:
            smpl_poses = torch.clone(self.all_poses[key].detach())
            smpl_poses[:3] = 0
            _mlp_in = torch.cat([self.embedder_dxyz(self.vert_xyz), torch.tile(smpl_poses, [self.vert_xyz.shape[0],1])], dim=-1)
            _mlp_dvertxyz = self.mlp_dxyz(_mlp_in)
            vert_dxyz = vert_dxyz + _mlp_dvertxyz
            cnt += 1
        self.vert_dxyz_avg = vert_dxyz / cnt
        self.vert_dxyz_avg = self.vert_dxyz_avg.detach()

###################### KNN part ########################
    
    def init_knn(self):
        # vert_xyz -> vert_xyz
        idx, _, _ = compute_knn(self.vert_xyz, self.vert_xyz, K=19, r=0.3)
        assert (idx >= 0).all()
        self.neighbor_idx = idx[:,1:]

    def compute_knn(self):
        # gs -> vert_xyz
        idx, dist2, self.knn_grid = compute_knn(self._xyz, self.vert_xyz, self.knn_K, self.knn_grid, r=0.1)
        _idx_mask = idx < 0
        dist2[_idx_mask] = 1e8
        self._opacity[torch.all(_idx_mask, dim=-1, keepdim=True)] = 0

        self.knn_idx = idx
        self.knn_weight =  1 / torch.clamp_min(torch.sqrt(dist2), 1e-8)
        self.knn_weight = self.knn_weight / torch.sum(self.knn_weight, dim=1, keepdim=True)

        self.cache_dict = {}

        # gs -> vertvb_xyz
        self.vb_knn_idx = torch.clone(self.knn_idx[:,:3])
        self.vb_knn_weight = torch.clone(self.knn_weight[:,:3])
        self.vb_knn_weight /= torch.sum(self.vb_knn_weight, dim=1, keepdim=True)

###################### Visibility part ########################    
        
    @property
    def get_vb_weights(self):
        return self.vert_weights

    @property
    def get_vb_Gweights(self):
        R_pose = SMPL.rotvec2mat(self.smpl_poses.reshape(-1,3))
        A = SMPL.transform_matrix(self.t_joints, R_pose)
        G = torch.matmul(A, self.Ac_inv)
        G_weight = torch.einsum('vp,pij->vij', self.get_vb_weights, G)
        return G_weight

    @property
    def get_vb_xyz(self):
        _xyz = self.vert_xyz
        _xyz = torch.einsum('vij,vj->vi', self.get_vb_Gweights, F.pad(_xyz,(0,1),value=1))[:,:3]
        _xyz = torch.einsum('ij,vj->vi', self.Rh, _xyz) + self.Th
        return _xyz
    
    def get_vb_visibility(self, envmap=None):
        with torch.set_grad_enabled(False):
            visib = self.visib.get_visibility(self.get_vb_xyz, envmap)
        return visib

###################### My densification part ########################

    def densify_by_density(self, densify_density_ratio=0.02, density=None):
        self.compute_knn()
        if density is None:
            density = 1e5 / len(self.vert_f)
        
        score = torch.zeros(self.vert_xyz.shape[0], dtype=torch.long, device='cuda')
        knn_idx = self.knn_idx.reshape(-1)
        knn_idx = knn_idx[knn_idx >= 0]
        score.scatter_add_(0, knn_idx.reshape(-1), torch.ones_like(knn_idx.reshape(-1)))
        score = torch.sum(score[self.vert_f], dim=-1) / 6 / self.knn_K
        need_gs_num = torch.sum(torch.clamp_min(density - score, 0)).item()
        need_gs_num = int(need_gs_num * densify_density_ratio)
        idx = torch.multinomial(torch.clamp_min(density - score, 0), need_gs_num)

        w = torch.rand((need_gs_num, 2), device='cuda')
        w = torch.where(torch.sum(w, dim=-1, keepdim=True) > 1, 1-w, w)
        w = torch.cat((w, 1-torch.sum(w, dim=-1, keepdim=True)), dim=-1)
        
        new_xyz = torch.einsum('nfi,nf->ni', self.vert_xyz[self.vert_f[idx]], w)
        new_opacities = torch.ones((need_gs_num, 1), dtype=torch.float32, device='cuda')

        self.densification_postfix(new_xyz, new_opacities)
    
        self.compute_knn()

###################### My densification part ########################

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] != 'xyz': continue  

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = self._opacity[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] != 'xyz': continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_opacities):
        d = {"xyz": new_xyz,}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = torch.cat([self._opacity, new_opacities], dim=0)

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_opacity) 

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]

        self.densification_postfix(new_xyz, new_opacities)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.compute_knn()
        self.densify_and_split(grads, max_grad, extent)
        self.compute_knn()

        prune_mask = (self._opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.compute_knn()

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1