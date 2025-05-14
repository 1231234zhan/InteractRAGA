
import numpy as np
import cv2 as cv
import os
from os import path

import torch
from torch import nn
import torch.nn.functional as F

import nvdiffrast.torch as dr
from pytorch3d.renderer.cameras import look_at_view_transform, FoVOrthographicCameras

def clamp_min(x):
    return torch.clamp_min(x, 1e-8)

class EnvMap:
    def __init__(self, hw=(16,32), device='cuda'):
        H, W = hw
        envmap = torch.full((H,W,3), -3, dtype=torch.float32, device=device)
        self._envmap = nn.Parameter(envmap.requires_grad_(True))
        
        _grid = np.mgrid[0:H,0:W].transpose(1,2,0).astype(np.float32)
        _grid += 0.5
        _grid[...,0] = _grid[...,0] / H * np.pi   # theta
        _grid[...,1] = _grid[...,1] / W * np.pi * 2   # phi
        self.sphe_angle = torch.tensor(_grid, device=device)
        self.sphe_vec = self.sphere_angle_to_vector(self.sphe_angle)

        _theta = np.arange(H+1, dtype=np.float32) / H * np.pi
        _arc_phi = np.sin(_theta) * 2 * np.pi / W
        _area = (_arc_phi[:-1] + _arc_phi[1:]) * np.pi / H / 2
        self.env_area = torch.tile(torch.tensor(_area[...,None], device=device), [1,W])
        self.env_all_area = torch.sum(self.env_area)

        self.env_H, self.env_W = H, W

    @property
    def envmap(self):
        return F.softplus(self._envmap)

    @staticmethod
    def inv_softplus(x):
        x = torch.where(x>20, x, torch.log(torch.clamp_min(torch.exp(x)-1, 1e-4)))
        return x

    @envmap.setter
    def envmap(self, value):
        self._envmap = self.inv_softplus(value)

    @staticmethod
    def sphere_angle_to_vector(angle):
        sin_theta, cos_theta = torch.sin(angle[...,0]), torch.cos(angle[...,0])
        sin_phi, cos_phi = torch.sin(angle[...,1]), torch.cos(angle[...,1])
        x, y, z = sin_theta*cos_phi, sin_theta*sin_phi, cos_theta
        return torch.stack((x,y,z), dim=-1)

    @staticmethod
    def sphere_vector_to_angle(vec):
        theta = torch.acos(vec[...,2])
        phi = torch.atan2(vec[...,1], vec[...,0]) % (2 * torch.pi)
        return torch.stack((theta, phi), dim=-1)

    def get_ray_color(self, wi):
        angle = self.sphere_vector_to_angle(wi)
        _idx = (angle / torch.tensor([torch.pi, 2*torch.pi], device=angle.device) * 2 -1)[None,None,:,[1,0]]
        _array = self.envmap.permute(2,0,1)[None]
        _result = F.grid_sample(_array, _idx, mode='nearest', padding_mode='border', align_corners=False)
        color = _result[0].permute(1,2,0)
        return color

    def env_integral_diffuse(self, wo, nor, albedo, visibility=None):
        wi = self.sphe_vec
        # wi: HW3  wo: M3  nor: M3
        H, W, M, C = self.env_H, self.env_W, wo.shape[0], 3
        front_mask = torch.sum(wo*nor, dim=-1) > 0 # M
        wo, nor, albedo = wo[front_mask], nor[front_mask], albedo[front_mask]

        # wi: HW3  wo: N3  nor: N3  albedo: N3
        f_value = albedo / torch.pi  # N3
        cos_i = torch.clamp_min(torch.einsum('hwc,nc->nhw', wi, nor), 0)
        if visibility is not None:
            visibility = visibility[front_mask]
            sum_L_f_cos_w = torch.einsum('nc,nhw,nhw,hwc,hw->nc', f_value, visibility.reshape(-1, H, W), cos_i, self.envmap, self.env_area)
        else:
            sum_L_f_cos_w = torch.einsum('nc,nhw,hwc,hw->nc', f_value, cos_i, self.envmap, self.env_area)

        idx = torch.argwhere(front_mask).tile(1,C)
        color = torch.zeros((M, C), dtype=wo.dtype, device=wo.device)
        color.scatter_(0, idx, sum_L_f_cos_w)
        return color

    def env_integral_specular(self, wo, nor, roughness, visibility=None):
        wi = self.sphe_vec
        # wi: HW3  wo: M3  nor: M3
        H, W, M, C = self.env_H, self.env_W, wo.shape[0], 3
        front_mask = torch.sum(wo*nor, dim=-1) > 0 # M
        wo, nor, roughness = wo[front_mask], nor[front_mask], roughness[front_mask]

        # wi: HW3  wo: N3  nor: N3  roughness: N1
        f_value = BRDF.cook_torrance(wi[:,:,None],wo,nor,roughness)  # HW13,N3,N3,N1 -> HWN1
        cos_i = torch.clamp_min(torch.einsum('hwc,nc->nhw', wi, nor), 0)
        if visibility is not None:
            visibility = visibility[front_mask]
            sum_L_f_cos_w = torch.einsum('hwnc,nhw,nhw,hwc,hw->nc', f_value, visibility.reshape(-1, H, W), cos_i, self.envmap, self.env_area)
        else: 
            sum_L_f_cos_w = torch.einsum('hwnc,nhw,hwc,hw->nc', f_value, cos_i, self.envmap, self.env_area)

        idx = torch.argwhere(front_mask).tile(1,C)
        color = torch.zeros((M, C), dtype=wo.dtype, device=wo.device)
        color.scatter_(0, idx, sum_L_f_cos_w)
        return color

class BRDF:
    @staticmethod
    def D_GGX(a, n, h):
        a2 = a*a
        return a2 / clamp_min( torch.pi * ( (n*h).sum(-1,True)**2 * (a2-1) + 1 )**2 )

    @staticmethod
    def G_Schlick(k, n, v, l):
        def G1(x):
            nx = (n*x).sum(-1,True)
            return nx / clamp_min( nx * (1-k) + k )
        return G1(l) * G1(v)

    @staticmethod
    def F_Fresnel(v, h, F0=0.04):
        _dot = (v*h).sum(-1,True)
        return F0 + (1-F0) * torch.pow(2, ((-5.55473*_dot-6.98316)*_dot))
        # return F0 + (1-F0) * (1 - torch.dot(v,h))**5

    @staticmethod
    def cook_torrance(l, v, n, roughness):
        a = roughness**2
        k = (roughness+1)**2 / 8
        h = nn.functional.normalize(l+v, dim=-1)

        D = BRDF.D_GGX(a, n, h)
        G = BRDF.G_Schlick(k, n, v, l)
        F = BRDF.F_Fresnel(v, h)

        f_value = D * F * G / clamp_min(4 * (n*l).sum(-1,True) * (n*v).sum(-1,True) )
        return f_value
    
    @staticmethod
    def diffuse(albedo):
        return albedo / torch.pi

FIX_RESOLUTION = False

def get_graph(V_num, f, max_edge_num=8):
    vert_G = -np.ones((V_num, 12), dtype=int)
    vert_enum = np.zeros(V_num, dtype=int)

    for v in f:
        for i in range(3):
            vert_G[v[i],vert_enum[v[i]]] = v[(i+1)%3]
            vert_enum[v[i]] += 1

    vert_enum = np.clip(vert_enum, 0, max_edge_num)
    vert_G = vert_G[:,:max_edge_num]
    return vert_enum, vert_G

def compute_edge_neighbor(V_num, vert_enum, vert_G, nbr_num=30):
    vert_Gn = -np.ones((V_num, nbr_num), dtype=int)
    visit = np.zeros(V_num, dtype=bool)
    def get_nbr(p_init):
        nbr_list = -np.ones(nbr_num + 10, dtype=int)
        nbr_list[0] = p_init
        visit[p_init] = True
        front, rear = 0, 1
        while rear < nbr_num:
            p = nbr_list[front]
            front += 1
            assert p >= 0
            for i in range(vert_enum[p]):
                q = vert_G[p,i]
                if visit[q]: continue
                visit[q] = True
                nbr_list[rear] = q
                rear += 1
        visit[nbr_list[:rear]] = False
        return nbr_list[:nbr_num]

    for i in range(V_num):
        vert_Gn[i] = get_nbr(i)
    return vert_Gn

class Visib():
    f=None
    cam_R=None
    cam_T=None
    N_cam=None
    vert_enum=None
    vert_Gn=None
    def __init__(self, resolution=512, device='cuda'):
        self.glctx = dr.RasterizeGLContext(output_db=False)
        self.resolution = resolution  

        self.f = None
        self.cam_R, self.cam_T = None, None
        self.N_cam = None

    def update(self, views, f):
        R, T = look_at_view_transform(eye=views*5, up=((0,0,1),), device=f.device)
        self.cam_R, self.cam_T = torch.permute(R, dims=(0,2,1)), T  # N33, N3. p3d is column-major order and left is +x
        self.N_cam = len(views)

        f = f.type(torch.int64)
        self.f = f

        V_num = torch.max(f).item() + 1
        vert_enum, vert_G = get_graph(V_num, f.cpu().numpy())
        vert_Gn = compute_edge_neighbor(V_num, vert_enum, vert_G)

        self.vert_enum = torch.tensor(vert_enum).cuda()
        self.vert_Gn = torch.tensor(vert_Gn).cuda()

    @staticmethod
    def get_bbox(xyz):
        _min, _ = torch.min(xyz, dim=0)
        _max, _ = torch.max(xyz, dim=0)
        return torch.stack([_min, _max], dim=1)

    def get_visibility(self, xyz, envmap=None):
        assert self.f is not None
        V_num = len(xyz)
        bbox = self.get_bbox(xyz)
        length = torch.norm(bbox[:,1] - bbox[:,0])
        center = torch.mean(bbox, dim=1)
        xyz = xyz - center

        cam_R, cam_T, N_cam = self.cam_R, self.cam_T, self.N_cam
        if envmap is not None:
            mask = torch.sum(envmap, dim=-1) > -1
            cam_R, cam_T, N_cam = cam_R[mask], cam_T[mask], torch.sum(mask).item()

        # world to ndc
        xyz = torch.einsum('nij,vj->nvi', cam_R, xyz) + cam_T[:,None]
        if not FIX_RESOLUTION:
            max_x, min_x = torch.max(xyz[:,:,0]).item(), torch.min(xyz[:,:,0]).item()
            max_y, min_y = torch.max(xyz[:,:,1]).item(), torch.min(xyz[:,:,1]).item()
            K = FoVOrthographicCameras(device='cuda').compute_projection_matrix(
                znear=0.1, zfar=10, max_y=max_y, min_y=min_y, max_x=max_x, min_x=min_x, scale_xyz=torch.ones((1,3)).cuda(),
            )[0]
        else:
            K = FoVOrthographicCameras(device='cuda').compute_projection_matrix(
                znear=0.1, zfar=10, max_y=length/2, min_y=-length/2, max_x=length/2, min_x=-length/2, scale_xyz=torch.ones((1,3)).cuda(),
            )[0]

        xyz = torch.einsum('ij,nvj->nvi', K, F.pad(xyz, [0,1], value=1)).contiguous()

        # rasterize to compute the visibility of vertices
        if not FIX_RESOLUTION:
            resolution = ( np.array([max_y - min_y, max_x - min_x]) * 240 ).astype(int).tolist()
        else:
            resolution = [self.resolution, self.resolution]
        rast, _ = dr.rasterize(self.glctx, xyz, self.f.type(torch.int32), resolution=resolution)
        rast = rast[...,3].type(torch.int64)

        rast = F.pad(self.f, [0,0,1,0], value=V_num)[rast]   # triangle index, offset by one. no triangle: zero
        rast = rast.reshape(N_cam, -1)
        vis_map = torch.zeros((N_cam, V_num+1), dtype=bool, device=xyz.device)
        vis_map.scatter_(1, rast, True)
        vis_map = vis_map[:,:-1]

        # if False:
        #     fvis_map = torch.zeros((self.N, len(self.f)+1), dtype=bool, device=xyz.device)
        #     fvis_map.scatter_(1, rast.reshape(self.N, -1), True)
        #     fvis_map = torch.where(fvis_map, torch.arange(len(self.f)+1, device=xyz.device), 0)
        #     rast = F.pad(self.f, [0,0,1,0], value=V)[fvis_map]
        #     vis_map = torch.zeros((self.N, V+1), dtype=torch.float16, device=xyz.device)
        #     rast = rast.reshape(self.N, -1)
        #     vis_map.scatter_add_(1, rast, torch.ones_like(rast, dtype=torch.float16))
        #     vis_map = vis_map[:,:-1].type(torch.int64)
        #     vis_map = vis_map * 2 >= self.vert_fnum

        # Smooth the visibility boundary and blur
        vis_map = vis_map.type(torch.float32)
        vis_map = torch.median(vis_map[:,self.vert_Gn], dim=-1)[0]
        vis_map = torch.median(vis_map[:,self.vert_Gn], dim=-1)[0]
        vis_map = torch.mean(vis_map[:,self.vert_Gn], dim=-1)

        if envmap is not None:
            idx = torch.argwhere(mask).tile(1,V_num)
            all_vis_map = torch.ones((self.N_cam, V_num), dtype=torch.float32, device=xyz.device)
            all_vis_map.scatter_(0, idx, vis_map)
        else:
            all_vis_map = vis_map

        all_vis_map = all_vis_map.permute(1,0)

        return all_vis_map


if __name__ == '__main__':
    envmap = EnvMap()
    print(envmap.sphe_vec[7,24])