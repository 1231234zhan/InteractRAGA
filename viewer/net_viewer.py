IMAGE_ENCODE = 'gpu'

import os
from os import path
import numpy as np
import time
import cv2 as cv
import imageio.v3 as iio
import pickle
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation
import math
import json
import copy

from net import net_init, recv, send, is_connected, close
from nvjpeg import NvJpeg

os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = np.tan((fovY / 2))
    tanHalfFovX = np.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P.astype(np.float32)

def exr_to_envmap(image):
    H, W = image.shape[:2]
    He, We = 16, 32
    envmap = np.zeros((He, We, 3), dtype=np.float32)

    for y in range(He):
        for x in range(We):
            yt, yb = math.floor(H/He*y), math.floor(H/He*(y+1))
            xl, xr = math.floor(W/We*x), math.floor(W/We*(x+1))
            envmap[y,x] = np.mean(image[yt:yb,xl:xr], axis=(0,1))
            
    envmap[:,0:16] = np.flip(envmap[:,0:16], axis=1) 
    envmap[:,16:32] = np.flip(envmap[:,16:32], axis=1)

    return envmap 

def load_pose(data_path):
    param = np.load(data_path, allow_pickle=True).item()
    param['poses'], param['Th'], param['Rh'] = np.squeeze(param['poses']), np.squeeze(param['Th']), np.squeeze(param['Rh'])
    param['Rh'] = Rotation.from_rotvec(param['Rh']).as_matrix()
    return param

def load_poses(data_dir):
    namelist = [s for s in os.listdir(data_dir) if path.splitext(s)[1] == '.npy']
    namelist = sorted(namelist, key=lambda s: (len(s), s))
    poses_list = []
    for name in namelist:
        poses_list.append(load_pose(path.join(data_dir, name)))

    return poses_list

def load_aist_poses(data_path):
    with open(data_path, 'rb') as file:
        pose_dict = pickle.load(file)

    poses_list = []
    N = pose_dict['smpl_poses'].shape[0]

    pose_dict['smpl_poses'][:,:3] = (Rotation.from_euler('x', np.pi/2) * Rotation.from_rotvec(pose_dict['smpl_poses'][:,:3])).as_rotvec()
    pose_dict['smpl_trans'] = (Rotation.from_euler('x', np.pi/2).as_matrix() @ pose_dict['smpl_trans'].T).T / pose_dict['smpl_scaling']
    pose_dict['smpl_trans'][:,2] = pose_dict['smpl_trans'][:,2] - np.mean(pose_dict['smpl_trans'][:,2]) + 1

    for k in range(N):
        param = {}
        param['poses'] = pose_dict['smpl_poses'][k].astype(np.float32)        
        param['Th'] = pose_dict['smpl_trans'][k].astype(np.float32)
        param['Rh'] = np.eye(3, dtype=np.float32)

        param = copy.deepcopy(param)
        poses_list.append(param)

    return poses_list

class OrbitCamera:
    def __init__(self, img_wh, center, r, fovx=np.pi/9*2, rot=None, z_near=0.01, z_far=100):
        self.W, self.H = img_wh
        self.radius = r
        self.center = center.astype(np.float32)
        self.fovx = fovx
        if rot is None:
            self.rot = np.eye(3)
        else:
            self.rot = rot

        self.focal = self.W / 2 / np.tan(self.fovx/2)
        self.fovy = 2 * np.arctan(self.H / 2 / self.focal)
        self.projection_matrix = getProjectionMatrix(z_near, z_far, fovx, self.fovy)

        self.z_near = z_near
        self.z_far = z_far

        self.old_rot = self.rot
        self.old_center = self.center

    def update_cam(self, W=None, H=None, fovx=None):
        if W is not None: self.W = W
        if H is not None: self.H = H
        if fovx is not None: self.fovx = fovx

        self.focal = self.W / 2 / np.tan(self.fovx/2)
        self.fovy = 2 * np.arctan(self.H / 2 / self.focal)
        self.projection_matrix = getProjectionMatrix(self.z_near, self.z_far, fovx, self.fovy)

    @property
    def extrinsic(self):
        # camera space: right-x down-y
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        # translate
        res[:3, 3] += self.center
        # res = qMat#

        # print(res[:3,3])
        res = np.linalg.inv(res)
        return res.astype(np.float32)
    
    @property
    def intrinsic(self):
        K = np.zeros((3, 3))
        K[0, 0] = self.focal
        K[1, 1] = self.focal
        K[2, 2] = 1
        K[0, 2], K[1, 2] = self.W/2, self.H/2
        return K.astype(np.float32)

    @property
    def view_matrix(self):
        return self.extrinsic

    @property
    def view_projection_matrix(self):
        return self.projection_matrix @ self.extrinsic

    def gaussian_cam_info(self):
        data = dict(
            resolution_x=self.W,
            resolution_y=self.H,
            fov_y=self.fovy,
            fov_x=self.fovx,
            z_near=self.z_near,
            z_far=self.z_far,
            view_matrix=self.view_matrix.T, # glm form
            view_projection_matrix=self.view_projection_matrix.T, # glm form
            projection_matrix=self.projection_matrix.T,
        )

        return data

    def update_orbit(self):
        self.old_rot = self.rot

    def orbit(self, dx, dy):
        rotvec_x = self.old_rot[:, 1] * np.radians(0.2 * dx)
        rotvec_y = self.old_rot[:, 0] * np.radians(-0.2 * dy)
        self.rot = Rotation.from_rotvec(rotvec_y).as_matrix() @ \
            Rotation.from_rotvec(rotvec_x).as_matrix() @ \
            self.old_rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def update_pan(self):
        self.old_center = self.center

    def pan(self, dx, dy, dz=0):
        self.center = self.old_center - 2e-3 * self.rot @ np.array([dx, dy, dz])

class GUI:
    def __init__(self, hw):
        H, W = hw
        self.H = H
        self.W = W
        self.cam = OrbitCamera(
            img_wh=(W, H),
            center=np.array([0,0,0]),
            r = 1.5,
        )
        self.image = None
        self.nj = NvJpeg()

        self.timer = 0

        self.scaling_modifier = 1.0
        self.camera_list = []
        self.pose_list = []
        self.render_type = 'image'
        self.env_rot = 0
        self.env_brighness = 1

        self.load_envmap = None
        self.attach_envmap = None
        self.load_envmap_path = ''

        self.poses = None
        self.Th = None
        self.Rh = None

        self.specular = False
        self.visibility = False
        self.is_novel_pose = False
        self.dxyz = True

        self.background = np.zeros(3, dtype=np.float32)

        self.novel_pose_list = []

        self.edit_status = None
        self.edit_texture = None
        self.edit_scale = 0.5
        self.image = np.zeros((H,W,3), dtype=np.float32)
        self.ori_image = None

        self.edit_resize_texture = None
        self.edit_texture_position = None

        self.output_render_type = None

    def decode_bytes(self, image_byte, hw):
        H, W = hw
        if IMAGE_ENCODE == 'cpu':
            image_byte = np.frombuffer(image_byte, dtype=np.uint8)
            print(len(image_byte))
            image = cv.imdecode(image_byte, cv.IMREAD_COLOR)
        elif IMAGE_ENCODE == 'gpu':
            image = self.nj.decode(image_byte)
        else:
            image = np.frombuffer(image_byte, dtype=np.uint8).reshape((H, W, 3))       
        image = image.astype(np.float32) / 255
        return image

    def gaussian_gui_info(self):
        info = self.cam.gaussian_cam_info()
        def set_val(s, v):
            if v is not None: info[s] = v

        set_val('scaling_modifier', self.scaling_modifier)
        set_val('render_type', self.render_type)
        set_val('env_rot', self.env_rot)
        set_val('env_brightness', self.env_brighness)
        set_val('load_envmap', self.load_envmap)
        set_val('poses', self.poses)
        set_val('Th', self.Th)
        set_val('Rh', self.Rh)
        set_val('specular', self.specular)
        set_val('visibility', self.visibility)
        set_val('is_novel_pose', self.is_novel_pose)
        set_val('dxyz', self.dxyz)
        set_val('background', self.background)
        
        return info

    def load_camera_list(self, cameras):
        self.camera_list = cameras
        dpg.configure_item('camera_id', max_value=len(self.camera_list))

    def load_pose_list(self, poses):
        self.pose_list = poses
        dpg.configure_item('frame_id', max_value=len(self.pose_list)-1)

    def edit_image(self):
        tex_H, tex_W = (np.array(self.edit_texture.shape[:2]) * self.edit_scale).astype(int)
        texture = cv.resize(self.edit_texture, None, fx=self.edit_scale, fy=self.edit_scale)
        mouse_pos = dpg.get_mouse_pos(local=False)
        image_pos = dpg.get_item_rect_min('image')
        tex_pos = np.array(mouse_pos, dtype=int) - np.array(image_pos, dtype=int) - np.array([tex_W//2, tex_H//2])
        
        self.edit_resize_texture = texture
        self.edit_texture_position = tex_pos

        bbox = np.array([max(tex_pos[0], 0), max(tex_pos[1], 0), min(tex_pos[0]+tex_W, self.W), min(tex_pos[1]+tex_H, self.H)])
        
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]: 
            return np.copy(self.ori_image)
        bbox_t = bbox - np.array([tex_pos[0], tex_pos[1], tex_pos[0], tex_pos[1]])

        texture_crop = texture[bbox_t[1]:bbox_t[3],bbox_t[0]:bbox_t[2]]
        image = np.copy(self.ori_image)
        image_crop = image[bbox[1]:bbox[3],bbox[0]:bbox[2]]

        image[bbox[1]:bbox[3],bbox[0]:bbox[2]] = ( texture_crop[...,:3] - image_crop) * texture_crop[...,3:] + image_crop

        return image

    def attach_envmap_to_image(self):
        He, We = self.attach_envmap.shape[:2]
        envmap = np.clip(self.attach_envmap * self.env_brighness, 0, 1)
        env_rot = np.floor(self.env_rot * 32).astype(int) * 6
        envmap = np.roll(envmap, env_rot, axis=1)

        self.image[:He,-We:] = envmap

    def render_seq_update_status(self): 
        if self.output_render_type == 'frame_seq':
            frame = dpg.get_value('frame_id')
            rot = math.floor(self.env_rot * 32)
            if frame >= len(self.pose_list):
                return -1, -1
            dpg.set_value('frame_id', frame + 1)
            pose_info = self.pose_list[frame]
            self.poses, self.Rh, self.Th = pose_info['poses'], pose_info['Rh'], pose_info['Th']
            return frame, rot
        if self.output_render_type == 'novel_pose_frame_seq':
            frame = dpg.get_value('novel_pose_frame_id')
            rot = math.floor(self.env_rot * 32)
            if frame >= len(self.novel_pose_list):
                return -1, -1
            dpg.set_value('novel_pose_frame_id', frame + 1)
            pose_info = self.novel_pose_list[frame]
            self.poses, self.Rh, self.Th = pose_info['poses'], pose_info['Rh'], pose_info['Th']
            return frame, rot
        if self.output_render_type == 'env_rot_seq':
            frame = dpg.get_value('frame_id')
            rot = math.floor(self.env_rot * 32) 
            if rot >= 32:
                return -1, -1
            self.env_rot += 1/32
            return frame, rot
        raise RuntimeError
    
    def loop_render_seq(self):
        time.sleep(0.5)  # ensure all the data are received
        recv()
        frame, rot = self.render_seq_update_status()
        if frame < 0:
            self.edit_status = None
            return

        data = self.gaussian_gui_info()
        H, W = data['resolution_y'], data['resolution_x']
        if is_connected():
            message = pickle.dumps(data) 
            send(message)
        else:
            print('Connection closed!!')
            self.edit_status == None
            return

        data = None
        while data is None: 
            data = recv()
            time.sleep(0.001)
        try:
            data = pickle.loads(data)
        except pickle.UnpicklingError:
            print('UnpicklingError!!')
            self.edit_status == None
            return

        image_byte = data['image_bytes']
        self.image = self.decode_bytes(image_byte, (H, W))
        dpg.set_value('texture', self.image)

        image_out_dir = dpg.get_value('image_out_dir')
        out_path = path.join(image_out_dir, f'{frame:04d}_{rot:02d}.png')
        iio.imwrite(out_path, (self.image * 255).astype(np.uint8))

    def render_loop(self):

        frame_cnt = 0
        total_byte = 0
        acc_time = 0 

        while dpg.is_dearpygui_running():
            if self.edit_status is None:
                data = self.gaussian_gui_info()
                H, W = data['resolution_y'], data['resolution_x']

                if is_connected():
                    message = pickle.dumps(data) 
                    send(message)
                else:
                    # time.sleep(0.5)
                    pass
                
                data = recv()
                if data:
                    total_byte += len(data)
                    data = pickle.loads(data)
                    if 'cameras' in data:
                        self.load_camera_list(data['cameras'])
                    if 'poses' in data:
                        self.load_pose_list(data['poses'])
                    if 'gaussian_num' in data:
                        dpg.set_value('gaussian_num', 'Gaussian number: ' + str(data['gaussian_num']))

                    image_byte = data['image_bytes']
                    self.image = self.decode_bytes(image_byte, (H, W))
                    if self.load_envmap is not None:
                        self.attach_envmap_to_image()

                    dpg.set_value('texture', self.image)

                acc_time += dpg.get_delta_time()
                if data: frame_cnt += 1
                if acc_time > 1: 
                    fps = frame_cnt / acc_time
                    dpg.set_value('fps', f'FPS: {fps:.1f} ')
                    dpg.set_value('mbps', f'{total_byte / 10e6} MB/s')
                    frame_cnt = 0
                    acc_time = 0
                    total_byte = 0

                dpg.render_dearpygui_frame()
            elif self.edit_status in ['hover', 'place']:
                dpg.set_value('texture', self.image)
                dpg.render_dearpygui_frame()
            elif self.edit_status == 'optimize':
                data = recv()
                if data:
                    data = pickle.loads(data)
                    if 'is_optimize_edit_finish' in data:
                        self.edit_status = None
                dpg.render_dearpygui_frame()
            elif self.edit_status == 'render_seq':
                raise NotImplementedError
                # self.loop_render_seq()
                # dpg.render_dearpygui_frame()
            
            self.timer += dpg.get_delta_time()
            dpg.set_value('timer', f'Timer: {self.timer:.2f} s')

        close()
        dpg.destroy_context()       

    def register_dpg(self):
        cam = self.cam
        H, W = self.H, self.W
        W_info = 500

        dpg.create_context()
        dpg.create_viewport(title="Net Viewer", width=W+20+W_info, height=H+20)
        dpg.set_viewport_pos([400, 400])

        # register mouse callback for cameras
        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("primary_window"): return
            if self.edit_status is not None: return
            cam.orbit(app_data[1], app_data[2])

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("primary_window"): return
            if self.edit_status is not None: return
            cam.scale(app_data)

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("primary_window"): return
            if self.edit_status is not None: return
            cam.pan(app_data[1], app_data[2])
        
        def callback_camera_release_rotate(sender, app_data):
            if not dpg.is_item_focused("primary_window"): return   
            if self.edit_status is not None: return
            cam.update_orbit()     

        def callback_camera_release_pan(sender, app_data):
            if not dpg.is_item_focused("primary_window"): return   
            if self.edit_status is not None: return
            cam.update_pan()   

        # register mouse callback for images
        def callback_image_hover(sender, app_data):
            if self.edit_status != 'hover': return
            self.image = self.edit_image()

        def callback_image_wheel_scale(sender, app_data):
            if self.edit_status != 'hover': return
            _scale = self.edit_scale + 0.04 * app_data
            if _scale < 1 and _scale > 0.1:
                self.edit_scale = _scale
                callback_image_hover(None, None)

        def callback_image_clicked(sender, app_data):
            if self.edit_status != 'hover': return
            callback_image_hover(None, None)
            self.edit_status = 'place'

        def callback_wheel_scale(sender, app_data):
            if self.edit_status is None:
                callback_camera_wheel_scale(sender, app_data)
            else:
                callback_image_wheel_scale(sender, app_data)

        # register info
        def callback_reconnect(sender, app_data):
            close()
            ip = dpg.get_value('ip')
            port = dpg.get_value('port')
            net_init(ip, port)

        def callback_reset_timer(sender, app_data):
            self.timer = 0

        def callback_update_camera(sender, app_data):
            fovx = dpg.get_value('fovx')
            cam.update_cam(fovx=fovx / 180 * np.pi)

        def callback_scale_slider(sender, app_data):
            self.scaling_modifier = app_data

        def callback_camera_id(sender, app_data):
            if self.edit_status is not None: return
            cam_info = self.camera_list[app_data]
            self.cam.rot = cam_info['rotation']
            self.cam.old_rot = self.cam.rot
            self.cam.radius = np.linalg.norm(cam_info['position'] - np.array([0,0,1]))
            self.cam.center = cam_info['position'] - self.cam.rot @ np.array([0,0,-self.cam.radius], dtype=np.float32)
            self.cam.old_center = self.cam.center

        def callback_render_type(sender, app_data):
            self.render_type = app_data

        def callback_env_rot(sender, app_data):
            self.env_rot = app_data

        def callback_env_brightness(sender, app_data):
            self.env_brighness = app_data

        def callback_load_envmap(sender, app_data):
            if app_data == '': self.load_envmap = None
            if not path.exists(app_data): return
            im = cv.imread(app_data, cv.IMREAD_UNCHANGED)
            if im is None: return
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            self.load_envmap = exr_to_envmap(im)
            self.attach_envmap = np.clip(cv.resize(im, [192, 96]), 0, 1)

        def callback_frame_id(sender, app_data):
            if self.edit_status is not None: return
            pose_info = self.pose_list[app_data]
            self.poses = pose_info['poses']
            self.Rh = pose_info['Rh']
            self.Th = pose_info['Th']

        def callback_tpose(sender, app_data):
            if self.edit_status is not None: return
            self.poses = np.zeros(72, dtype=np.float32)
            self.Rh = Rotation.from_euler('x', 90, degrees=True).as_matrix()
            self.Th = np.array([0, 0, 1.1], dtype=np.float32)

        def callback_bigpose(sender, app_data):
            if self.edit_status is not None: return
            big_poses = np.zeros(72, dtype=np.float32)
            big_poses[5] = np.deg2rad(30)
            big_poses[8] = np.deg2rad(-30)   
            self.poses = big_poses
            self.Rh = Rotation.from_euler('x', 90, degrees=True).as_matrix()
            self.Th = np.array([0, 0, 1.1], dtype=np.float32)

        def callback_specular(sender, app_data):
            self.specular = app_data

        def callback_visibility(sender, app_data):
            self.visibility = app_data

        def callback_is_novel_pose(sender, app_data):
            self.is_novel_pose = app_data

        def callback_dxyz(sender, app_data):
            self.dxyz = app_data

        def callback_background(sender, app_data):
            self.background = np.array(app_data[:3], dtype=np.float32)

        def callback_load_novel_pose(sender, app_data):
            app_data = dpg.get_value('novel_pose_text')
            if not path.exists(app_data): return
            if path.isdir(app_data):    # assume ZJUMoCap pose dirrectory
                data_dir = app_data
                poses = load_poses(data_dir)
                self.novel_pose_list = poses
                dpg.configure_item('novel_pose_frame_id', max_value=len(self.novel_pose_list)-1)
            elif path.splitext(app_data)[1] == '.npy':  # assume ZJUMoCap pose npy file
                data_path = app_data
                pose = load_pose(data_path)
                self.novel_pose_list = [pose, pose]
                dpg.configure_item('novel_pose_frame_id', max_value=len(self.novel_pose_list)-1)
            elif path.splitext(app_data)[1] == '.pkl':  # assume AIST motion pkl file
                data_path = app_data
                poses = load_aist_poses(data_path)
                self.novel_pose_list = poses
                dpg.configure_item('novel_pose_frame_id', max_value=len(self.novel_pose_list)-1)

        def callback_novel_frame_id(sender, app_data):
            if self.edit_status is not None: return
            pose_info = self.novel_pose_list[app_data]
            self.poses = pose_info['poses']
            self.Rh = pose_info['Rh']
            self.Th = pose_info['Th']

        # def callback_load_camera(sender, app_data):
        #     if self.edit_status is not None: return
        #     camera_path = dpg.get_value('camera_path')
        #     with open(camera_path, "r") as file:
        #         data = json.load(file)

        #     self.cam.rot = np.array(data['rot']).astype(np.float32)
        #     self.cam.old_rot = self.cam.rot
        #     self.cam.radius = data['r']
        #     self.cam.center = np.array(data['center']).astype(np.float32)
        #     self.cam.old_center = self.cam.center

        # def callback_save_camera(sender, app_data):
        #     camera_path = dpg.get_value('camera_path')
        #     data = {}
        #     data['img_wh'] = [cam.W, cam.H]
        #     data['center'] = cam.center.tolist()
        #     data['r'] = cam.radius
        #     data['fovx'] = cam.fovx
        #     data['rot'] = [x.tolist() for x in cam.rot]

        #     with open(camera_path, "w") as file:
        #         json.dump(data, file)

        def callback_load_texture(sender, app_data):
            image_path = dpg.get_value('texture_path')
            image = iio.imread(image_path, mode='RGBA')
            assert image.shape[-1] == 4
            H, W = image.shape[:2]
            H, W = int(400/W*H) ,400
            image = cv.resize(image, [W, H]).astype(np.float32) / 255

            if self.edit_status is None:
                self.ori_image = np.copy(self.image)
            self.edit_status = 'hover'
            self.edit_texture = image
            self.image = np.copy(self.ori_image)

        def callback_edit_resume(sender, app_data):
            self.edit_status = None

        def callback_optimize_edit(sender, app_data):
            if self.edit_status != 'place': return
            self.edit_status = 'optimize'

            # in this status, sending the data here is thread-safe
            data = self.gaussian_gui_info()
            data['edit_texture'] = self.edit_resize_texture
            data['edit_texture_position'] = self.edit_texture_position
            if is_connected():
                message = pickle.dumps(data) 
                send(message)
            else:
                print('Network Error!')
                self.edit_status = None

        # def callback_key(sender, app_data):
        #     if app_data == dpg.mvKey_Right:
        #         frame = dpg.get_value('novel_pose_frame_id')
        #         dpg.set_value('novel_pose_frame_id', frame + 2)
        #         callback_novel_frame_id(None, frame)
        #     if app_data == dpg.mvKey_Left:
        #         frame = dpg.get_value('novel_pose_frame_id')
        #         dpg.set_value('novel_pose_frame_id', frame - 2)
        #         callback_novel_frame_id(None, frame)
        #     if app_data == dpg.mvKey_D:
        #         frame = dpg.get_value('frame_id')
        #         dpg.set_value('frame_id', frame + 2)
        #         callback_frame_id(None, frame)
        #     if app_data == dpg.mvKey_A:
        #         frame = dpg.get_value('frame_id')
        #         dpg.set_value('frame_id', frame - 2)
        #         callback_frame_id(None, frame)

        # def callback_render_seq(sender, app_data):
        #     rtype = dpg.get_value('image_out_type')
        #     if rtype == 'image':
        #         raise NotImplementedError
            
        #     if rtype == 'frame_seq':
        #         dpg.set_value('frame_id', 0)
        #         callback_frame_id(None, 0)
        #         self.output_render_type = rtype
        #     if rtype == 'novel_pose_frame_seq':
        #         dpg.set_value('novel_pose_frame_id', 0)
        #         callback_novel_frame_id(None, 0)
        #         self.output_render_type = rtype
        #     if rtype == 'env_rot_seq':
        #         # dpg.set_value('frame_id', 0)
        #         callback_env_rot(None, 1e-3)
        #         self.output_render_type = rtype
        #     self.edit_status = 'render_seq'

        def my_separator():
            dpg.add_text('')
            dpg.add_separator()
            dpg.add_text('')

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                width=W,
                height=H,
                format=dpg.mvFormat_Float_rgb,
                default_value=np.zeros((H,W,3), dtype=np.float32),
                tag="texture")

        with dpg.window(tag='primary_window'):
            dpg.add_image('texture', tag='image')
            dpg.set_primary_window('primary_window', True)

        with dpg.window(label='Info', width=W_info, pos=[W+10, 0]):

            with dpg.group(horizontal=True, xoffset=250):
                dpg.add_text('', tag='fps')
                dpg.add_text('- MB/s', tag='mbps')

            with dpg.group(horizontal=True, xoffset=250):
                dpg.add_text('Timer: -', tag='timer')
                dpg.add_button(label='Reset', callback=callback_reset_timer)

            my_separator()

            with dpg.group(horizontal=True, xoffset=250):
                dpg.add_input_text(label='ip', tag='ip', default_value='127.0.0.1', width=150)
                dpg.add_input_int(label='port', tag='port', default_value=23456, width=150)
            dpg.add_button(label='Reconnect', callback=callback_reconnect)

            with dpg.group(horizontal=True, xoffset=250):
                dpg.add_text(f'Width: {W}')
                dpg.add_text(f'Height: {H}')

            dpg.add_input_float(label='Fovx', tag='fovx', default_value=40, min_value=10, max_value=170, width=150)
            dpg.add_button(label='Update camera', callback=callback_update_camera)

            my_separator()

            dpg.add_text('Gaussian number: -', tag='gaussian_num')
            dpg.add_slider_float(default_value=1, min_value=0, max_value=1, label='Scaling Modifier', tag='scale_modifier', callback=callback_scale_slider)
            dpg.add_input_int(default_value=0, label='Camera id', tag='camera_id', min_value=0, max_value=0, min_clamped=True, max_clamped=True, callback=callback_camera_id)
            dpg.add_combo(['image', 'albedo', 'normal', 'roughness', 'specularTint', 'visibility'], label='Render Type', default_value=self.render_type, callback=callback_render_type)
            dpg.add_slider_float(default_value=0, min_value=-1, max_value=1, label='Envmap rotation', callback=callback_env_rot)
            dpg.add_slider_float(default_value=1, min_value=0, max_value=2, label='Envmap brightness', callback=callback_env_brightness)

            dpg.add_slider_int(default_value=0, label='Frame id', tag='frame_id', min_value=0, max_value=0, callback=callback_frame_id)
            with dpg.group(horizontal=True, xoffset=150):
                dpg.add_button(label='Big pose', tag='big_pose', callback=callback_bigpose)
                dpg.add_button(label='T pose', tag='t_pose', callback=callback_tpose)

            with dpg.group(horizontal=True, xoffset=120):
                dpg.add_checkbox(label='Specular', tag='specular', default_value=False, callback=callback_specular)
                dpg.add_checkbox(label='Visibility', tag='visibility', default_value=False, callback=callback_visibility)
                dpg.add_checkbox(label='is_novel_pose', tag='is_novel_pose', default_value=True, callback=callback_is_novel_pose)
                dpg.add_checkbox(label='Dxyz', tag='dxyz', default_value=True, callback=callback_dxyz)

            dpg.add_color_edit((0, 0, 0, 255), label="Background color", width=200, tag='background', callback=callback_background, no_alpha=True)

            my_separator()

            dpg.add_text('Load novel envmap, e.g., /tmp/sunrise.exr')
            dpg.add_input_text(default_value='', callback=callback_load_envmap)

            dpg.add_text('Load novel poses (AIST format), e.g., /tmp/gBR_sBM_cAll_d05_mBR4_ch08.pkl')
            dpg.add_input_text(tag='novel_pose_text', default_value='')
            dpg.add_button(label='Load', callback=callback_load_novel_pose)
            dpg.add_slider_int(default_value=0, label='Novel pose frame id', tag='novel_pose_frame_id', min_value=0, max_value=0, callback=callback_novel_frame_id)

            # dpg.add_text('Camera path')
            # dpg.add_input_text(tag='camera_path', default_value='')
            # with dpg.group(horizontal=True):
            #     dpg.add_button(label='Load camera', callback=callback_load_camera)
            #     dpg.add_button(label='Save camera', callback=callback_save_camera)

            dpg.add_text('Texture path (RGBA png file)')
            dpg.add_input_text(tag='texture_path', default_value='')
            with dpg.group(horizontal=True):
                dpg.add_button(label='Load texture', callback=callback_load_texture)
                dpg.add_button(label='Resume', callback=callback_edit_resume)
                dpg.add_button(label='Edit!', callback=callback_optimize_edit)

            # dpg.add_text('Image output directory')
            # dpg.add_input_text(tag='image_out_dir', default_value='')
            # dpg.add_combo(['image', 'frame_seq', 'novel_pose_frame_seq', 'env_rot_seq'], tag='image_out_type', label='Output type', default_value='image')
            # with dpg.group(horizontal=True, xoffset=200):
            #     dpg.add_button(label='Render!!', callback=callback_render_seq)
            #     dpg.add_button(label='Resume', callback=callback_edit_resume)

        with dpg.item_handler_registry(tag='edit_handler'):
            dpg.add_item_hover_handler(callback=callback_image_hover)
            dpg.add_item_clicked_handler(callback=callback_image_clicked)
        dpg.bind_item_handler_registry('image', 'edit_handler')

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_drag_pan)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_release_rotate)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_release_pan)

        # with dpg.handler_registry():
        #     dpg.add_key_press_handler(dpg.mvKey_Left, callback=callback_key)
        #     dpg.add_key_press_handler(dpg.mvKey_Right, callback=callback_key)
        #     dpg.add_key_press_handler(dpg.mvKey_A, callback=callback_key)
        #     dpg.add_key_press_handler(dpg.mvKey_D, callback=callback_key)

        dpg.setup_dearpygui()
        dpg.show_viewport()

def main():
    H, W = (1000, 1000)
    # H, W = (1024, 750)
    net_init('127.0.0.1', 23456)
    gui = GUI(
        hw=(H, W),
    )
    gui.register_dpg()
    gui.render_loop()


if __name__ == '__main__':
    main()