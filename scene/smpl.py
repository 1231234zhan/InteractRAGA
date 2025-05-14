
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
try:
    from pytorch3d.transforms import axis_angle_to_matrix
except:
    pass

pose_num = 24
shape_num = 10

v_template = None
shapedirs = None
J_regressor = None
f = None
weights = None
posedirs = None
parent = None

t_poses = np.zeros(72, dtype=np.float32)
big_poses = np.zeros(72, dtype=np.float32)
big_poses[5] = np.deg2rad(30)
big_poses[8] = np.deg2rad(-30) 

def init(pkl_path):
    '''v_template (6890, 3) float32
    shapedirs (6890, 3, 10) float32
    J_regressor (24, 6890) float32
    kintree_table (2, 24) int64
    f (13776, 3) int64
    weights (6890, 24) float32
    posedirs (6890, 3, 207) float32
    '''
    global v_template, shapedirs, J_regressor, f, weights, posedirs, parent
    smpl_param = load_smpl_weight(pkl_path)
    v_template = smpl_param['v_template']
    shapedirs = smpl_param['shapedirs'][:,:,:10]
    J_regressor = smpl_param['J_regressor']
    f = smpl_param['f']
    weights = smpl_param['weights']
    posedirs = smpl_param['posedirs']
    parent = smpl_param['kintree_table'][0]

def load_smpl_weight(pkl_path):
    import pickle
    with open(pkl_path, 'rb') as file:
        smpl_pickle_obj = pickle.load(file, encoding='latin1')

    smpl_param = {}
    _key = ['v_template', 'shapedirs', 'J_regressor', 'kintree_table', 'f', 'weights', 'posedirs']
    for key in _key:
        if key == 'J_regressor':
            smpl_param[key] = np.array(smpl_pickle_obj[key].toarray(), dtype=np.float32)
        elif key == 'kintree_table' or key == 'f':
            smpl_param[key] = np.array(smpl_pickle_obj[key], dtype=int)
        else:
            smpl_param[key] = np.array(smpl_pickle_obj[key], dtype=np.float32)
    smpl_param['parent'] = smpl_param['kintree_table'][0]
    return smpl_param

def rotvec2mat(vec):
    if isinstance(vec, np.ndarray):
        return Rotation.from_rotvec(vec).as_matrix()
    return axis_angle_to_matrix(vec)

def transform_matrix(joint, R_pose):
    global parent
    pose_num = 24
    if isinstance(joint, np.ndarray):
        assert isinstance(R_pose, np.ndarray)
        pa = parent
        joint_offset = np.copy(joint)
        G = np.tile(np.eye(4, dtype=np.float32), [pose_num, 1, 1])

        joint_offset[1:] = joint[1:] - joint[pa[1:]]

        G[:,:3,:3] = R_pose
        G[:,:3,3] = joint_offset

        for i in range(1, pose_num):
            G[i] = G[pa[i]] @ G[i]
        return G

    elif isinstance(joint, torch.Tensor):
        assert isinstance(R_pose, torch.Tensor)
        pa = torch.tensor(parent)
        joint_offset = torch.clone(joint)
        joint_offset[1:] = joint[1:] - joint[pa[1:]]
        R_pose_0 = F.pad(R_pose, [0,0,0,1], value=0)   # N33 -> N43
        joint_offset_1 = F.pad(joint_offset, [0,1], value=1)[...,None]   # N3 -> N41
        G = torch.cat([R_pose_0, joint_offset_1], dim=-1)

        G_list = [G[0]]
        for i in range(1, pose_num):
            Gi = G_list[pa[i]] @ G[i]
            G_list.append(Gi)
        G = torch.stack(G_list, dim=0)
        return G

def get_joint_vert(pose, shape):
    global v_template, shapedirs, J_regressor, f, weights, posedirs, parent
    v_shape = np.dot(shapedirs, shape[:10]) + v_template   # (6890, 3, 10), (10) -> (6890, 3)
    joint = np.matmul(J_regressor, v_shape)  # (24, 6890), (6890, 3) -> (24, 3)

    R_pose = rotvec2mat(pose.reshape((-1, 3)))  # (72) -> (24, 3, 3)
    R_pose_c = rotvec2mat(np.zeros_like(pose).reshape((-1, 3)))
    A = transform_matrix(joint, R_pose)
    Ac = transform_matrix(joint, R_pose_c)
    Ac_inv = np.linalg.inv(Ac)
    joint_posed = A[:,:3,3]

    G = np.matmul(A, Ac_inv)

    _R_tiled_eye = np.eye(3)[None,...]
    v_pose = v_shape + np.dot(posedirs, (R_pose[1:] - _R_tiled_eye).reshape(-1)) # (6890, 3, 207), (23, 3, 3) -> (6890, 3)

    G_weight = np.einsum('vp,pij->vij', weights, G) # (6890, 24), (24, 4, 4) -> (6890, 4, 4)
    _v_pose_pad = np.pad(v_pose, [[0,0],[0,1]], constant_values=1)
    v_posed = np.einsum('vij,vj->vi', G_weight, _v_pose_pad)[:,:3]

    return joint_posed, v_posed
