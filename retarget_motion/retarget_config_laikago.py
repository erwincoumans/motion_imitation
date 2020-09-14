import numpy as np
from motion_imitation.utilities import pose3d
from pybullet_utils  import transformations

URDF_FILENAME = "laikago/laikago_toes.urdf"

REF_POS_SCALE = 1
INIT_POS = np.array([0, 0, 0])
INIT_ROT = transformations.quaternion_from_euler(ai=np.pi / 2.0, aj=0, ak=np.pi / 2.0, axes="sxyz")

SIM_TOE_JOINT_IDS = [
    7, # left hand
    15, # left foot
    3, # right hand
    11 # right foot
]
SIM_HIP_JOINT_IDS = [4, 12, 0, 8]
SIM_ROOT_OFFSET = np.array([0, 0, 0])
SIM_TOE_OFFSET_LOCAL = [
    np.array([-0.02, 0.0, 0.0]),
    np.array([-0.02, 0.0, 0.01]),
    np.array([-0.02, 0.0, 0.0]),
    np.array([-0.02, 0.0, 0.01])
]

DEFAULT_JOINT_POSE = np.array([0, 0.67, -1.25, 0, 0.67, -1.25, 0, 0.67, -1.25, 0, 0.67, -1.25])
JOINT_DAMPING = [0.5, 0.05, 0.01,
                 0.5, 0.05, 0.01,
                 0.5, 0.05, 0.01,
                 0.5, 0.05, 0.01]


FORWARD_DIR_OFFSET = np.array([0, 0, 0.025])
