import numpy as np
from motion_imitation.utilities  import pose3d
from pybullet_utils import transformations

URDF_FILENAME = "quadruped/vision60.urdf"

REF_POS_SCALE = 1
INIT_POS = np.array([0, 0, 0])
INIT_ROT = np.array([0, 0, 0, 1.0])

SIM_TOE_JOINT_IDS = [
    3, # left hand
    7, # left foot
    11, # right hand
    15 # right foot
]
SIM_HIP_JOINT_IDS = [0, 4, 8, 12]
SIM_ROOT_OFFSET = np.array([0, 0, 0])
SIM_TOE_OFFSET_LOCAL = [
    np.array([0, -0.05, 0.0]),
    np.array([0, -0.05, 0.01]),
    np.array([0, 0.05, 0.0]),
    np.array([0, 0.05, 0.01])
]

DEFAULT_JOINT_POSE = np.array([0, 0.7, 1.5, 0, 0.7, 1.5, 0, 0.7, 1.5, 0, 0.7, 1.5])
JOINT_DAMPING = [0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01]

FORWARD_DIR_OFFSET = np.array([0, 0, 0])

