# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pybullet simulation of a Laikago robot."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import re
import numpy as np
import pybullet as pyb  # pytype: disable=import-error

from motion_imitation.robots import laikago_constants
from motion_imitation.robots import laikago_motor
from motion_imitation.robots import minitaur
from motion_imitation.robots import robot_config
from motion_imitation.envs import locomotion_gym_config

NUM_MOTORS = 12
NUM_LEGS = 4
MOTOR_NAMES = [
    "FR_hip_joint",
    "FR_upper_joint",
    "FR_lower_joint",
    "FL_hip_joint",
    "FL_upper_joint",
    "FL_lower_joint",
    "RR_hip_joint",
    "RR_upper_joint",
    "RR_lower_joint",
    "RL_hip_joint",
    "RL_upper_joint",
    "RL_lower_joint",
]
INIT_RACK_POSITION = [0, 0, 1]
INIT_POSITION = [0, 0, 0.32]
JOINT_DIRECTIONS = np.ones(12)
HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0.0
KNEE_JOINT_OFFSET = 0.0
DOFS_PER_LEG = 3
JOINT_OFFSETS = np.array(
    [HIP_JOINT_OFFSET, UPPER_LEG_JOINT_OFFSET, KNEE_JOINT_OFFSET] * 4)
PI = math.pi

MAX_MOTOR_ANGLE_CHANGE_PER_STEP = 0.2
_DEFAULT_HIP_POSITIONS = (
    (0.21, -0.1157, 0),
    (0.21, 0.1157, 0),
    (-0.21, -0.1157, 0),
    (-0.21, 0.1157, 0),
)

ABDUCTION_P_GAIN = 100.0
ABDUCTION_D_GAIN = 1.
HIP_P_GAIN = 100.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 100.0
KNEE_D_GAIN = 2.0

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8] * NUM_LEGS)

HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

URDF_FILENAME = "a1/a1.urdf"

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3


class A1(minitaur.Minitaur):
  """A simulation for the Laikago robot."""

  # At high replanning frequency, inaccurate values of BODY_MASS/INERTIA
  # doesn't seem to matter much. However, these values should be better tuned
  # when the replan frequency is low (e.g. using a less beefy CPU).
  MPC_BODY_MASS = 108 / 9.8
  MPC_BODY_INERTIA = (0.07335, 0, 0, 0, 0.25068, 0, 0, 0, 0.25447)

  MPC_BODY_HEIGHT = 0.24
  MPC_VELOCITY_MULTIPLIER = 0.5
  ACTION_CONFIG = [
      locomotion_gym_config.ScalarField(name="FR_hip_motor",
                                        upper_bound=0.802851455917,
                                        lower_bound=-0.802851455917),
      locomotion_gym_config.ScalarField(name="FR_upper_joint",
                                        upper_bound=4.18879020479,
                                        lower_bound=-1.0471975512),
      locomotion_gym_config.ScalarField(name="FR_lower_joint",
                                        upper_bound=-0.916297857297,
                                        lower_bound=-2.69653369433),
      locomotion_gym_config.ScalarField(name="FL_hip_motor",
                                        upper_bound=0.802851455917,
                                        lower_bound=-0.802851455917),
      locomotion_gym_config.ScalarField(name="FL_upper_joint",
                                        upper_bound=4.18879020479,
                                        lower_bound=-1.0471975512),
      locomotion_gym_config.ScalarField(name="FL_lower_joint",
                                        upper_bound=-0.916297857297,
                                        lower_bound=-2.69653369433),
      locomotion_gym_config.ScalarField(name="RR_hip_motor",
                                        upper_bound=0.802851455917,
                                        lower_bound=-0.802851455917),
      locomotion_gym_config.ScalarField(name="RR_upper_joint",
                                        upper_bound=4.18879020479,
                                        lower_bound=-1.0471975512),
      locomotion_gym_config.ScalarField(name="RR_lower_joint",
                                        upper_bound=-0.916297857297,
                                        lower_bound=-2.69653369433),
      locomotion_gym_config.ScalarField(name="RL_hip_motor",
                                        upper_bound=0.802851455917,
                                        lower_bound=-0.802851455917),
      locomotion_gym_config.ScalarField(name="RL_upper_joint",
                                        upper_bound=4.18879020479,
                                        lower_bound=-1.0471975512),
      locomotion_gym_config.ScalarField(name="RL_lower_joint",
                                        upper_bound=-0.916297857297,
                                        lower_bound=-2.69653369433),
  ]

  def __init__(
      self,
      pybullet_client,
      urdf_filename=URDF_FILENAME,
      enable_clip_motor_commands=False,
      time_step=0.001,
      action_repeat=10,
      sensors=None,
      control_latency=0.002,
      on_rack=False,
      enable_action_interpolation=True,
      enable_action_filter=False,
      motor_control_mode=None,
      reset_time=1,
      allow_knee_contact=False,
  ):
    self._urdf_filename = urdf_filename
    self._allow_knee_contact = allow_knee_contact
    self._enable_clip_motor_commands = enable_clip_motor_commands

    motor_kp = [
        ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN,
        HIP_P_GAIN, KNEE_P_GAIN, ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN,
        ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN
    ]
    motor_kd = [
        ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN,
        HIP_D_GAIN, KNEE_D_GAIN, ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN,
        ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN
    ]


    super(A1, self).__init__(
        pybullet_client=pybullet_client,
        time_step=time_step,
        action_repeat=action_repeat,
        num_motors=NUM_MOTORS,
        dofs_per_leg=DOFS_PER_LEG,
        motor_direction=JOINT_DIRECTIONS,
        motor_offset=JOINT_OFFSETS,
        motor_overheat_protection=False,
        motor_control_mode=motor_control_mode,
        motor_model_class=laikago_motor.LaikagoMotorModel,
        sensors=sensors,
        motor_kp=motor_kp,
        motor_kd=motor_kd,
        control_latency=control_latency,
        on_rack=on_rack,
        enable_action_interpolation=enable_action_interpolation,
        enable_action_filter=enable_action_filter,
        reset_time=reset_time)

  def _LoadRobotURDF(self):
    a1_urdf_path = self.GetURDFFile()
    if self._self_collision_enabled:
      self.quadruped = self._pybullet_client.loadURDF(
          a1_urdf_path,
          self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation(),
          flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
    else:
      self.quadruped = self._pybullet_client.loadURDF(
          a1_urdf_path, self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation())

  def _SettleDownForReset(self, default_motor_angles, reset_time):
    self.ReceiveObservation()
    if reset_time <= 0:
      return

    for _ in range(500):
      self._StepInternal(
          INIT_MOTOR_ANGLES,
          motor_control_mode=robot_config.MotorControlMode.POSITION)

    if default_motor_angles is not None:
      num_steps_to_reset = int(reset_time / self.time_step)
      for _ in range(num_steps_to_reset):
        self._StepInternal(
            default_motor_angles,
            motor_control_mode=robot_config.MotorControlMode.POSITION)

  def GetHipPositionsInBaseFrame(self):
    return _DEFAULT_HIP_POSITIONS

  def GetFootContacts(self):
    all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)

    contacts = [False, False, False, False]
    for contact in all_contacts:
      # Ignore self contacts
      if contact[_BODY_B_FIELD_NUMBER] == self.quadruped:
        continue
      try:
        toe_link_index = self._foot_link_ids.index(
            contact[_LINK_A_FIELD_NUMBER])
        contacts[toe_link_index] = True
      except ValueError:
        continue

    return contacts

  def ResetPose(self, add_constraint):
    del add_constraint
    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)
    for name, i in zip(MOTOR_NAMES, range(len(MOTOR_NAMES))):
      if "hip_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + HIP_JOINT_OFFSET
      elif "upper_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + UPPER_LEG_JOINT_OFFSET
      elif "lower_joint" in name:
        angle = INIT_MOTOR_ANGLES[i] + KNEE_JOINT_OFFSET
      else:
        raise ValueError("The name %s is not recognized as a motor joint." %
                         name)
      self._pybullet_client.resetJointState(self.quadruped,
                                            self._joint_name_to_id[name],
                                            angle,
                                            targetVelocity=0)

  def GetURDFFile(self):
    return self._urdf_filename

  def _BuildUrdfIds(self):
    """Build the link Ids from its name in the URDF file.

    Raises:
      ValueError: Unknown category of the joint name.
    """
    num_joints = self.pybullet_client.getNumJoints(self.quadruped)
    self._hip_link_ids = [-1]
    self._leg_link_ids = []
    self._motor_link_ids = []
    self._lower_link_ids = []
    self._foot_link_ids = []
    self._imu_link_ids = []

    for i in range(num_joints):
      joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      if HIP_NAME_PATTERN.match(joint_name):
        self._hip_link_ids.append(joint_id)
      elif UPPER_NAME_PATTERN.match(joint_name):
        self._motor_link_ids.append(joint_id)
      # We either treat the lower leg or the toe as the foot link, depending on
      # the urdf version used.
      elif LOWER_NAME_PATTERN.match(joint_name):
        self._lower_link_ids.append(joint_id)
      elif TOE_NAME_PATTERN.match(joint_name):
        #assert self._urdf_filename == URDF_WITH_TOES
        self._foot_link_ids.append(joint_id)
      elif IMU_NAME_PATTERN.match(joint_name):
        self._imu_link_ids.append(joint_id)
      else:
        raise ValueError("Unknown category of joint %s" % joint_name)

    self._leg_link_ids.extend(self._lower_link_ids)
    self._leg_link_ids.extend(self._foot_link_ids)

    #assert len(self._foot_link_ids) == NUM_LEGS
    self._hip_link_ids.sort()
    self._motor_link_ids.sort()
    self._lower_link_ids.sort()
    self._foot_link_ids.sort()
    self._leg_link_ids.sort()

  def _GetMotorNames(self):
    return MOTOR_NAMES

  def _GetDefaultInitPosition(self):
    if self._on_rack:
      return INIT_RACK_POSITION
    else:
      return INIT_POSITION

  def _GetDefaultInitOrientation(self):
    # The Laikago URDF assumes the initial pose of heading towards z axis,
    # and belly towards y axis. The following transformation is to transform
    # the Laikago initial orientation to our commonly used orientation: heading
    # towards -x direction, and z axis is the up direction.
    init_orientation = pyb.getQuaternionFromEuler([0., 0., 0.])
    return init_orientation

  def GetDefaultInitPosition(self):
    """Get default initial base position."""
    return self._GetDefaultInitPosition()

  def GetDefaultInitOrientation(self):
    """Get default initial base orientation."""
    return self._GetDefaultInitOrientation()

  def GetDefaultInitJointPose(self):
    """Get default initial joint pose."""
    joint_pose = (INIT_MOTOR_ANGLES + JOINT_OFFSETS) * JOINT_DIRECTIONS
    return joint_pose

  def ApplyAction(self, motor_commands, motor_control_mode=None):
    """Clips and then apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).N
      motor_control_mode: A MotorControlMode enum.
    """
    if self._enable_clip_motor_commands:
      motor_commands = self._ClipMotorCommands(motor_commands)
    super(A1, self).ApplyAction(motor_commands, motor_control_mode)

  def _ClipMotorCommands(self, motor_commands):
    """Clips motor commands.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).

    Returns:
      Clipped motor commands.
    """

    # clamp the motor command by the joint limit, in case weired things happens
    max_angle_change = MAX_MOTOR_ANGLE_CHANGE_PER_STEP
    current_motor_angles = self.GetMotorAngles()
    motor_commands = np.clip(motor_commands,
                             current_motor_angles - max_angle_change,
                             current_motor_angles + max_angle_change)
    return motor_commands

  @classmethod
  def GetConstants(cls):
    del cls
    return laikago_constants
