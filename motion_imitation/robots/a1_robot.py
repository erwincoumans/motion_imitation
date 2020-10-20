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
# pytype: disable=attribute-error
"""Real robot interface of A1 robot."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from absl import logging
from io import BytesIO
import math
import re
import numpy as np
import lcm
import threading
import time

from motion_imitation.robots import laikago_pose_utils
from motion_imitation.robots import a1
from motion_imitation.robots import a1_robot_velocity_estimator
from motion_imitation.robots import minitaur
from motion_imitation.robots import robot_config
from motion_imitation.robots.unitree_legged_sdk import comm
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
INIT_POSITION = [0, 0, 0.48]
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
ABDUCTION_D_GAIN = 1.0
HIP_P_GAIN = 100.0
HIP_D_GAIN = 2.0
KNEE_P_GAIN = 100.0
KNEE_D_GAIN = 2.0

COMMAND_CHANNEL_NAME = 'LCM_Low_Cmd'
STATE_CHANNEL_NAME = 'LCM_Low_State'

# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([
    laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
    laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
    laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE
] * NUM_LEGS)

HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

URDF_FILENAME = "a1/a1.urdf"

_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3


class A1Robot(a1.A1):
  """Interface for real A1 robot."""
  MPC_BODY_MASS = 14.2
  MPC_BODY_INERTIA = (0.07335, 0, 0, 0, 0.25068, 0, 0, 0, 0.25447)

  MPC_BODY_HEIGHT = 0.24
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

  def __init__(self, pybullet_client, time_step=0.002, **kwargs):
    """Initializes the robot class."""
    # Initialize pd gain vector
    self.motor_kps = np.array([ABDUCTION_P_GAIN, HIP_P_GAIN, KNEE_P_GAIN] * 4)
    self.motor_kds = np.array([ABDUCTION_D_GAIN, HIP_D_GAIN, KNEE_D_GAIN] * 4)
    self._pybullet_client = pybullet_client
    self.time_step = time_step

    # Robot state variables
    self._init_complete = False
    self._base_orientation = None
    self._raw_state = None
    self._last_raw_state = None
    self._motor_angles = np.zeros(12)
    self._motor_velocities = np.zeros(12)
    self._joint_states = None
    self._last_reset_time = time.time()
    self._velocity_estimator = a1_robot_velocity_estimator.VelocityEstimator(
        self)

    # Initiate LCM channel for robot state and actions
    self.lc = lcm.LCM()
    self._command_channel_name = COMMAND_CHANNEL_NAME
    # Send dummy command so that the robot starts responding
    command = comm.LowCmd()
    command.levelFlag = 0xff  # pylint: disable=C0103
    self.lc.publish(self._command_channel_name, command)
    self._state_channel_name = STATE_CHANNEL_NAME
    self._state_channel = self.lc.subscribe(STATE_CHANNEL_NAME,
                                            self.ReceiveObservationAsync)

    self._is_alive = True
    self.subscribe_thread = threading.Thread(target=self._LCMSubscribeLoop,
                                             args=())
    self.subscribe_thread.start()
    while self._last_raw_state is None:
      logging.info("Robot sensor reading not ready yet, sleep for 1 second...")
      time.sleep(1)

    kwargs['on_rack'] = True
    super(A1Robot, self).__init__(pybullet_client,
                                  time_step=time_step,
                                  **kwargs)
    self._init_complete = True

  def _LCMSubscribeLoop(self):
    while self._is_alive:
      self.lc.handle_timeout(100)

  def ReceiveObservation(self):
    """Receives observation from robot.

    Synchronous ReceiveObservation is not supported in A1,
    so changging it to noop instead.
    """
    pass

  def ReceiveObservationAsync(self, channel, data):
    """Receive the observation from sensors.

    This function is called once per step. The observations are only updated
    when this function is called.
    """
    del channel  # unused
    stream = BytesIO(data)
    state = comm.LowState()
    stream.readinto(state)  # pytype: disable=wrong-arg-types

    self._last_raw_state = self._raw_state
    self._raw_state = state
    # Convert quaternion from wxyz to xyzw, which is default for Pybullet.
    q = state.imu.quaternion
    self._base_orientation = np.array([q[1], q[2], q[3], q[0]])
    self._motor_angles = np.array([motor.q for motor in state.motorState[:12]])
    self._motor_velocities = np.array(
        [motor.dq for motor in state.motorState[:12]])
    self._joint_states = np.array(
        list(zip(self._motor_angles, self._motor_velocities)))
    if self._init_complete:
      self._SetMotorAnglesInSim(self._motor_angles, self._motor_velocities)
      self._velocity_estimator.update(self._raw_state)

  def _SetMotorAnglesInSim(self, motor_angles, motor_velocities):
    for i, motor_id in enumerate(self._motor_id_list):
      self._pybullet_client.resetJointState(self.quadruped, motor_id,
                                            motor_angles[i],
                                            motor_velocities[i])

  def GetTrueMotorAngles(self):
    return self._motor_angles

  def GetMotorAngles(self):
    return minitaur.MapToMinusPiToPi(self._motor_angles)

  def GetMotorVelocities(self):
    return self._motor_velocities

  def GetBasePosition(self):
    return self._pybullet_client.getBasePositionAndOrientation(
        self.quadruped)[0]

  def GetBaseRollPitchYaw(self):
    return self._pybullet_client.getEulerFromQuaternion(self._base_orientation)

  def GetTrueBaseRollPitchYaw(self):
    return self._pybullet_client.getEulerFromQuaternion(self._base_orientation)

  def GetBaseRollPitchYawRate(self):
    return self.GetTrueBaseRollPitchYawRate()

  def GetTrueBaseRollPitchYawRate(self):
    return np.array(self._raw_state.imu.gyroscope)

  def GetBaseVelocity(self):
    return self._velocity_estimator.estimated_velocity

  def GetFootContacts(self):
    return np.array(self._raw_state.footForce) > 20

  def GetTimeSinceReset(self):
    return time.time() - self._last_reset_time

  @property
  def motor_velocities(self):
    return self._motor_velocities


  def ApplyAction(self, motor_commands, motor_control_mode=None):
    """Clips and then apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands,
        or motor pwms (for Minitaur only).
      motor_control_mode: A MotorControlMode enum.
    """
    if motor_control_mode is None:
      motor_control_mode = self._motor_control_mode

    command = comm.LowCmd()
    command.levelFlag = 0xff  #pylint:disable=invalid-name

    if motor_control_mode == robot_config.MotorControlMode.POSITION:
      for motor_id in range(NUM_MOTORS):
        command.motorCmd[motor_id].mode = 0x0A
        command.motorCmd[motor_id].q = motor_commands[motor_id]
        command.motorCmd[motor_id].Kp = self.motor_kps[motor_id]
        command.motorCmd[motor_id].dq = 0
        command.motorCmd[motor_id].Kd = self.motor_kds[motor_id]
        command.motorCmd[motor_id].tau = 0

      # Gravity compensation
      command.motorCmd[0].tau = -0.65
      command.motorCmd[3].tau = 0.65
      command.motorCmd[6].tau = -0.65
      command.motorCmd[9].tau = 0.65
    elif motor_control_mode == robot_config.MotorControlMode.TORQUE:
      for motor_id in range(NUM_MOTORS):
        command.motorCmd[motor_id].mode = 0x0A
        command.motorCmd[motor_id].q = 0
        command.motorCmd[motor_id].Kp = 0
        command.motorCmd[motor_id].dq = 0
        command.motorCmd[motor_id].Kd = 0
        command.motorCmd[motor_id].tau = motor_commands[motor_id]
    elif motor_control_mode == robot_config.MotorControlMode.HYBRID:
      for motor_id in range(NUM_MOTORS):
        command.motorCmd[motor_id].mode = 0x0A
        command.motorCmd[motor_id].q = motor_commands[motor_id * 5]
        command.motorCmd[motor_id].Kp = motor_commands[motor_id * 5 + 1]
        command.motorCmd[motor_id].dq = motor_commands[motor_id * 5 + 2]
        command.motorCmd[motor_id].Kd = motor_commands[motor_id * 5 + 3]
        command.motorCmd[motor_id].tau = motor_commands[motor_id * 5 + 4]
    else:
      raise ValueError('Unknown motor control mode for A1 robot: {}.'.format(
          motor_control_mode))

    self.lc.publish(self._command_channel_name, command)

  def Reset(self, reload_urdf=True, default_motor_angles=None, reset_time=3.0):
    """Reset the robot to default motor angles."""
    super(A1Robot, self).Reset(reload_urdf=reload_urdf,
                               default_motor_angles=default_motor_angles,
                               reset_time=-1)
    logging.warning(
        "About to reset the robot, make sure the robot is hang-up.")

    if not default_motor_angles:
      default_motor_angles = a1.INIT_MOTOR_ANGLES

    current_motor_angles = self.GetMotorAngles()
    for t in np.arange(0, reset_time, self.time_step * self._action_repeat):
      blend_ratio = t / reset_time
      action = blend_ratio * default_motor_angles + (
          1 - blend_ratio) * current_motor_angles
      self.ApplyAction(action, robot_config.MotorControlMode.POSITION)
      time.sleep(self.time_step * self._action_repeat)

    if self._enable_action_filter:
      self._ResetActionFilter()

    self._velocity_estimator.reset()
    self._state_action_counter = 0
    self._step_counter = 0
    self._last_reset_time = time.time()

  def Terminate(self):
    self._is_alive = False
