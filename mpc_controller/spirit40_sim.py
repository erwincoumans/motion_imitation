import re
import numpy as np

URDF_NAME = "quadruped/spirit40newer.urdf"
START_POS = [0, 0, 0.43]
MPC_BODY_MASS = 12
MPC_BODY_INERTIA = (0.07335, 0, 0, 0, 0.25068, 0, 0, 0, 0.25447)
MPC_BODY_HEIGHT = 0.32
time_step = 0.001
ACTION_REPEAT = 10
MPC_VELOCITY_MULTIPLIER = 0.7


_IDENTITY_ORIENTATION=[0,0,0,1]
HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

_DEFAULT_HIP_POSITIONS = (
    (0.23, -0.12, 0),
    (0.23, 0.12, 0),
    (-0.23, -0.12, 0),
    (-0.23, 0.12, 0),
)
_BODY_B_FIELD_NUMBER = 2
_LINK_A_FIELD_NUMBER = 3

HIP_JOINT_OFFSET = 0.0
UPPER_LEG_JOINT_OFFSET = 0
KNEE_JOINT_OFFSET = 0


SPIRIT40_DEFAULT_ABDUCTION_ANGLE = 0
SPIRIT40_DEFAULT_HIP_ANGLE = -0.7
SPIRIT40_DEFAULT_KNEE_ANGLE = 1.4
NUM_LEGS = 4
NUM_MOTORS = 12
# Bases on the readings from Laikago's default pose.
INIT_MOTOR_ANGLES = np.array([
    SPIRIT40_DEFAULT_ABDUCTION_ANGLE,
    SPIRIT40_DEFAULT_HIP_ANGLE,
    SPIRIT40_DEFAULT_KNEE_ANGLE
] * NUM_LEGS)

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

#Use a PD controller
MOTOR_CONTROL_POSITION = 1 
# Apply motor torques directly.
MOTOR_CONTROL_TORQUE = 2
# Apply a tuple (q, qdot, kp, kd, tau) for each motor. Here q, qdot are motor
# position and velocities. kp and kd are PD gains. tau is the additional
# motor torque. This is the most flexible control mode.
MOTOR_CONTROL_HYBRID = 3
MOTOR_CONTROL_PWM = 4 #only for Minitaur



MOTOR_COMMAND_DIMENSION = 5

# These values represent the indices of each field in the motor command tuple
POSITION_INDEX = 0
POSITION_GAIN_INDEX = 1
VELOCITY_INDEX = 2
VELOCITY_GAIN_INDEX = 3
TORQUE_INDEX = 4


class LaikagoMotorModel(object):
  """A simple motor model for Laikago.

    When in POSITION mode, the torque is calculated according to the difference
    between current and desired joint angle, as well as the joint velocity.
    For more information about PD control, please refer to:
    https://en.wikipedia.org/wiki/PID_controller.

    The model supports a HYBRID mode in which each motor command can be a tuple
    (desired_motor_angle, position_gain, desired_motor_velocity, velocity_gain,
    torque).

  """

  def __init__(self,
               kp,
               kd,
               torque_limits=None,
               motor_control_mode=MOTOR_CONTROL_POSITION):
    self._kp = kp
    self._kd = kd
    self._torque_limits = torque_limits
    if torque_limits is not None:
      if isinstance(torque_limits, (collections.Sequence, np.ndarray)):
        self._torque_limits = np.asarray(torque_limits)
      else:
        self._torque_limits = np.full(NUM_MOTORS, torque_limits)
    self._motor_control_mode = motor_control_mode
    self._strength_ratios = np.full(NUM_MOTORS, 1)

  def set_strength_ratios(self, ratios):
    """Set the strength of each motors relative to the default value.

    Args:
      ratios: The relative strength of motor output. A numpy array ranging from
        0.0 to 1.0.
    """
    self._strength_ratios = ratios

  def set_motor_gains(self, kp, kd):
    """Set the gains of all motors.

    These gains are PD gains for motor positional control. kp is the
    proportional gain and kd is the derivative gain.

    Args:
      kp: proportional gain of the motors.
      kd: derivative gain of the motors.
    """
    self._kp = kp
    self._kd = kd

  def set_voltage(self, voltage):
    pass

  def get_voltage(self):
    return 0.0

  def set_viscous_damping(self, viscous_damping):
    pass

  def get_viscous_dampling(self):
    return 0.0

  def convert_to_torque(self,
                        motor_commands,
                        motor_angle,
                        motor_velocity,
                        true_motor_velocity,
                        motor_control_mode=None):
    """Convert the commands (position control or torque control) to torque.

    Args:
      motor_commands: The desired motor angle if the motor is in position
        control mode. The pwm signal if the motor is in torque control mode.
      motor_angle: The motor angle observed at the current time step. It is
        actually the true motor angle observed a few milliseconds ago (pd
        latency).
      motor_velocity: The motor velocity observed at the current time step, it
        is actually the true motor velocity a few milliseconds ago (pd latency).
      true_motor_velocity: The true motor velocity. The true velocity is used to
        compute back EMF voltage and viscous damping.
      motor_control_mode: A MotorControlMode enum.

    Returns:
      actual_torque: The torque that needs to be applied to the motor.
      observed_torque: The torque observed by the sensor.
    """
    del true_motor_velocity
    if not motor_control_mode:
      motor_control_mode = self._motor_control_mode

    # No processing for motor torques
    if motor_control_mode is MOTOR_CONTROL_TORQUE:
      assert len(motor_commands) == NUM_MOTORS
      motor_torques = self._strength_ratios * motor_commands
      return motor_torques, motor_torques

    desired_motor_angles = None
    desired_motor_velocities = None
    kp = None
    kd = None
    additional_torques = np.full(NUM_MOTORS, 0)
    if motor_control_mode is MOTOR_CONTROL_POSITION:
      assert len(motor_commands) == NUM_MOTORS
      kp = self._kp
      kd = self._kd
      desired_motor_angles = motor_commands
      desired_motor_velocities = np.full(NUM_MOTORS, 0)
    elif motor_control_mode is MOTOR_CONTROL_HYBRID:
      # The input should be a 60 dimension vector
      assert len(motor_commands) == MOTOR_COMMAND_DIMENSION * NUM_MOTORS
      kp = motor_commands[POSITION_GAIN_INDEX::MOTOR_COMMAND_DIMENSION]
      kd = motor_commands[VELOCITY_GAIN_INDEX::MOTOR_COMMAND_DIMENSION]
      desired_motor_angles = motor_commands[
          POSITION_INDEX::MOTOR_COMMAND_DIMENSION]
      desired_motor_velocities = motor_commands[
          VELOCITY_INDEX::MOTOR_COMMAND_DIMENSION]
      additional_torques = motor_commands[TORQUE_INDEX::MOTOR_COMMAND_DIMENSION]
    motor_torques = -1 * (kp * (motor_angle - desired_motor_angles)) - kd * (
        motor_velocity - desired_motor_velocities) + additional_torques
    motor_torques = self._strength_ratios * motor_torques
    if self._torque_limits is not None:
      if len(self._torque_limits) != len(motor_torques):
        raise ValueError(
            "Torque limits dimension does not match the number of motors.")
      motor_torques = np.clip(motor_torques, -1.0 * self._torque_limits,
                              self._torque_limits)

    return motor_torques, motor_torques

  

class SimpleRobot(object):
  def __init__(self, pybullet_client, robot_uid):
    self.pybullet_client = pybullet_client
    self.quadruped = robot_uid
    self.num_legs = NUM_LEGS
    self.num_motors = NUM_MOTORS
    self._BuildJointNameToIdDict()
    self._BuildUrdfIds()
    self._BuildMotorIdList()
    self.ResetPose()
    self._motor_enabled_list = [True] * self.num_motors
    self._step_counter = 0
    self._state_action_counter = 0
    self._motor_offset= np.array([0]*12)
    self._motor_direction= np.array([1,  1,  1,  1,  1,  1, 1,  1,  1,  1,  1,  1])
    self.ReceiveObservation()
    self._kp = self.GetMotorPositionGains()
    self._kd = self.GetMotorVelocityGains()
    self._motor_model = LaikagoMotorModel(kp=self._kp, kd=self._kd, motor_control_mode=MOTOR_CONTROL_HYBRID)
    self._SettleDownForReset(reset_time=1.0)


  def ResetPose(self):
    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self.pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self.pybullet_client.VELOCITY_CONTROL,
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
      self.pybullet_client.resetJointState(
          self.quadruped, self._joint_name_to_id[name], angle, targetVelocity=0)
          

  def _SettleDownForReset(self, reset_time):
    self.ReceiveObservation()
    if reset_time <= 0:
      return
    for _ in range(500):
      self._StepInternal(
          INIT_MOTOR_ANGLES,
          motor_control_mode=MOTOR_CONTROL_POSITION)
        
  def _GetMotorNames(self):
    return MOTOR_NAMES
    
  def _BuildMotorIdList(self):
    self._motor_id_list = [
        self._joint_name_to_id[motor_name]
        for motor_name in self._GetMotorNames()
    ]
  
  def GetMotorPositionGains(self):
    return [220.]*self.num_motors
    
  def GetMotorVelocityGains(self):
    return np.array([1., 2., 2., 1., 2., 2., 1., 2., 2., 1., 2., 2.])
    
  def compute_jacobian(self, robot, link_id):
    """Computes the Jacobian matrix for the given link.

    Args:
      robot: A robot instance.
      link_id: The link id as returned from loadURDF.

    Returns:
      The 3 x N transposed Jacobian matrix. where N is the total DoFs of the
      robot. For a quadruped, the first 6 columns of the matrix corresponds to
      the CoM translation and rotation. The columns corresponds to a leg can be
      extracted with indices [6 + leg_id * 3: 6 + leg_id * 3 + 3].
    """
    all_joint_angles = [state[0] for state in robot._joint_states]
    zero_vec = [0] * len(all_joint_angles)
    jv, _ = self.pybullet_client.calculateJacobian(robot.quadruped, link_id,
                                                    (0, 0, 0), all_joint_angles,
                                                    zero_vec, zero_vec)
    jacobian = np.array(jv)
    assert jacobian.shape[0] == 3
    return jacobian
  
  def ComputeJacobian(self, leg_id):
    """Compute the Jacobian for a given leg."""
    # Does not work for Minitaur which has the four bar mechanism for now.
    assert len(self._foot_link_ids) == self.num_legs
    return self.compute_jacobian(
        robot=self,
        link_id=self._foot_link_ids[leg_id],
    )
    
  def MapContactForceToJointTorques(self, leg_id, contact_force):
    """Maps the foot contact force to the leg joint torques."""
    jv = self.ComputeJacobian(leg_id)
    all_motor_torques = np.matmul(contact_force, jv)
    motor_torques = {}
    motors_per_leg = self.num_motors // self.num_legs
    com_dof = 6
    for joint_id in range(leg_id * motors_per_leg,
                          (leg_id + 1) * motors_per_leg):
      motor_torques[joint_id] = all_motor_torques[
          com_dof + joint_id] * self._motor_direction[joint_id]

    return motor_torques
    
  def GetBaseRollPitchYaw(self):
    """Get minitaur's base orientation in euler angle in the world frame.

    Returns:
      A tuple (roll, pitch, yaw) of the base in world frame.
    """
    orientation = self.GetTrueBaseOrientation()
    roll_pitch_yaw = self.pybullet_client.getEulerFromQuaternion(orientation)
    return np.asarray(roll_pitch_yaw)
    
  
  def joint_angles_from_link_position(
  self,
    robot,
    link_position,
    link_id,
    joint_ids,
    position_in_world_frame,
    base_translation = (0, 0, 0),
    base_rotation = (0, 0, 0, 1)):
    """Uses Inverse Kinematics to calculate joint angles.

    Args:
      robot: A robot instance.
      link_position: The (x, y, z) of the link in the body or the world frame,
        depending on whether the argument position_in_world_frame is true.
      link_id: The link id as returned from loadURDF.
      joint_ids: The positional index of the joints. This can be different from
        the joint unique ids.
      position_in_world_frame: Whether the input link_position is specified
        in the world frame or the robot's base frame.
      base_translation: Additional base translation.
      base_rotation: Additional base rotation.

    Returns:
      A list of joint angles.
    """
    if not position_in_world_frame:
      # Projects to local frame.
      base_position, base_orientation = self.pybullet_client.getBasePositionAndOrientation(self.quadruped)#robot.GetBasePosition(), robot.GetBaseOrientation()
      base_position, base_orientation = robot.pybullet_client.multiplyTransforms(
          base_position, base_orientation, base_translation, base_rotation)

      # Projects to world space.
      world_link_pos, _ = robot.pybullet_client.multiplyTransforms(
          base_position, base_orientation, link_position, _IDENTITY_ORIENTATION)
    else:
      world_link_pos = link_position

    ik_solver = 0
    all_joint_angles = robot.pybullet_client.calculateInverseKinematics(
        robot.quadruped, link_id, world_link_pos, solver=ik_solver)

    # Extract the relevant joint angles.
    joint_angles = [all_joint_angles[i] for i in joint_ids]
    return joint_angles
  
  def ComputeMotorAnglesFromFootLocalPosition(self, leg_id,
                                              foot_local_position):
    """Use IK to compute the motor angles, given the foot link's local position.

    Args:
      leg_id: The leg index.
      foot_local_position: The foot link's position in the base frame.

    Returns:
      A tuple. The position indices and the angles for all joints along the
      leg. The position indices is consistent with the joint orders as returned
      by GetMotorAngles API.
    """
    return self._EndEffectorIK(
        leg_id, foot_local_position, position_in_world_frame=False)
  def _EndEffectorIK(self, leg_id, position, position_in_world_frame):
    """Calculate the joint positions from the end effector position."""
    assert len(self._foot_link_ids) == self.num_legs
    toe_id = self._foot_link_ids[leg_id]
    motors_per_leg = self.num_motors // self.num_legs
    joint_position_idxs = [
        i for i in range(leg_id * motors_per_leg, leg_id * motors_per_leg +
                         motors_per_leg)
    ]
    joint_angles = self.joint_angles_from_link_position(
        robot=self,
        link_position=position,
        link_id=toe_id,
        joint_ids=joint_position_idxs,
        position_in_world_frame=position_in_world_frame)
    # Joint offset is necessary for Laikago.
    joint_angles = np.multiply(
        np.asarray(joint_angles) -
        np.asarray(self._motor_offset)[joint_position_idxs],
        self._motor_direction[joint_position_idxs])
    # Return the joing index (the same as when calling GetMotorAngles) as well
    # as the angles.
    return joint_position_idxs, joint_angles.tolist()
    
  def GetTimeSinceReset(self):
    return self._step_counter * time_step
    
  def GetHipPositionsInBaseFrame(self):
    return _DEFAULT_HIP_POSITIONS
    
  def GetBaseVelocity(self):
    """Get the linear velocity of minitaur's base.

    Returns:
      The velocity of minitaur's base.
    """
    velocity, _ = self.pybullet_client.getBaseVelocity(self.quadruped)
    return velocity

  def GetTrueBaseOrientation(self):
    pos,orn = self.pybullet_client.getBasePositionAndOrientation(
        self.quadruped)
    return orn
    
  def TransformAngularVelocityToLocalFrame(self, angular_velocity, orientation):
    """Transform the angular velocity from world frame to robot's frame.

    Args:
      angular_velocity: Angular velocity of the robot in world frame.
      orientation: Orientation of the robot represented as a quaternion.

    Returns:
      angular velocity of based on the given orientation.
    """
    # Treat angular velocity as a position vector, then transform based on the
    # orientation given by dividing (or multiplying with inverse).
    # Get inverse quaternion assuming the vector is at 0,0,0 origin.
    _, orientation_inversed = self.pybullet_client.invertTransform([0, 0, 0],
                                                                    orientation)
    # Transform the angular_velocity at neutral orientation using a neutral
    # translation and reverse of the given orientation.
    relative_velocity, _ = self.pybullet_client.multiplyTransforms(
        [0, 0, 0], orientation_inversed, angular_velocity,
        self.pybullet_client.getQuaternionFromEuler([0, 0, 0]))
    return np.asarray(relative_velocity)
    
  def GetBaseRollPitchYawRate(self):
    """Get the rate of orientation change of the minitaur's base in euler angle.

    Returns:
      rate of (roll, pitch, yaw) change of the minitaur's base.
    """
    angular_velocity = self.pybullet_client.getBaseVelocity(self.quadruped)[1]
    orientation = self.GetTrueBaseOrientation()
    return self.TransformAngularVelocityToLocalFrame(angular_velocity,
                                                     orientation)
                                                     
  def GetFootContacts(self):
    all_contacts = self.pybullet_client.getContactPoints(bodyA=self.quadruped)

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
    
  def GetTrueMotorAngles(self):
    """Gets the eight motor angles at the current moment, mapped to [-pi, pi].

    Returns:
      Motor angles, mapped to [-pi, pi].
    """
    self.ReceiveObservation()
    
    motor_angles = [state[0] for state in self._joint_states]
    motor_angles = np.multiply(
        np.asarray(motor_angles) - np.asarray(self._motor_offset),
        self._motor_direction)
    return motor_angles

  def GetPDObservation(self):
    self.ReceiveObservation()
    observation = []
    observation.extend(self.GetTrueMotorAngles())
    observation.extend(self.GetTrueMotorVelocities())
    q = observation[0:self.num_motors]
    qdot = observation[self.num_motors:2 * self.num_motors]
    return (np.array(q), np.array(qdot))


  def GetTrueMotorVelocities(self):
    """Get the velocity of all eight motors.

    Returns:
      Velocities of all eight motors.
    """
    motor_velocities = [state[1] for state in self._joint_states]

    motor_velocities = np.multiply(motor_velocities, self._motor_direction)
    return motor_velocities
    

  def GetTrueObservation(self):
    self.ReceiveObservation()
    observation = []
    observation.extend(self.GetTrueMotorAngles())
    observation.extend(self.GetTrueMotorVelocities())
    observation.extend(self.GetTrueMotorTorques())
    observation.extend(self.GetTrueBaseOrientation())
    observation.extend(self.GetTrueBaseRollPitchYawRate())
    return observation
    
  def ApplyAction(self, motor_commands, motor_control_mode):
    """Apply the motor commands using the motor model.

    Args:
      motor_commands: np.array. Can be motor angles, torques, hybrid commands
      motor_control_mode: A MotorControlMode enum.
    """
    motor_commands = np.asarray(motor_commands)
    q, qdot = self.GetPDObservation()
    qdot_true = self.GetTrueMotorVelocities()
    actual_torque, observed_torque = self._motor_model.convert_to_torque(
        motor_commands, q, qdot, qdot_true, motor_control_mode)
    
    # The torque is already in the observation space because we use
    # GetMotorAngles and GetMotorVelocities.
    self._observed_motor_torques = observed_torque

    # Transform into the motor space when applying the torque.
    self._applied_motor_torque = np.multiply(actual_torque,
                                             self._motor_direction)
    motor_ids = []
    motor_torques = []

    for motor_id, motor_torque, motor_enabled in zip(self._motor_id_list,
                                                     self._applied_motor_torque,
                                                     self._motor_enabled_list):
      if motor_enabled:
        motor_ids.append(motor_id)
        motor_torques.append(motor_torque)
      else:
        motor_ids.append(motor_id)
        motor_torques.append(0)
    self._SetMotorTorqueByIds(motor_ids, motor_torques)

  def _SetMotorTorqueByIds(self, motor_ids, torques):
    self.pybullet_client.setJointMotorControlArray(
        bodyIndex=self.quadruped,
        jointIndices=motor_ids,
        controlMode=self.pybullet_client.TORQUE_CONTROL,
        forces=torques)
        
  def ReceiveObservation(self):    
    self._joint_states = self.pybullet_client.getJointStates(self.quadruped, self._motor_id_list)

  def _StepInternal(self, action, motor_control_mode):
    self.ApplyAction(action, motor_control_mode)
    self.pybullet_client.stepSimulation()
    self.ReceiveObservation()
    self._state_action_counter += 1
    
  def Step(self, action):
    """Steps simulation."""
    #if self._enable_action_filter:
    #  action = self._FilterAction(action)

    for i in range(ACTION_REPEAT):
      #proc_action = self.ProcessAction(action, i)
      proc_action = action
      self._StepInternal(proc_action, motor_control_mode=MOTOR_CONTROL_HYBRID)
      self._step_counter += 1
    
  def _BuildJointNameToIdDict(self):
    num_joints = self.pybullet_client.getNumJoints(self.quadruped)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
      self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
      
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

    return

  def link_position_in_base_frame( self,   link_id ):
    """Computes the link's local position in the robot frame.

    Args:
      robot: A robot instance.
      link_id: The link to calculate its relative position.

    Returns:
      The relative position of the link.
    """
    base_position, base_orientation = self.pybullet_client.getBasePositionAndOrientation(self.quadruped)
    inverse_translation, inverse_rotation = self.pybullet_client.invertTransform(
        base_position, base_orientation)

    link_state = self.pybullet_client.getLinkState(self.quadruped, link_id)
    link_position = link_state[0]
    link_local_position, _ = self.pybullet_client.multiplyTransforms(
        inverse_translation, inverse_rotation, link_position, (0, 0, 0, 1))

    return np.array(link_local_position)



  def GetFootLinkIDs(self):
    """Get list of IDs for all foot links."""
    return self._foot_link_ids
    
  def GetFootPositionsInBaseFrame(self):
    """Get the robot's foot position in the base frame."""
    assert len(self._foot_link_ids) == self.num_legs
    foot_positions = []
    for foot_id in self.GetFootLinkIDs():
      foot_positions.append(
          self.link_position_in_base_frame(link_id=foot_id)
          )
    return np.array(foot_positions)
    
