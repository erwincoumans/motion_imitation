"""Example of whole body controller on A1 robot."""
from absl import app
from absl import flags
from absl import logging

import numpy as np
import os
from datetime import datetime
import pickle
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client
import scipy

from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller

# from motion_imitation.envs import env_builder
from motion_imitation.robots import a1_robot
from motion_imitation.robots import robot_config

flags.DEFINE_float("max_time_secs", 1., "max time to run the controller.")
flags.DEFINE_string("logdir", None, "where to log trajectories.")
FLAGS = flags.FLAGS

# Stand
# _DUTY_FACTOR = [1.] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0., 0., 0.]
# _MAX_TIME_SECONDS = 5

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
# )

# Tripod
# _STANCE_DURATION_SECONDS = [
#     0.7
# ] * 4
# _DUTY_FACTOR = [.8] * 4
# _INIT_PHASE_FULL_CYCLE = [0., 0.25, 0.5, 0.75]
# _MAX_TIME_SECONDS = 5

# _INIT_LEG_STATE = (
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
#     gait_generator_lib.LegState.STANCE,
# )

# Trotting
_STANCE_DURATION_SECONDS = [0.3] * 4
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]
_MAX_TIME_SECONDS = 5

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)


def _setup_controller(robot):
  """Demonstrates how to create a locomotion controller."""
  desired_speed = (0, 0)
  desired_twisting_speed = 0

  gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor=_DUTY_FACTOR,
      initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
      initial_leg_state=_INIT_LEG_STATE)

  state_estimator = com_velocity_estimator.COMVelocityEstimator(robot,
                                                                window_size=1)
  sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_height=robot.MPC_BODY_HEIGHT,
      foot_clearance=0.01)

  st_controller = torque_stance_leg_controller.TorqueStanceLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_body_height=robot.MPC_BODY_HEIGHT,
      body_mass=robot.MPC_BODY_MASS,
      body_inertia=robot.MPC_BODY_INERTIA)

  controller = locomotion_controller.LocomotionController(
      robot=robot,
      gait_generator=gait_generator,
      state_estimator=state_estimator,
      swing_leg_controller=sw_controller,
      stance_leg_controller=st_controller,
      clock=robot.GetTimeSinceReset)
  return controller


def _update_controller_params(controller, lin_speed, ang_speed):
  controller.swing_leg_controller.desired_speed = lin_speed
  controller.swing_leg_controller.desired_twisting_speed = ang_speed
  controller.stance_leg_controller.desired_speed = lin_speed
  controller.stance_leg_controller.desired_twisting_speed = ang_speed

def _generate_example_linear_angular_speed(t):
  """Creates an example speed profile based on time for demo purpose."""
  vx = 0.2
  vy = 0.2
  wz = 0.8

  time_points = (0, 5, 10, 15, 20, 25, 30)
  speed_points = ((0, 0, 0, 0), (0, 0, 0, wz), (vx, 0, 0, 0), (0, 0, 0, -wz),
                  (0, -vy, 0, 0), (0, 0, 0, 0), (0, 0, 0, wz))

  speed = scipy.interpolate.interp1d(time_points,
                                     speed_points,
                                     kind="previous",
                                     fill_value="extrapolate",
                                     axis=0)(t)

  return speed[0:3], speed[3]


def _run_example():
  """Runs the locomotion controller example."""
  p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  robot = a1_robot.A1Robot(
      pybullet_client=p,
      motor_control_mode=robot_config.MotorControlMode.HYBRID,
      enable_action_interpolation=False,
      time_step=0.002,
      action_repeat=1)
  controller = _setup_controller(robot)
  controller.reset()

  actions = []
  states = []

  start_time = robot.GetTimeSinceReset()
  current_time = start_time

  while current_time - start_time < FLAGS.max_time_secs:
    # Updates the controller behavior parameters.
    lin_speed, ang_speed = _generate_example_linear_angular_speed(current_time)
    _update_controller_params(controller, lin_speed, ang_speed)

    # Needed before every call to get_action().
    controller.update()
    current_time = robot.GetTimeSinceReset()
    hybrid_action, info = controller.get_action()
    actions.append(hybrid_action)
    robot.Step(hybrid_action)
    states.append(
        dict(timestamp=robot.GetTimeSinceReset(),
             base_rpy=robot.GetBaseRollPitchYaw(),
             motor_angles=robot.GetMotorAngles(),
             base_vel=robot.GetBaseVelocity(),
             base_vels_body_frame=controller.state_estimator.
             com_velocity_body_frame,
             base_rpy_rate=robot.GetBaseRollPitchYawRate(),
             motor_vels=robot.GetMotorVelocities(),
             contacts=robot.GetFootContacts(),
             qp_sol=info['qp_sol']))

  # robot.Reset()
  robot.Terminate()
  if FLAGS.logdir:
    logdir = os.path.join(FLAGS.logdir,
                          datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(logdir)
    np.savez(os.path.join(logdir, 'action.npz'), action=actions)
    pickle.dump(states, open(os.path.join(logdir, 'states.pkl'), 'wb'))
    logging.info("logged to: {}".format(logdir))


def main(argv):
  del argv
  _run_example()


if __name__ == "__main__":
  app.run(main)
