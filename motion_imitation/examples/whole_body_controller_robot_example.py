"""Example of whole body controller on A1 robot."""
from absl import app
from absl import flags
from absl import logging

import copy
import numpy as np
import os
from datetime import datetime
import time
import pickle
import pybullet  # pytype:disable=import-error
import pybullet_data
from pybullet_utils import bullet_client

from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller

# from motion_imitation.envs import env_builder
from motion_imitation.robots import a1_robot
from motion_imitation.robots import robot_config

flags.DEFINE_integer("max_time_secs", 1, "max time to run the controller.")
flags.DEFINE_string("logdir", None, "where to log trajectories.")
FLAGS = flags.FLAGS

_NUM_SIMULATION_ITERATION_STEPS = 300
_STANCE_DURATION_SECONDS = [
    0.5
] * 4  # For faster trotting (v > 1.5 ms reduce this to 0.13s).
_DUTY_FACTOR = [.75] * 4
_INIT_PHASE_FULL_CYCLE = [0., 0.25, 0.5, 0.]

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.STANCE,
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
  raw_states = []
  timestamps, com_vels, imu_rates = [], [], []
  start_time = robot.GetTimeSinceReset()
  current_time = start_time

  while current_time - start_time < FLAGS.max_time_secs:
    # Updates the controller behavior parameters.
    lin_speed, ang_speed = (0., 0., 0.), 0.
    _update_controller_params(controller, lin_speed, ang_speed)

    # Needed before every call to get_action().
    controller.update()
    hybrid_action = controller.get_action()
    raw_states.append(copy.deepcopy(robot._raw_state))  # pylint:disable=protected-access
    com_vels.append(robot.GetBaseVelocity().copy())
    imu_rates.append(robot.GetBaseRollPitchYawRate().copy())
    actions.append(hybrid_action)
    robot.Step(hybrid_action)
    current_time = robot.GetTimeSinceReset()
    timestamps.append(current_time)
    time.sleep(0.003)

  robot.Reset()
  robot.Terminate()
  if FLAGS.logdir:
    logdir = os.path.join(FLAGS.logdir,
                          datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(logdir)
    np.savez(os.path.join(logdir, 'action.npz'),
             action=actions,
             com_vels=com_vels,
             imu_rates=imu_rates,
             timestamps=timestamps)
    pickle.dump(raw_states, open(os.path.join(logdir, 'raw_states.pkl'), 'wb'))
    logging.info("logged to: {}".format(logdir))


def main(argv):
  del argv
  _run_example()


if __name__ == "__main__":
  app.run(main)
