"""Simple script for executing random actions on A1 robot."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from absl import app
from absl import flags
from tqdm import tqdm

from motion_imitation.envs import env_builder
from motion_imitation.robots import a1
from motion_imitation.robots import laikago
from motion_imitation.robots import robot_config

FLAGS = flags.FLAGS
flags.DEFINE_enum('robot_type', 'A1', ['A1', 'Laikago'], 'Robot Type.')
flags.DEFINE_enum('motor_control_mode', 'Torque',
                  ['Torque', 'Position', 'Hybrid'], 'Motor Control Mode.')
flags.DEFINE_bool('on_rack', False, 'Whether to put the robot on rack.')

ROBOT_CLASS_MAP = {'A1': a1.A1, 'Laikago': laikago.Laikago}

MOTOR_CONTROL_MODE_MAP = {
    'Torque': robot_config.MotorControlMode.TORQUE,
    'Position': robot_config.MotorControlMode.POSITION,
    'Hybrid': robot_config.MotorControlMode.HYBRID
}


def main(_):
  robot = ROBOT_CLASS_MAP[FLAGS.robot_type]
  motor_control_mode = MOTOR_CONTROL_MODE_MAP[FLAGS.motor_control_mode]
  env = env_builder.build_regular_env(robot,
                                      motor_control_mode=motor_control_mode,
                                      enable_rendering=True,
                                      on_rack=FLAGS.on_rack)

  env.reset()
  for _ in tqdm(range(1000)):
    _, _, done, _ = env.step(env.action_space.sample())
    if done:
      break


if __name__ == "__main__":
  app.run(main)
