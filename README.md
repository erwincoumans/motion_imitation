# Motion Imitation

Further development (new features, bug fixes etc) happen in the master branch.
The 'paper' branch of this repository contains the original code accompanying the paper:

"Learning Agile Robotic Locomotion Skills by Imitating Animals",

by Xue Bin Peng et al. It provides a Gym environment for training a simulated quadruped robot to imitate various reference motions, and example training code for learning the policies.

[![Learning Agile Robotic Locomotion Skills by Imitating Animals](https://github.com/erwincoumans/motion_imitation/blob/master/motion_imitation/data/motion_imitation_gif.gif)](https://www.youtube.com/watch?v=lKYh6uuCwRY&feature=youtu.be&hd=1 "Learning Agile Robotic Locomotion Skills by Imitating Animals")

Project page: https://xbpeng.github.io/projects/Robotic_Imitation/index.html

## Getting Started

We use this repository with Python 3.7 or Python 3.8 on Ubuntu, MacOS and Windows.

- Install MPC extension (Optional) `python3 setup.py install --user`

Install dependencies:

- Install MPI: `sudo apt install libopenmpi-dev`
- Install requirements: `pip3 install -r requirements.txt`

and it should be good to go.

## Training Imitation Models

To train a policy, run the following command:

``python3 motion_imitation/run.py --mode train --motion_file motion_imitation/data/motions/dog_pace.txt --int_save_freq 10000000 --visualize``

- `--mode` can be either `train` or `test`.
- `--motion_file` specifies the reference motion that the robot is to imitate. `motion_imitation/data/motions/` contains different reference motion clips.
- `--int_save_freq` specifies the frequency for saving intermediate policies every n policy steps.
- `--visualize` enables visualization, and rendering can be disabled by removing the flag.
- the trained model and logs will be written to `output/`.

For parallel training with MPI run:

``mpiexec -n 8 python3 motion_imitation/run.py --mode train --motion_file motion_imitation/data/motions/dog_pace.txt --int_save_freq 10000000``

- `-n` is the number of parallel.

## Testing Imitation Models

To test a trained model, run the following command

``python3 motion_imitation/run.py --mode test --motion_file motion_imitation/data/motions/dog_pace.txt --model_file motion_imitation/data/policies/dog_pace.zip --visualize``

- `--model_file` specifies the `.zip` file that contains the trained model. Pretrained models are available in `motion_imitation/data/policies/`.


## Motion Capture Data

- `motion_imitation/data/motions/` contains different reference motion clips.
- `motion_imitation/data/policies/` contains pretrained models for the different reference motions.

For more information on the reference motion data format, see the [DeepMimic documentation](https://github.com/xbpeng/DeepMimic)

# Locomotion using Model Predictive Control

[![whole body MPC locomotion for real A1 robot and PyBullet](https://github.com/erwincoumans/motion_imitation/blob/master/motion_imitation/data/mpc_a1.png)](https://www.youtube.com/watch?v=NPvuap-SD78&hd=1 "whole body MPC locomotion for real A1 robot and PyBullet")

## Getting started with MPC and the environment
To start, just clone the codebase, and install the dependencies using
```bash
pip install -r requirements.txt
```

Then, you can explore the environments by running:
```bash
python3 -m motion_imitation.examples.test_env_gui --robot_type=A1 --motor_control_mode=Position --on_rack=True
```

The three commandline flags are:

`robot_type`: choose between `A1` and `Laikago` for different robot.

`motor_control_mode`: choose between `Position` ,`Torque` for different motor control modes.

`on_rack`: whether to fix the robot's base on a rack. Setting `on_rack=True` is handy for debugging visualizing open-loop gaits.

## The gym interface
Additionally, the codebase can be directly installed as a pip package. Just run:
```bash
pip3 install motion_imitation --user
```

Then, you can directly invoke the default gym environment in Python:
```python
import gym
env = gym.make('motion_imitation:A1GymEnv-v1')
```

Note that the pybullet rendering is slightly different from Mujoco. To enable GUI rendering and visualize the training process, you can call:

```python
import gym
env = gym.make('motion_imitation:A1GymEnv-v1', render=True)
```

which will pop up the standard pybullet renderer.

And you can always call env.render(mode='rgb_array') to generate frames.

## Running MPC on the real A1 robot
Since the [SDK](https://github.com/unitreerobotics/unitree_legged_sdk) from Unitree is implemented in C++, we find the optimal way of robot interfacing to be via C++-python interface using pybind11.

### Step 1: Build and Test the robot interface

To start, build the python interface by running the following:
```bash
cd third_party/unitree_legged_sdk
mkdir build
cd build
cmake ..
make
```
Then copy the built `robot_interface.XXX.so` file to the main directory (where you can see this README.md file).

### Step 2: Setup correct permissions for non-sudo user
Since the Unitree SDK requires memory locking and high-priority process, which is not usually granted without sudo, add the following lines to `/etc/security/limits.conf`:

```
<username> soft memlock unlimited
<username> hard memlock unlimited
<username> soft nice eip
<username> hard nice eip
```

You may need to reboot the computer for the above changes to get into effect.

### Step 3: Test robot interface.

Test the python interfacing by running:
'sudo python3 -m motion_imitation.examples.test_robot_interface'

If the previous steps were completed correctly, the script should finish without throwing any errors.

Note that this code does *not* do anything on the actual robot.

## Running the Whole-body MPC controller

To see the whole-body MPC controller in sim, run:
```bash
python3 -m motion_imitation.examples.whole_body_controller_example
```

To see the whole-body MPC controller on the real robot, run:
```bash
sudo python3 -m motion_imitation.examples.whole_body_controller_robot_example
```

### Credits

This repo was developed at Google Robotics and is maintained by one of its members, Erwin Coumans.
The original Motion Imitation code was written by Jason Peng as part of an internship and student researcher at Google Robotics.
Some MPC parts for A1 and running on real A1 are written by Yuxiang Yang, a former resident researcher at Google Robotics.

---

*Disclaimer: This is not an official Google product.*

