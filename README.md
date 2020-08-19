# Motion Imitation

Further development (bug fixes, new features etc) happen in the master branch.
The 'paper' branch of this repository contains the original code accompanying the paper:

"Learning Agile Robotic Locomotion Skills by Imitating Animals",

by Xue Bin Peng et al. It provides a Gym environment for training a simulated quadruped robot to imitate various reference motions, and example training code for learning the policies.

Project page: https://xbpeng.github.io/projects/Robotic_Imitation/index.html

## Getting Started

Install dependencies:

- Install MPI: `sudo apt install libopenmpi-dev`
- Install requirements: `pip3 install -r requirements.txt`

and it should be good to go.

## Training Models

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

## Testing Models

To test a trained model, run the following command

``python3 motion_imitation/run.py --mode test --motion_file motion_imitation/data/motions/dog_pace.txt --model_file motion_imitation/data/policies/dog_pace.zip --visualize``

- `--model_file` specifies the `.zip` file that contains the trained model. Pretrained models are available in `motion_imitation/data/policies/`.


## Data

- `motion_imitation/data/motions/` contains different reference motion clips.
- `motion_imitation/data/policies/` contains pretrained models for the different reference motions.

For more information on the reference motion data format, see the [DeepMimic documentation](https://github.com/xbpeng/DeepMimic)

---

*Disclaimer: This is not an official Google product.*

