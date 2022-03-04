# Efficient Hierarchical Exploration with Stable Subgoal Representation Learning

## Prerequisites

* Python 3.6 or above
* [PyTorch](https://pytorch.org/)
* [Gym](https://gym.openai.com/)
* [Mujoco](https://www.roboti.us/license.html)

## Running Experiments
We provide the scripts for training and evaluation in
`
./scripts/near_goal.sh
`.

The parameter setting can be found in `./arguments`.

#### Training Example

```
python train_hier_sac.py --c 50 --abs_range 20  --env-name AntMaze1Test-v1 --test AntMaze1Test-v1 --weight_decay 1e-5 --device cuda:0 --seed 2
```

#### Evaluating Example

```
python train_hier_sac.py --c 50 --abs_range 20  --test AntMaze1Test-v1 --resume True --eval True --weight_decay 1e-5  --device cuda:0 --seed 124  --animate True
```

You also need to specify the location of saved models in the resume-path argument.
