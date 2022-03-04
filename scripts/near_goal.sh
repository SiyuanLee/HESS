#!/usr/bin/env bash

# ant maze

# single goal, c=50
python train_hier_sac.py --c 50 --abs_range 20  --env-name AntMaze1Test-v1 --test AntMaze1Test-v1 \
--weight_decay 1e-5 --device cuda:1 --seed 2

# single goal, c=50, image input
python train_hier_sac.py --c 50 --abs_range 20  --env-name AntMaze1Test-v1 --test AntMaze1Test-v1 \
--image True --weight_decay 1e-5 --device cuda:4 --seed 7

################################################################################################################

# Ant FourRoom
# single goal, c=50
python train_hier_sac.py --c 50 --abs_range 20  --env-name AntMazeTest-v2 --test AntMazeTest-v2 \
--weight_decay 1e-5 --device cuda:6 --seed 18

# single goal, c=50, image input
python train_hier_sac.py --c 50 --abs_range 20  --env-name AntMazeTest-v2 --test AntMazeTest-v2 \
--image True --weight_decay 1e-5 --device cuda:6 --seed 13
