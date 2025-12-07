#!/bin/bash
# conda_wrapper.sh
source /home/svdighe/miniconda3/etc/profile.d/conda.sh
conda activate vit_env

python code/train_vit_cifar100.py
