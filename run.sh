# !/bin/bash

# # env 
# conda install -c "nvidia/label/cuda-12.1.1" cuda-nvcc -y
# pip install flash-attn --no-build-isolation

set -x
cd /home/sky/.docker/git/aifs.deploy
# PY=/home/sky/miniconda3/envs/AI_model/bin/python
# PY=/home/sky/.conda/envs/aifs/bin/python
PY=/home/sky/.conda/envs/aifs.py.3.12/bin/python

export CUDA_VISIBLE_DEVICES=7
$PY main.py