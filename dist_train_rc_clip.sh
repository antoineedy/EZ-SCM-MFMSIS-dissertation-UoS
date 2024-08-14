#!/usr/bin/env bash

CONFIG=$1
WORK_DIR=$2
PORT=${PORT:-29500}

#/mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python
#/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/zegenv/bin/python

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#CUDA_VISIBLE_DEVICES="0,1,2,3" /mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python -m torch.distributed.launch  --nproc_per_node=4 --master_port=$((RANDOM + 10000)) \
#    $(dirname "$0")/train.py $CONFIG --work-dir=$WORK_DIR --launcher pytorch ${@:3} --seed 9 --deterministic

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#CUDA_VISIBLE_DEVICES="0,1,2,3" /mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python \
#    $(dirname "$0")/run_net.py --config-file=$CONFIG --task=train

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#CUDA_VISIBLE_DEVICES="0,1,2,3" /mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python -m torch.distributed.launch --nproc_per_node=4 --master_port=$((RANDOM + 10000)) \
#    $(dirname "$0")/run_net.py --config-file=$CONFIG --task=train

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#CUDA_VISIBLE_DEVICES="0,1,2,3" mpirun -n 8 python $(dirname "$0")/run_net.py --config-file=project/fcn/fcn_r50-d8_512x1024_cityscapes_80k.py --task=train

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# CUDA_VISIBLE_DEVICES="0" mpirun -n 1 \
#     /mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python \
#     /mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/run_net.py \
#     --config-file=/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/configs/voc12/clip_rc_zero_vit-b_512x512_40k_voc_10_16.py \
#     --task=train \
#     --local_rank 0

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#CUDA_VISIBLE_DEVICES="0,1,2,3" /mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python \
#    -m torch.distributed.launch  --nproc_per_node=4 --master_port=$((RANDOM + 10000)) \
#    /mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/run_net.py \
#    --config-file=/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/configs/voc12/clip_rc_zero_vit-b_512x512_40k_voc_10_16.py \
#    --task=train \

#python tools/run_net.py --config-file=project/fcn/fcn_r50-d8_512x1024_cityscapes_80k.py --task=trai

    # export nvcc_path="/usr/local/cuda/bin/nvcc"

    # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    # CUDA_VISIBLE_DEVICES="0,1,2,3" \
    # /mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python \
    # /mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/run_net.py \
    # --config-file=/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/configs/voc12/clip_rc_zero_vit-b_512x512_40k_voc_10_16.py \
    # --task=train

    # # replace this var with your nvcc location 
    # export nvcc_path="/usr/local/cuda/bin/nvcc" 
    # # run a simple cuda test
    # /mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python \
    # /mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/run_net.py \
    # --config-file=/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/configs/voc12/clip_rc_zero_vit-b_512x512_40k_voc_10_16.py \
    # --task=train

    pip install cupy

    #export nvcc_path="/usr/local/cuda/bin/nvcc"

    #/mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python \
    #-m jittor.test.test_cudnn_op

    #nvidia-smi

    #/mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python \
    #-m jittor_utils.install_cuda

    #/mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python \
    #-m jittor.test.test_example

    #find / -type d -name cuda 2>/dev/null

    #nvcc --version

    #which nvcc


