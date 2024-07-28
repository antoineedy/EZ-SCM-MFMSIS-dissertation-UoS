#!/usr/bin/env bash

CONFIG=$1
WORK_DIR=$2
PORT=${PORT:-29500}

#/mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python
#/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/zegenv/bin/python

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#CUDA_VISIBLE_DEVICES="0,1,2,3" /mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python -m torch.distributed.launch  --nproc_per_node=4 --master_port=$((RANDOM + 10000)) \
#    $(dirname "$0")/train.py $CONFIG --work-dir=$WORK_DIR --launcher pytorch ${@:3} --seed 9 --deterministic

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="0,1,2,3" /mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python \
    $(dirname "$0")/run_net.py --config-file=$CONFIG --task=train
