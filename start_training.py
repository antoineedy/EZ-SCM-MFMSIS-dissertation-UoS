print("Beginning training script")

import os

import mmcv

# from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

base = "/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos"
base_scratch = "/mnt/fast/nobackup/scratch4weeks/ae01116"

config1 = f"{base}/configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10.py"
config2 = f"{base}/configs/voc12/xxscales_input_vpt_seg_zero_vit-b_512x512_20k_12_10.py"
config3 = (
    f"{base}/configs/voc12/xxscales_output_vpt_seg_zero_vit-b_512x512_20k_12_10.py"
)
config4 = f"{base}/configs/voc12/inner_vpt_seg_zero_vit-b_512x512_20k_12_10.py"
config5 = f"{base}/configs/voc12/dilation_vpt_seg_zero_vit-b_512x512_20k_12_10.py"
config6 = f"{base}/configs/voc12/inner_bis_vpt_seg_zero_vit-b_512x512_20k_12_10.py"
config7 = f"{base}/configs/voc12/double_inner_vpt_seg_zero_vit-b_512x512_20k_12_10.py"

config8 = f"{base}/configs/voc12/clip_rc_zero_vit-b_512x512_40k_voc_10_16.py"

# command = f"bash {base}/dist_train.sh {config7} {base_scratch}/data/VOC2012"

command = f"bash {base}/dist_train_rc_clip.sh"

os.system(command)
