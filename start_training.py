import os

from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    # env_info['MMSegmentation'] = f'{mmseg.__version__}+{get_git_hash()[:7]}'

    return env_info


print("---- Environment info: ----")
for name, val in collect_env().items():
    print("{}: {}".format(name, val))
print("---------------------------")

base = "/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos"
base_scratch = "/mnt/fast/nobackup/scratch4weeks/ae01116"

config = f"{base}/configs/voc12/xxscales_input_vpt_seg_zero_vit-b_512x512_20k_12_10.py"

config = f"{base}/configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10.py"

command = f"bash {base}/dist_train.sh {config} {base_scratch}/data/VOC2012"
os.system(command)
