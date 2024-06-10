import os

# os.chdir("multi-modal-dissertation-uos")

os.system("ls >> test.txt")

base = "/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos"
base_scratch = "/mnt/fast/nobackup/scratch4weeks/ae01116"

config = f"{base}/configs/voc12/xxscales_input_vpt_seg_zero_vit-b_512x512_20k_12_10.py"

command = f"bash {base}/dist_train.sh {config} {base_scratch}/data/VOC2012"
os.system(command)
