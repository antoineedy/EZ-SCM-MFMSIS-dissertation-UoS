import os

base= "/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos"
base_scratch = "/mnt/fast/nobackup/scratch4weeks/ae01116"

weight = f"{base_scratch}/data/VOC2012/latest.pth"

command = f"{base}/zegenv/bin/python {base}/test.py {base}/configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10.py {weight} --eval=mIoU"

os.system(command)
