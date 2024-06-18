import os

base = "/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos"
base_scratch = "/mnt/fast/nobackup/scratch4weeks/ae01116"

# weight = f"{base_scratch}/data/VOC2012/latest.pth"

weight1 = "/mnt/fast/nobackup/scratch4weeks/ae01116/data/VOC2012/save1/latest.pth"
weight2 = "/mnt/fast/nobackup/scratch4weeks/ae01116/data/VOC2012/save2/latest.pth"

model1 = f"{base}/configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10.py"
model2 = f"{base}/configs/voc12/xxscales_input_vpt_seg_zero_vit-b_512x512_20k_12_10.py"

# command = f"{base}/zegenv/bin/python {base}/test.py {base}/configs/voc12/xxscales_input_vpt_seg_zero_vit-b_512x512_20k_12_10.py {weight2} --eval=mIoU"

command = f"{base}/zegenv/bin/python {base}/test.py {model2} {weight1} --eval=mIoU"


os.system(command)
