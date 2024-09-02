import os

base = "/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos"
base_scratch = "/mnt/fast/nobackup/scratch4weeks/ae01116"

# weight = f"{base_scratch}/data/VOC2012/latest.pth"

weight0 = (
    "/mnt/fast/nobackup/scratch4weeks/ae01116/pretrained/voc_inductive_512_vit_base.pth"
)
weight1 = "/mnt/fast/nobackup/scratch4weeks/ae01116/data/VOC2012/save1/latest.pth"
weight2 = "/mnt/fast/nobackup/scratch4weeks/ae01116/data/VOC2012/save2/latest.pth"
weight3 = "/mnt/fast/nobackup/scratch4weeks/ae01116/data/VOC2012/save3/latest.pth"
weight5 = "/mnt/fast/nobackup/scratch4weeks/ae01116/data/VOC2012/save5/latest.pth"
weight_base = "/mnt/fast/nobackup/scratch4weeks/ae01116/data/VOC2012/latest.pth"

unique_weight = "/mnt/fast/nobackup/scratch4weeks/ae01116/data/VOC2012/iter_19450.pth"

new_w = "/mnt/fast/nobackup/scratch4weeks/ae01116/data/VOC2012/iter_19000.pth"

model1 = f"{base}/configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10.py"
model2 = f"{base}/configs/voc12/xxscales_input_vpt_seg_zero_vit-b_512x512_20k_12_10.py"
model3 = f"{base}/configs/voc12/xxscales_output_vpt_seg_zero_vit-b_512x512_20k_12_10.py"
model4 = f"{base}/configs/voc12/inner_vpt_seg_zero_vit-b_512x512_20k_12_10.py"

model7 = f"{base}/configs/voc12/double_inner_vpt_seg_zero_vit-b_512x512_20k_12_10.py"

# command = f"{base}/zegenv/bin/python {base}/test.py {base}/configs/voc12/xxscales_input_vpt_seg_zero_vit-b_512x512_20k_12_10.py {weight2} --eval=mIoU"

last_w = "/mnt/fast/nobackup/scratch4weeks/ae01116/data/VOC2012/iter_10000.pth"

transductive = "/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/configs/voc12/vpt_seg_zero_vit-b_512x512_10k_12_10_st.py"


command = f"/mnt/fast/nobackup/scratch4weeks/ae01116/zegenv/bin/python3.9 {base}/test.py {transductive} {last_w} --eval=mIoU"

os.system(command)
