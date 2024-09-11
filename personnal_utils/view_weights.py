import torch

PATH = "/mnt/fast/nobackup/scratch4weeks/ae01116/data/VOC2012/iter_20000.pth"
model = torch.load(PATH)
# open file weights.txt and write the weights to it

l = [
    "decode_head.q6_proj",
    "decode_head.q8_proj",
    "decode_head.q12_proj",
    "decode_head.cls_proj_6",
    "decode_head.cls_proj_8",
    "decode_head.cls_proj_12",
    "decode_head.layer6_proj",
    "decode_head.layer8_proj",
    "decode_head.layer12_proj",
    "decode_head.text_proj_6",
    "decode_head.text_proj_8",
    "decode_head.text_proj_12",
]

path_all = "/mnt/fast/nobackup/users/ae01116/multi-modal-dissertation-uos/weights/"

for k in l:
    w = k + ".weight"
    b = k + ".bias"
    path_w = path_all + w
    path_b = path_all + b
    torch.save(model["state_dict"][w], path_w)
    torch.save(model["state_dict"][b], path_b)
