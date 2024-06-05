import os

# os.chdir("multi-modal-dissertation-uos")

os.system("ls >> test.txt")

command = "bash /user/HS400/ae01116/Documents/multi-modal-dissertation-uos/dist_train.sh configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10.py data/VOC2012"

os.system(command)
