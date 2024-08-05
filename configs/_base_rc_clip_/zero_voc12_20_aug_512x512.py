_base_ = './zero_voc12_20_512x512.py'
# dataset settings, merge voc12 and voc12aug


# INITIAL

dataset = dict(train=dict(ann_dir='SegmentationClassAug',
                          split='ImageSets/Segmentation/trainaug.txt')) # merge voc12 and voc12aug

# CHANGED BY ANTOINE

# dataset = dict(
#     train=dict(
#         ann_dir=['SegmentationClass', 'SegmentationClassAug'],
#         split=[
#             'ImageSets/Segmentation/train.txt',
#             'ImageSets/Segmentation/aug.txt'
#         ]))
