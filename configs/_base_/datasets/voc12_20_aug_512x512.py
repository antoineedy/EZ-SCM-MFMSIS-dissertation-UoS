_base_ = './voc12_20_512x512.py'
# dataset settings
data = dict(
#dataset = dict( # changed by antoine to match rc-clip
    train=dict(
        ann_dir=['SegmentationClass', 'SegmentationClassAug'],
        split=[
            'ImageSets/Segmentation/train.txt',
            'ImageSets/Segmentation/aug.txt'
        ]))