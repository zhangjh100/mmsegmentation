_base_ = './danet_r50-d8_4xb2-40k_acdc-256x256.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
