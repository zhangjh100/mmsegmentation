_base_ = './pspnet_r50-d8_4xb2-40k_la-512x512.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(type='ResNet', depth=101))
