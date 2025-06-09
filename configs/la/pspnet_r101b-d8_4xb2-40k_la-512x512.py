_base_ = './pspnet_r50-d8_4xb2-40k_rv-256x256.py'
model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(type='ResNet', depth=101))
