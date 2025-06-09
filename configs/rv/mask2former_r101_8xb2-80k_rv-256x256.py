_base_ = ['./mask2former_r50_8xb2-80k_rv-256x256.py']

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
