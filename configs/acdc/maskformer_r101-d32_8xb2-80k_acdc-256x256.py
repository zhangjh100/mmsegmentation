_base_ = './maskformer_r50-d32_8xb2-80k_acdc-256x256.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
