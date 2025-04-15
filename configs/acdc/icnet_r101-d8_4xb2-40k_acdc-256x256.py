_base_ = './icnet_r50-d8_4xb2-40k_acdc-256x256.py'
model = dict(backbone=dict(backbone_cfg=dict(depth=101)))
