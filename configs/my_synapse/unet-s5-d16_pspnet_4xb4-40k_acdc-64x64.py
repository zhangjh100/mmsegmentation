_base_ = [
    '../_base_/models/pspnet_unet_s5-d16.py', '../_base_/datasets/my_synapse.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
# crop_size = (64, 64)
# data_preprocessor = dict(size=crop_size)
# model = dict(
#     data_preprocessor=data_preprocessor,
#     test_cfg=dict(crop_size=(64, 64), stride=(42, 42)))

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=3),
    auxiliary_head=dict(num_classes=3),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
train_dataloader = dict(batch_size=16, num_workers=4)
val_dataloader = dict(batch_size=8, num_workers=4)
test_dataloader = val_dataloader