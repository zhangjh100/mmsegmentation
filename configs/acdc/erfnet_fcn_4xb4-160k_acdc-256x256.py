_base_ = [
    '../_base_/models/erfnet_fcn.py', '../_base_/datasets/acdc.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
train_dataloader = dict(batch_size=16, num_workers=4)
val_dataloader = dict(batch_size=4, num_workers=4)
test_dataloader = val_dataloader
