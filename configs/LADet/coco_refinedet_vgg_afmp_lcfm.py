# model settings
model = dict(
    type='SingleStageDetector',
    pretrained='/home/user1/data/pretrained/vgg16_reducedfc.pth',
    backbone=dict(
        type='RefineDet',
        num_classes=81,
        size=512),
    neck=dict(
        type='DFPNv5',
        in_channels=[512, 512, 1024, 512],
        in_scales=[8, 16, 32, 64],
        out_channels=243,
        activation='relu',
        start_level=0,
        add_extra_convs=True,
        num_outs=4,
        return_inputs=True),
    bbox_head=dict(
        type='RefinedetHead',
        num_classes=81,
        feat_channel=243,
        arm_channels=[512, 512, 1024, 512],
        variance=[0.1, 0.2],
        min_dim=512,
        feature_maps=[64, 32, 16, 8],
        min_sizes=[32, 64, 128, 256],
        max_sizes=None,
        steps=[8, 16, 32, 64],
        aspect_ratios=[[2], [2], [2], [2]],
        clip=True,
        head_type='thin'))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    smoothl1_beta=0.11,
    gamma=2.0,
    alpha=0.25,
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.0001,
    nms=dict(type='nms', iou_thr=0.45),
    max_per_img=200)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=12,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=[data_root + 'annotations/instances_train2014.json',
                  data_root + 'annotations/instances_minival2014.json'],
        img_prefix=[data_root + 'train2014/', data_root + 'val2014/'],
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=64,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        test_mode=False,
        extra_aug=dict(
            photo_metric_distortion=dict(
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            expand=dict(
                mean=img_norm_cfg['mean'],
                to_rgb=img_norm_cfg['to_rgb'],
                ratio_range=(1, 4)),
            random_crop=dict(
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3)),
        resize_keep_ratio=False),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_valminusminival2014.json',
        img_prefix=data_root + 'val2014/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=64,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        resize_keep_ratio=False),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_valminusminival2014.json',
        img_prefix=data_root + 'val2014/',
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=64,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True,
        resize_keep_ratio=False))
# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=600,
    warmup_ratio=1.0 / 3,
    step=[300, 350])
checkpoint_config = dict(interval=500)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        #dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 400
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
