norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
conv_stem_norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='DepthEncoderDecoder',
    pretrained=
    'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth',
    backbone=dict(
        type='DepthFormerSwin',
        pretrain_img_size=224,
        embed_dims=192,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        conv_norm_cfg=dict(type='BN', requires_grad=True),
        depth=50,
        num_stages=0),
    decode_head=dict(
        type='DenseDepthHead',
        in_channels=[64, 192, 384, 768, 1536],
        up_sample_channels=[64, 192, 384, 768, 1536],
        channels=64,
        align_corners=True,
        loss_decode=dict(type='SigLoss', valid_mask=True, loss_weight=1.0),
        act_cfg=dict(type='LeakyReLU', inplace=True),
        min_depth=0,
        max_depth=200),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    neck=dict(
        type='HAHIHeteroNeck',
        positional_encoding=dict(type='SinePositionalEncoding', num_feats=256),
        in_channels=[64, 192, 384, 768, 1536],
        out_channels=[64, 192, 384, 768, 1536],
        embedding_dim=512,
        scales=[1, 1, 1, 1, 1]))
dataset_type = 'nDSMTileDataset'
root = '/nas/k8s/dev/research/yongjin117/PROJECT_nautilus/Datasets/'
data_root = '/nas/k8s/dev/research/yongjin117/PROJECT_nautilus/Datasets/Toy_tile'
img_dir = 'RGB'
dsm_dir = 'DSM'
dtm_dir = 'DTM'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadSATIMGFromFile'),
    dict(type='LoadANDProcessNDSM'),
    dict(
        type='DSM_Transform',
        tr_list=['HFlip', 'VFlip', 'random_crop', 'random_rotate90'],
        is_normalize=False,
        crop_size=512,
        aug_probs=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'depth_gt'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'img_norm_cfg'))
]
test_pipeline = [
    dict(type='LoadSATIMGFromFile'),
    dict(type='LoadANDProcessNDSM'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img', 'depth_gt']),
    dict(
        type='Collect',
        keys=['img', 'depth_gt'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'img_norm_cfg'))
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='nDSMTileDataset',
        data_root=
        '/nas/k8s/dev/research/yongjin117/PROJECT_nautilus/Datasets/Toy_tile',
        pipeline=[
            dict(type='LoadSATIMGFromFile'),
            dict(type='LoadANDProcessNDSM'),
            dict(
                type='DSM_Transform',
                tr_list=['HFlip', 'VFlip', 'random_crop', 'random_rotate90'],
                is_normalize=False,
                crop_size=512,
                aug_probs=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'depth_gt'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor',
                           'img_norm_cfg'))
        ],
        img_dir='RGB',
        dsm_dir='DSM',
        dtm_dir='DTM',
        phase='Training',
        AoIs=['JAX'],
        test_mode=False,
        pose_aligned=True,
        min_depth=0,
        max_depth=200),
    val=dict(
        type='nDSMTileDataset',
        data_root=
        '/nas/k8s/dev/research/yongjin117/PROJECT_nautilus/Datasets/Toy_tile',
        pipeline=[
            dict(type='LoadSATIMGFromFile'),
            dict(type='LoadANDProcessNDSM'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img', 'depth_gt']),
            dict(
                type='Collect',
                keys=['img', 'depth_gt'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor',
                           'img_norm_cfg'))
        ],
        img_dir='RGB',
        dsm_dir='DSM',
        dtm_dir='DTM',
        phase='Testing',
        AoIs=['JAX'],
        test_mode=True,
        pose_aligned=True,
        min_depth=0,
        max_depth=200),
    test=dict(
        type='nDSMTileDataset',
        data_root=
        '/nas/k8s/dev/research/yongjin117/PROJECT_nautilus/Datasets/Toy_tile',
        pipeline=[
            dict(type='LoadSATIMGFromFile'),
            dict(type='LoadANDProcessNDSM'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img', 'depth_gt']),
            dict(
                type='Collect',
                keys=['img', 'depth_gt'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor',
                           'img_norm_cfg'))
        ],
        img_dir='RGB',
        dsm_dir='DSM',
        dtm_dir='DTM',
        phase='Testing',
        AoIs=['JAX'],
        test_mode=True,
        pose_aligned=True,
        min_depth=0,
        max_depth=200))
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
max_lr = 0.0001
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=12800,
    warmup_ratio=0.001,
    min_lr_ratio=1e-08,
    by_epoch=False)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='IterBasedRunner', max_iters=38400)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=1600)
evaluation = dict(
    by_epoch=False,
    start=0,
    interval=100,
    pre_eval=True,
    rule='less',
    save_best='abs_rel',
    greater_keys=('a1', 'a2', 'a3'),
    less_keys=('abs_rel', 'rmse'))
work_dir = 'w_dirs/depthformer/2024_03_06_04_45'
gpu_ids = range(0, 1)
