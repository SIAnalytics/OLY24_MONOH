# dataset settings
dataset_type    = 'nDSMTileDataset'
root            = '/nas/k8s/dev/research/yongjin117/PROJECT_nautilus/Datasets/'
data_root       = root + 'Toy_tile'
img_dir         = 'RGB'
dsm_dir         = 'DSM'
dtm_dir         = 'DTM'

img_norm_cfg = dict(
        mean = [123.675, 116.28, 103.53],
        std  = [58.395, 57.12, 57.375],
        to_rgb = True
    )

crop_size       = (512,512)
train_pipeline  = [
        dict(type='LoadSATIMGFromFile'),
        dict(type='LoadANDProcessNDSM'),
        dict(type='DSM_Transform',
            tr_list         = ['HFlip', 'VFlip', 'random_crop', 'random_rotate90'],
            is_normalize    = False,
            crop_size       = crop_size[0],
            aug_probs       = 0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect',
             keys           = ['img', 'depth_gt'],
             meta_keys      = ('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg')),
    ]
test_pipeline   = [
        dict(type='LoadSATIMGFromFile'),
        dict(type='LoadANDProcessNDSM'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img', 'depth_gt']),
        dict(type='Collect',
             keys           = ['img','depth_gt'],
             meta_keys      = ('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg')),
    ]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type        = dataset_type,
        data_root   = data_root,
        pipeline    = train_pipeline,
        img_dir     = 'RGB',
        dsm_dir     = 'DSM',
        dtm_dir     = 'DTM',
        phase       = 'Training',
        AoIs        = ['JAX'],
        test_mode   = True,
        pose_aligned= False,
        min_depth   = 0,
        max_depth   = 200),
    val=dict(
        type        = dataset_type,
        data_root   = data_root,
        pipeline    = test_pipeline,
        img_dir     = 'RGB',
        dsm_dir     = 'DSM',
        dtm_dir     = 'DTM',
        phase       = 'Testing',
        AoIs        = ['JAX'],
        test_mode   = True,
        pose_aligned= True,
        min_depth   = 0,
        max_depth   = 200),
    test=dict(
        type        = dataset_type,
        data_root   = data_root,
        pipeline    = test_pipeline,
        img_dir     = 'RGB',
        dsm_dir     = 'DSM',
        dtm_dir     = 'DTM',
        phase       = 'Testing',
        AoIs        = ['JAX'],
        test_mode   = True,
        pose_aligned= True,
        min_depth   = 0,
        max_depth   = 200))
