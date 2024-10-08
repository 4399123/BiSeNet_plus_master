
cfg = dict(
    model_type='bisenetv1_efficientnetv2_b3',
    n_cats=9,
    num_aux_heads=2,
    lr_start=0.0002,
    weight_decay=5e-4,
    warmup_iters=2500,
    max_iter=539,
    max_epochs=300,
    dataset='BlueFaceDataset',
    im_root='../../BlueFaceDataX4',
    train_im_anns='../../BlueFaceDataX4/train.txt',
    val_im_anns='../../BlueFaceDataX4/val.txt',
    scales=[0.85, 1.15],
    cropsize=[512, 512],
    eval_crop=[512, 512],
    eval_scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
    ims_per_gpu=16,
    eval_ims_per_gpu=1,
    use_fp16=False,
    use_sync_bn=False,
    respth='./res',
)
