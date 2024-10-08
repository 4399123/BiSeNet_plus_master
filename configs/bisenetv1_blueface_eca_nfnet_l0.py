
cfg = dict(
    model_type='bisenetv1_eca_nfnet_l0',
    n_cats=9,
    num_aux_heads=2,
    lr_start=0.0005,
    weight_decay=5e-4,
    warmup_iters=2500,
    max_iter=180,
    max_epochs=600,
    dataset='BlueFaceDataset',
    im_root='../../BlueFaceDataX1',
    train_im_anns='../../BlueFaceDataX1/train.txt',
    val_im_anns='../../BlueFaceDataX1/val.txt',
    scales=[0.85, 1.15],
    cropsize=[512, 512],
    eval_crop=[512, 512],
    eval_scales=[0.5, 0.75, 1, 1.25, 1.5, 1.75],
    ims_per_gpu=32,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res',
)
