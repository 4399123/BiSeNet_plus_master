
cfg = dict(
    model_type='bisenetv1_efficientnetv2_b1',
    n_cats=9,
    num_aux_heads=2,
    lr_start=0.0003,
    weight_decay=3e-4,
    warmup_iters=2500,
    max_iter=380,
    max_epochs=300,
    dataset='BlueFaceDataset',
    im_root='../../BlueFaceDataX',
    train_im_anns='../../BlueFaceDataX/train.txt',
    val_im_anns='../../BlueFaceDataX/val.txt',
    scales=[0.75, 1.25],
    cropsize=[512, 512],
    ims_per_gpu=18,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res',
)
