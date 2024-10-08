
cfg = dict(
    model_type='bisenetv1_mobileone_s3',
    n_cats=9,
    num_aux_heads=2,
    lr_start=0.0003,
    weight_decay=5e-4,
    max_epochs=300,
    dataset='BlueFaceDataset',
    im_root='../../BlueFaceDataX',
    train_im_anns='../../BlueFaceDataX/train.txt',
    val_im_anns='../../BlueFaceDataX/val.txt',
    scales=[0.75, 1.5],
    cropsize=[512, 512],
    ims_per_gpu=44,
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res',
)
