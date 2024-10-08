
cfg = dict(
    model_type='topformer_fastvit_sa24',   #使用 one 训练
    n_cats=9,
    num_aux_heads=1,
    lr_start=0.0002,
    weight_decay=5e-4,
    max_epochs=300,
    dataset='BlueFaceDataset',
    im_root='../../BlueFaceDataX',
    train_im_anns='../../BlueFaceDataX/train.txt',
    val_im_anns='../../BlueFaceDataX/val.txt',
    scales=[0.85, 1.15],
    cropsize=[512, 512],
    ims_per_gpu=18,
    eval_ims_per_gpu=1,
    use_fp16=False,
    use_sync_bn=True,
    respth='./res',
)
