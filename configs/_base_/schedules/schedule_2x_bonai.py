# optimizer for 4 GPUs
optimizer = dict(type='SGD', lr=1.5*1.25*0.02/4, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.00049/1000,
    step=[16, 22, 28])
total_epochs = 32
# fp16 = dict(loss_scale=512.)
