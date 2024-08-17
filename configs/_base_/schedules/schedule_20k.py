# optimizer
optimizer = dict(type="SGD", lr=0.01, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy="poly", power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type="IterBasedRunner", max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=50, max_keep_ckpts=20)

base_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
novel_class = [15, 16, 17, 18, 19]

evaluation = dict(
    start=19000,
    # start=1,
    interval=50,
    metric="mIoU",
    seen_idx=base_class,
    unseen_idx=novel_class,
)  # interval == 20001
