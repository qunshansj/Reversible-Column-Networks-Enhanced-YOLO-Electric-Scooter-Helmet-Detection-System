python
class Model:
    def __init__(self, pretrained=None):
        self.norm_cfg = dict(type='SyncBN', requires_grad=True)
        self.pretrained = pretrained
        self.backbone = dict(
            type='RevCol',
            channels=[48, 96, 192, 384],
            layers=[3, 3, 9, 3],
            num_subnet=4,
            drop_path=0.2,
            save_memory=False, 
            out_indices=[0, 1, 2, 3],
            init_cfg=dict(type='Pretrained', checkpoint=self.pretrained)
        )
        self.decode_head = dict(
            type='UPerHead',
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=self.norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            )
        )
        self.auxiliary_head = dict(
            type='FCNHead',
            in_channels=384,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=self.norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4
            )
        )
        self.train_cfg = dict()
        self.test_cfg = dict(mode='whole')
        ......
