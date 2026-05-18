from encoder.vit import ViT


class ViTTiny(ViT):
    def __init__(self, with_classifier=False) -> None:
        super().__init__(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            dim=192,
            depth=12,
            heads=3,
            mlp_dim=768,
            dropout=0.1,
            emb_dropout=0.1,
            with_classifier=with_classifier,
        )
