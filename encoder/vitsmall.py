from encoder.vit import ViT


class ViTSmall(ViT):
    def __init__(self, with_classifier=False) -> None:
        super().__init__(
            image_size=224,
            patch_size=16,
            num_classes=1000,
            dim=384,
            depth=12,
            heads=6,
            mlp_dim=1536,
            dropout=0.1,
            emb_dropout=0.1,
            with_classifier=with_classifier,
        )
