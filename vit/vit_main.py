import torch
from torch import nn


class ImagePatchEmbedding(nn.Module):
    def __init__(
        self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 768
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patches = nn.Conv2d(
            in_channels, embed_dim, stride=patch_size, kernel_size=patch_size, padding=0
        )
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_patched = self.patches(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)


class AttentionBlock(nn.Module):
    def __init__(
        self, embeding_dim: int = 768, attn_heads: int = 12, attn_dropout: float = 0
    ) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embeding_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embeding_dim,
            attn_heads=attn_heads,
            attn_dropout=attn_dropout,
            batch_first=True,
        )

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.attn(query=x, key=x, value=x, need_weights=False)
        return attn_output


class MLP(nn.Module):
    def __init__(
        self, embeding_dim: int = 768, mlp_size: int = 3072, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embeding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embeding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embeding_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embeding_dim: int = 768,
        num_heads: int = 12,
        mlp_size: int = 3072,
        attn_droupot: float = 0,
        mlp_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.atn_block = AttentionBlock(
            embeding_dim=embeding_dim, attn_heads=num_heads, attn_dropout=attn_droupot
        )
        self.mlp_block = MLP(
            embeding_dim=embeding_dim, mlp_size=mlp_size, dropout=mlp_dropout
        )

    def forward(self, x):
        x = self.atn_block(x) + x
        x = self.mlp_block(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        num_transformer_layers: int = 12,
        embeding_dim: int = 768,
        mlp_size: int = 3072,
        attn_dropout: float = 0,
        mlp_dropout: float = 0.1,
        embed_dropout: float = 0.1,
        num_heads: int = 12,
        num_class: int = 1000,
    ) -> None:
        super().__init__()

        assert (
            img_size % patch_size == 0
        ), f"Input image size must be divisble by patch size, image shape: {img_size}, patch size: {patch_size}"

        self.num_patches = (img_size * img_size) // patch_size**2

        self.class_embed = nn.Parameter(
            data=torch.randn(1, 1, embeding_dim), requires_grad=True
        )

        self.position_embed = nn.Parameter(
            data=torch.randn(1, self.num_patches + 1, embeding_dim), requires_grad=True
        )

        self.embeding_dropout = nn.Dropout(p=embed_dropout)

        self.image_embed = ImagePatchEmbedding(
            in_channels=in_channels, patch_size=patch_size, embed_dim=embeding_dim
        )

        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoder(
                    embeding_dim=embeding_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_size,
                    attn_droupot=attn_dropout,
                    mlp_dropout=mlp_dropout,
                )
            ]
            * num_transformer_layers
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embeding_dim),
            nn.Linear(in_features=embeding_dim, out_features=num_class),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embed.expand(batch_size, -1, -1)

        x = self.image_embed(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embed + x
        x = self.embeding_dropout(x)
        for f in self.transformer_encoder:
            x = f(x)
        x = self.classifier(x[:, 0])

        return x


if __name__ == "__main__":
    img = torch.randn(size=(3, 224, 224))
    pe = ImagePatchEmbedding()
    res = pe(img.unsqueeze(0))
