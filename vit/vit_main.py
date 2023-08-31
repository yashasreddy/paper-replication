import torch
from torch import nn


class ImagePatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 768) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patches = nn.Conv2d(in_channels, 
                                 embed_dim, 
                                 stride=patch_size, 
                                 kernel_size=patch_size, 
                                 padding=0)
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        x_patched = self.patches(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0,2,1)
    
    
class AttentionBlock(nn.Module):
    def __init__(self, embeding_dim: int = 768, attn_heads: int = 12, attn_dropout: float = 0 ) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embeding_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embeding_dim, 
                                          attn_heads=attn_heads, 
                                          attn_dropout=attn_dropout, 
                                          batch_first=True)
        
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.attn(query = x , key=x, value=x, need_weights=False)
        return attn_output
    
    
class MLP(nn.Module):
    def __init__(self, embeding_dim: int = 768, mlp_size: int = 3072, dropout: float = 0.1) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embeding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embeding_dim, 
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embeding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, 
                embeding_dim: int = 768,
                num_heads: int = 12, 
                mlp_size: int = 3072,
                attn_droupot: float = 0,
                mlp_dropout: float = 0.1) -> None:
        super().__init__()

        self.atn_block = AttentionBlock(embeding_dim=embeding_dim, 
                                        attn_heads=num_heads, 
                                        attn_dropout=attn_droupot)
        self.mlp_block = MLP(embeding_dim=embeding_dim, 
                             mlp_size=mlp_size, 
                             dropout=mlp_dropout)
        
    def forward(self, x):
        x = self.atn_block(x) + x
        x = self.mlp_block(x) + x
        return x
    

class ViT(nn.Module):
    def __init__(self,) -> None:
        super().__init__()


if __name__ == "__main__":
    img = torch.randn(size=(3,224,224))
    pe = ImagePatchEmbedding()
    res = pe(img.unsqueeze(0))