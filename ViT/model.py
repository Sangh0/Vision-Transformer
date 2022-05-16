import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True
    ):
        super(PatchEmbed, self).__init__()
        self.img_size = (img_size, img_size)
        self.flatten = flatten
        self.n_patches = (img_size//patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H==self.img_size[0], \
            f"Input image height ({H}) dosen't match model ({self.img_size[0]})."
        assert W==self.img_size[1], \
            f"Input image width ({W}) dosen't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1,2) # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads=8,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.
    ):
        super(Attention, self).__init__()
        assert dim % n_heads == 0, \
            'dim should be devisible by num_heads'
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        n_samples, n_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(
            n_samples, n_tokens, 3, self.n_heads, dim//self.n_heads
        ).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        
        attn = (q@k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn@v).transpose(1,2).reshape(n_samples, n_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(
        self,
        in_dim, 
        mid_dim, 
        out_dim, 
        drop=0.
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.drop2 = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        mlp_ratio=4.,
        qkv_bias=True,
        p=0.,
        attn_p=0.
    ):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_p,
            proj_drop=p,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_dim=dim,
            mid_dim=hidden_features,
            out_dim=dim,
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        n_classes=2,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        p=0.,
        attn_p=0.,
    ):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)
        
        self.encoders = nn.ModuleList([
            TransformerEncoder(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                p=p,
                attn_p=attn_p,
            )
            for _ in range(depth) # 12 blocks
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for encoder in self.encoders:
            x = encoder(x)
        
        x = self.norm(x)
        cls_token_final = x[:,0]
        x = self.head(cls_token_final)
        return x 