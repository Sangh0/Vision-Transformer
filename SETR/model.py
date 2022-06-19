import torch
import torch.nn as nn

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class Embedding(nn.Module):
    def __init__(
        self,
        img_size=768,
        patch_size=32,
        in_dim=3,
        embed_dim=1024,
        norm_layer=None,
        flatten=True,
    ):
        super(Embedding, self).__init__()
        self.img_size = (img_size, img_size)
        self.flatten = flatten
        self.n_patches = (img_size//patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_dim, embed_dim, kernel_size=patch_size, stride=patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H==self.img_size[0], \
            f"Input image height ({H}) dosen't match model ({self.img_size[0]})."
        assert W==self.img_size[1], \
            f"Input image height ({W}) dosen't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1,2) # BCHW -> BNC
        x = self.norm(x)
        return x
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.,
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
        drop=0.,
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
        attn_p=0,
        drop_path=0.,
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
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_dim=dim,
            mid_dim=hidden_features,
            out_dim=dim,
        )
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=768,
        patch_size=16,
        in_dim=3,
        n_classes=1000,
        embed_dim=1024,
        depth=24,
        n_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        p=0.,
        attn_p=0.,
    ):
        super(VisionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.embedding = Embedding(
            img_size=img_size,
            patch_size=patch_size,
            in_dim=in_dim,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+self.embedding.n_patches, embed_dim))
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
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x):
        n_samples = x.shape[0]
        x = self.embedding(x)
#         cls_token = self.cls_token.expand(n_samples, -1, -1)
#         x = torch.cat((cls_token, x), dim=1)
#         x = x + self.pos_embed
#         x = self.pos_drop(x)
        
        for encoder in self.encoders:
            x = encoder(x)
            
        x = self.norm(x)
#         cls_token_final = x[:,0]
#         x = self.head(cls_token_final)
        return x


######################### DECODER #########################
# 1. Naive Decoder
"""
not Implementation
"""

# 2. PUP Decoder
class PUPDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        n_features=256,
        auxiliary_indices=[9,14,19,23],
    ):
        super(PUPDecoder, self).__init__()
        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_dim, n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.final_out = nn.Conv2d(n_features, out_dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.final_out(x)
        return x

# 3. MLA Decoder
class MLADecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        n_features,
    ):
        super(MLADecoder, self).__init__()
        ########### 1x1 convolution ###########
        self.p2_conv_1x1 = nn.Sequential(
            nn.Conv2d(in_dim, n_features, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(n_features),
            nn.ReLU(inplace=True),
        )
        self.p3_conv_1x1 = nn.Sequential(
            nn.Conv2d(in_dim, n_features, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(n_features),
            nn.ReLU(inplace=True),
        )
        self.p4_conv_1x1 = nn.Sequential(
            nn.Conv2d(in_dim, n_features, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(n_features),
            nn.ReLU(inplace=True),
        )
        self.p5_conv_1x1 = nn.Sequential(
            nn.Conv2d(in_dim, n_features, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(n_features),
            nn.ReLU(inplace=True),
        )
        ########### 3x3 convolution ###########
        self.mla_p2 = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(n_features),
            nn.ReLU(inplace=True),
        )
        self.mla_p3 = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(n_features),
            nn.ReLU(inplace=True),
        )
        self.mla_p4 = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(n_feautres),
            nn.ReLU(inplace=True),
        )
        self.mla_p5 = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1),
            nn.SyncBatchNorm(n_features),
            nn.ReLU(inplace=True),
        )
        
    def to_2d(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(2,1).reshape(n, h, w, c)
        return x
    
    def forward(self, x2, x3, x4, x5):
        x2 = self.to_2d(x2)
        x3 = self.to_2d(x3)
        x4 = self.to_2d(x4)
        x5 = self.to_2d(x5)
        
        x2 = self.p2_conv_1x1(x2)
        x3 = self.p2_conv_1x1(x3)
        x4 = self.p2_conv_1x1(x4)
        x5 = self.p2_conv_1x1(x5)
        
        mla_p4_plus = x4 + x5
        mla_p3_plus = x3 + mla_p4_plus
        mla_p2_plus = x2 + mla_p3_plus
        
        mla_p2 = self.mla_p2(mla_p2_plus)
        mla_p3 = self.mla_p3(mla_p3_plus)
        mla_p4 = self.mla_p4(mla_p4_plus)
        mla_p5 = self.mla_p5(x5)
        
        return mla_p2, mla_p3, mla_p4, mla_p5

class SETR(nn.Module):
    def __init__(
        self,
        encoder,
        n_classes,
        decoder_type='pup',
    ):
        super(SETR, self).__init__()
        assert decoder_type in ('naive', 'pup', 'mla')
        self.encoder = encoder
        self.n_classes = n_classes
        
        if decoder_type=='naive':
            self.decoder = None # NaiveDecoder()
        elif decoder_type=='pup':
            self.decoder = PUPDecoder()
        else:
            self.decoder = MLADecoder()
        
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = x.reshape(
            -1, 
            self.encoder.embed_dim, 
            inputs.size()[2]//self.encoder.patch_size, 
            inputs.size()[3]//self.encoder.patch_size
        )
        x = self.decoder(x)
        return x