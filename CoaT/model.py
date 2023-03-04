from typing import *
from timm.models.layers import PatchEmbed, DropPath, Mlp, trunc_normal_

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPE(nn.Module):
    """
    Convolutional Position Encoding
    """
    def __init__(self, dim: int, kernel_size: int=3):
        super(CPE, self).__init__()
        self.proj = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, 
            stride=1, padding=kernel_size//2, groups=dim,
        )

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == 1 + H * W

        # seprate class token and image tokens from inputs
        cls_token, img_tokens = x[:, :1], x[:, 1:]

        # depthwise convolution
        feat = img_tokens.transpose(1, 2).view(B, C, H, W)
        x = self.proj(feat) + feat
        x = x.flatten(2).transpose(1, 2)

        # combine with class token
        x = torch.cat((cls_token, x), dim=1)

        return x


class CRPE(nn.Module):
    """
    Convolutional Relative Position Encoding
    """
    def __init__(self, Ch: int, h: int, window: Dict[int]):
        super(CRPE, self).__init__()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(cur_head_split * channels, cur_head_split * channels,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * channels,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size: Tuple[int, int]):
        B, h, N, Ch = q.shape
        H, W = size
        assert N == 1 + H * W

        q_img = q[:, :, 1:, :] # B x h x H*W x Ch
        v_img = v[:, :, 1:, :] # B x h x H*W x Ch

        v_img = v_img.transpose(-1, -2).reshape(B, h * Ch, H, W)
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        conv_v_img_list = []
        for i, conv in enumerate(self.conv_list):
            conv_v_img_list.append(conv(v_img_list[i]))
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = conv_v_img.reshape(B, h, Ch, H * W).transpose(-1, -2)

        EV_hat = q_img * conv_v_img
        EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0)) # B x h x N x Ch
        return EV_hat


class FactorizedAttention(nn.Module):
    
    def __init__(
        self,
        dim: int,
        num_head: int=8,
        qkv_bias: bool=False,
        attn_drop: float=0.,
        proj_drop: float=0.,
        shared_crpe: Optional[nn.Module]=None,
    ):
        super(FactorizedAttention, self).__init__()
        self.num_head = num_head
        head_dim = dim // num_head
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_softmax = k.softmax(dim=2)
        factor_att = k_softmax.transpose(-1, -2) @ v
        factor_att = q @ factor_att

        crpe = self.crpe(q, v, size=size)

        x = self.scale * factor_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SerialBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int=4,
        qkv_bias: bool=False,
        drop: float=0.,
        attn_drop: float=0.,
        drop_path: float=0.,
        act_layer: nn.Module=nn.GELU,
        norm_layer: nn.Module=nn.LayerNorm,
        shared_cpe: Optional[nn.Module]=None,
        shared_crpe: Optional[nn.Module]=None,
    ):
        super(SerialBlock, self).__init__()
        
        self.cpe = shared_cpe

        self.norm1 = norm_layer(dim)
        self.factor_attn = FactorizedAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, 
            proj_drop=proj_drop, shared_crpe=shared_crpe
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
        )

    def forward(self, x, size: Tuple[int, int]):
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.factor_attn(cur, size)
        x = x + self.drop_path(cur)

        cur = self.norm2(x)
        cur = self.mlp(x)
        x = x + self.drop_path(cur)
        return x


class ParallelBlock(nn.Module):
    
    def __init__(
        self,
        dims: List,
        num_heads: int,
        mlp_ratio: List=[],
        qkv_bias: bool=False,
        drop: float=0.,
        attn_drop: float=0.,
        drop_path: float=0.,
        act_layer: nn.Module=nn.GELU,
        norm_layer: nn.Module=nn.LayerNorm,
        shared_crpes: Optional[nn.Module]=None,
    ):
        super(ParallelBlock, self).__init__()

        self.norm12 = norm_layer(dims[1])
        self.norm13 = norm_layer(dims[2])
        self.norm14 = norm_layer(dims[3])

        self.factor_attn2 = FactorizedAttention(
            dim=dims[1], num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            proj_drop=proj_drop, shared_crpe=shared_crpes[1],
        )
        self.factor_attn3 = FactorizedAttention(
            dim=dims[2], num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            proj_drop=proj_drop, shared_crpe=shared_crpes[2],
        )
        self.factor_attn4 = FactorizedAttention(
            dim=dims[3], num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            proj_drop=proj_drop, shared_crpe=shared_crpes[3],
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm22 = norm_layer(dims[1])
        self.norm23 = norm_layer(dims[2])
        self.norm24 = norm_layer(dims[3])

        assert dims[1] == dims[2] == dims[3]
        assert mlp_ratio[1] == mlp_ratio[2] == mlp_ratio[3]
        mlp_hidden_dim = int(dims[1] * mlp_ratio[1])
        self.mlp2 = self.mlp3 = self.mlp4 = Mlp(
            in_features=dims[1], hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
        )

    def interpolate(self, x, scale_factor: float, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == 1 + H * W

        cls_token = x[:, :1, :]
        img_tokens = x[:, 1:, :]

        img_tokens = img_tokens.transpose(1, 2).reshape(B, C, H, W)
        img_tokens = F.interpolate(
            img_tokens, scale_factor=scale_factor, recompute_scale_factor=False, mode='bilinear', align_corners=False,
        )
        img_tokens = img_tokens.reshape(B, C, -1).transpose(1, 2)

        out = torch.cat((cls_token, img_tokens), dim=1)
        return out

    def upsample(self, x, factor: float, size: Tuple[int, int]):
        return self.interpolate(x, scale_factor=factor, size=size)
    
    def downsample(self, x, factor: float, size: Tuple[int, int]):
        return self.interpolate(x, scale_factor=1./factor, size=size)

    def forward(self, x1, x2, x3, x4, sizes: List[Tuple[int, int]]):
        _, S2, S3, S4 = sizes
        cur2 = self.norm12(x2)
        cur3 = self.norm13(x3)
        cur4 = self.norm14(x4)

        cur2 = self.factor_attn2(cur2, size=S2)
        cur3 = self.factor_attn3(cur3, size=S3)
        cur4 = self.factor_attn4(cur4, size=S4)

        upsample3_2 = self.upsample(cur3, factor=2., size=S3)
        upsample4_3 = self.upsample(cur4, factor=2., size=S4)
        upsample4_2 = self.upsample(cur4, factor=4., size=S4)

        downsample2_3 = self.downsample(cur2, factor=2., size=S2)
        downsample3_4 = self.downsample(cur3, factor=2., size=S3)
        downsmaple2_4 = self.downsample(cur2, factor=4., size=S2)

        cur2 = cur2 + upsample3_2 + upsample4_2
        cur3 = cur3 + upsample4_3 + downsample2_3
        cur4 = cur4 + downsample3_4 + downsmaple2_4

        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)

        cur2 = self.norm22(x2)
        cur3 = self.norm23(x3)
        cur4 = self.norm24(x4)

        cur2 = self.mlp2(cur2)
        cur3 = self.mlp3(cur3)
        cur4 = self.mlp4(cur4)

        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)

        return x1, x2, x3, x4


class CoaT(nn.Module):

    def __init__(
        self,
        img_size: int=224,
        patch_size: int=16,
        in_dim: int=3,
        num_classes: int=1000,
        embed_dims: Tuple[int]=(0,0,0,0),
        serial_depths: Tuple[int]=(0,0,0,0),
        parallel_depth: int=0,
        num_heads: int=0,
        mlp_ratios: Tuple[int]=(0,0,0,0),
        qkv_bias: bool=True,
        drop_rate: float=0.,
        attn_drop_rate: float=0.,
        drop_path_rate: float=0.,
        norm_layer: nn.Module=nn.LayerNorm,
        return_interm_layers: bool=False,
        out_dim: Optional[str]=None,
        crpe_window: Optional[dict]=None,
        global_pool: str='token',
    ):
        super(CoaT, self).__init__()
        assert global_pool in ('token', 'avg')
        crpe_window = crpe_window or {3: 2, 5: 3, 7: 3}
        self.return_interm_layers = return_interm_layers
        self.out_dim = out_dim
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]
        self.num_classes = num_classes
        self.global_pool = global_pool

        # patch embeddings
        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_dim,
            embed_dim=embed_dims[0], norm_layer=nn.LayerNorm)
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=patch_size, in_chans=in_dim,
            embed_dim=embed_dims[1], norm_layer=nn.LayerNorm)
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=patch_size, in_chans=in_dim,
            embed_dim=embed_dims[2], norm_layer=nn.LayerNorm)
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=patch_size, in_chans=in_dim,
            embed_dim=embed_dims[3], norm_layer=nn.LayerNorm)
        
        # class tokens
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.cls_token3 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.cls_token4 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # convolutional position encodings
        self.cpe1 = CPE(dim=embed_dims[0], kernel_size=3)
        self.cpe2 = CPE(dim=embed_dims[1], kernel_size=3)
        self.cpe3 = CPE(dim=embed_dims[2], kernel_size=3)
        self.cpe4 = CPE(dim=embed_dims[3], kernel_size=3)

        # convolutional relative position encodings
        self.crpe1 = CRPE(Ch=embed_dims[0] // num_heads, h=num_heads, window=crpe_window)
        self.crpe2 = CRPE(Ch=embed_dims[1] // num_heads, h=num_heads, window=crpe_window)
        self.crpe3 = CRPE(Ch=embed_dims[2] // num_heads, h=num_heads, window=crpe_window)
        self.crpe4 = CRPE(Ch=embed_dims[3] // num_heads, h=num_heads, window=crpe_window)
        
        # serial blocks 1
        self.serial_blocks1 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[0], num_heads=num_heads, mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
                shared_cpe=self.cpe1, shared_crpe=self.crpe1,
            )
            for _ in range(serial_depths[0])
        ])

        # serial blocks 2
        self.serial_blocks2 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
                shared_cpe=self.cpe2, shared_crpe=self.crpe2,
            )
            for _ in range(serial_depths[1])
        ])
        
        # serial blocks 3
        self.serial_blocks3 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[2], num_heads=num_heads, mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
                shared_cpe=self.cpe3, shared_crpe=self.crpe3,
            )
            for _ in range(serial_depths[2])
        ])
        
        # serial blocks 4
        self.serial_blocks4 = nn.ModuleList([
            SerialBlock(
                dim=embed_dims[3], num_heads=num_heads, mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
                shared_cpe=self.cpe4, shared_crpe=self.crpe4,
            )
            for _ in range(serial_depths[3])
        ])
        
        # parallel blocks
        self.parallel_depth = parallel_depth
        if self.parallel_depth > 0:
            self.parallel_blocks = nn.ModuleList([
                ParallelBlock(
                    dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
                    shared_crpes=(self.crpe1, self.crpe2, self.crpe3, self.crpe4),
                )
                for _ in range(parallel_depth
                )
            ])
        else:
            self.parallel_blocks = 
            
        # classification head
        if not self.return_interm_layers:
            if self.parallel_blocks is not None:
                self.norm2 = norm_layer(embed_dims[1])
                self.norm3 = norm_layer(embed_dims[2])
            else:
                self.norm2 = self.norm3 = None
            self.norm4 = norm_layer(embed_dims[3])

            if self.parallel_depth > 0: # CoaT
                assert embed_dims[1] == embed_dims[2] == embed_dims[3]
                self.aggregate = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
            else: # CoaT Lite
                self.aggregate = None
            
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    trunc_normal_(self.cls_token1, std=0.02)
    trunc_normal_(self.cls_token2, std=0.02)
    trunc_normal_(self.cls_token3, std=0.02)
    trunc_normal_(self.cls_token4, std=0.02)
    self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif: isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x0):
        B = x0.shape[0]

        # serial blocks 1
        x1 = self.patch_embed1(x0)
        H1, W1 = self.patch_embed1.grid_size
        x1 = insert_cls(x1, self.cls_token1)
        for block in self.serial_blocks1:
            x1 = block(x1, size=(H1, W1))
        x1_ncols = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        # serial blocks 2
        x2 = self.patch_embed2(x1_ncols)
        H2, W2 = self.patch_embed2.grid_size
        x2 = insert_cls(x2, self.cls_token2)
        for block in self.serial_blocks2:
            x1 = block(x2, size=(H2, W2))
        x2_ncols = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

        # serial blocks 3
        x3 = self.patch_embed3(x2_ncols)
        H3, W3 = self.patch_embed3.grid_size
        x3 = insert_cls(x3, self.cls_token3)
        for block in self.serial_blocks3:
            x3 = block(x3, size=(H3, W3))
        x3_ncols = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()

        # serial blocks 4
        x4 = self.patch_embed4(x3_ncols)
        H4, W4 = self.patch_embed4.grid_size
        x4 = insert_cls(x4, self.cls_token4)
        for block in self.serial_blocks4:
            x4 = block(x4, size=(H4, W4))
        x4_ncols = remove_cls(x4).reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()

        if self.parallel_blocks is None:
            if self.return_interm_layers: # Return intermediate features for down-stream tasks
                feat_out = {}
                if 'x1_ncols' in self.out_dim:
                    feat_out['x1_ncols'] = x1_ncols
                if 'x2_ncols' in self.out_dim:
                    feat_out['x2_ncols'] = x2_ncols
                if 'x3_ncols' in self.out_dim:
                    feat_out['x3_ncols'] = x3_ncols
                if 'x4_ncols' in self.out_dim:
                    feat_out['x4_ncols'] = x4_ncols
                return feat_out
            else:
                x4 = self.norm4(x4)
                return x4

        # parallel blocks
        for block in self.parallel_blocks:
            x2, x3, x4 = self.cpe2(x2, (H2, W2)), self.cpe3(x3, (H3, W3)), self.cpe4(x4, (H4, W4))
            x1, x2, x3, x4 = block(x1, x2, x3, x4, sizes=[(H1, W1), (H2, W2), (H3, W3), (H4, W4)])

        if self.return_interm_layers:
            feat_out = {}
            if 'x1_ncols' in self.out_dim:
                x1_ncols = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x1_ncols'] = x1_ncols
            if 'x2_ncols' in self.out_dim:
                x2_ncols = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x2_ncols'] = x2_ncols
            if 'x3_ncols' in self.out_dim:
                x3_ncols = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x3_ncols'] = x3_ncols
            if 'x4_ncols' in self.out_dim:
                x4_ncols = remove_cls(x4).reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
                feat_out['x4_ncols'] = x4_ncols
            return feat_out
        else:
            x2 = self.norm2(x2)
            x3 = self.norm3(x3)
            x4 = self.norm4(x4)
            return [x2, x3, x4]

    def forward_head(self, x_feat: Union[torch.Tensor, List[torch.Tensor]], pre_logits: bool=False):
        if isinstance(x_feat, list):
            assert self.aggregate is not None
            if self.global_pool == 'avg':
                x = torch.cat([xl[:, 1:].mean(dim=1, keepdim=True) for xl in x_feat], dim=1)
            else:
                x = torch.stack([xl[:, 0] for xl in x_feat], dim=1)
            x = self.aggregate(x).squeeze(dim=1)
        else:
            x = x_feat[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x_feat[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        if self.return_interm_layers:
            return self.forward_features(x)
        else:
            x_feat = self.forward_features(x)
            x = self.forward_head(x_feat)
            return x


def insert_cls(x, cls_token):
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    return x


def remove_cls(x):
    return x[:, 1:, :]


def coat_tiny(num_classes):
    return CoaT(
        patch_size=4, 
        embed_dims=[152, 152, 152, 152], 
        serial_depths=[2, 2, 2, 2],
        parallel_depth=6,
        num_heads=8,
        mlp_ratios=[4, 4, 4, 4],
        num_classes=num_classes,
    )


def coat_mini(num_classes):
    return CoaT(
        patch_size=4,
        embed_dims=[152, 216, 216, 216],
        serial_depths=[2, 2, 2, 2],
        parallel_depth=6,
        num_heads=8,
        mlp_ratios=[4, 4, 4, 4],
        num_classes=num_classes,
    )


def coat_small(num_classes):
    return CoaT(
        patch_size=4,
        embed_dims=[152, 320, 320, 320],
        serial_depths=[2, 2, 2, 2],
        parallel_depth=6,
        num_heads=8,
        mlp_ratios=[4, 4, 4, 4],
        num_classes=num_classes,
    )


def coat_lite_tiny(num_classes):
    return CoaT(
        patch_size=4,
        embed_dims=[64, 128, 256, 320],
        serial_depths=[2, 2, 2, 2],
        parallel_depth=0,
        num_heads=8,
        mlp_ratios=[8, 8, 4, 4],
        num_classes=num_classes,
    )


def coat_lite_mini(num_classes):
    return CoaT(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        serial_depths=[2, 2, 2, 2],
        parallel_depth=0,
        num_heads=8,
        mlp_ratios=[8, 8, 4, 4],
        num_classes=num_classes,
    )


def coat_lite_small(num_classes):
    return CoaT(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        serial_depths=[3, 4, 6, 3],
        parallel_depth=0,
        num_heads=8,
        mlp_ratios=[8, 8, 4, 4],
        num_classes=num_classes,
    )


def coat_lite_medium(num_classes):
    return CoaT(
        patch_size=4,
        embed_dims=[128, 256, 320, 512],
        serial_depths=[3, 6, 10, 8],
        parallel_depth=0,
        num_heads=8,
        mlp_ratios=[4, 4, 4, 4],
        num_classes=num_classes,
    )