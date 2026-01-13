import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from math import ceil

from torch import einsum

NUM_PATHWAYS = 1280

def exists(val):
    return val is not None


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim) # 128
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout), # 0.1
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x): # x: torch.Size([1, 4427, 128])
        return self.net(self.norm(x))

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=1.0, dropout=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # LoRA components
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.A = nn.Parameter(torch.randn(out_features, r) * 0.02)
            self.B = nn.Parameter(torch.randn(r, in_features) * 0.02)
            self.scaling = alpha / r
        else:
            self.A = self.B = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            # lora = self.dropout(F.linear(x, self.B.T)) #  mat1 and mat2 shapes cannot be multiplied (4226x384 and 8x384)
            lora = self.dropout(F.linear(x, self.B))
            lora = F.linear(lora, self.A)
            return base + self.scaling * lora
        else:
            return base

class DualAttentionLora(nn.Module):
    def __init__(
        self,
        dim = 256,
        dim_head = 64,
        heads = 8,
        residual = True,
        res_conv2d = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        # dropout = 0.,
        num_pathways = 281,
        attn_drop=0.,
        proj_drop=0.,
        r=8,
        alpha=16,
        dropout=0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head # # 8*64=512

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.q = LoRALinear(dim, dim//2, r=r, alpha=alpha, dropout=dropout)
        self.kv = LoRALinear(dim, dim, r=r, alpha=alpha, dropout=dropout)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        # self.proj_drop = nn.Dropout(proj_drop)

        self.residual = residual
        if self.residual:
            self.residual_weight = nn.Parameter(torch.ones(1))  

        self.residual_trans = nn.Linear(dim, dim//2, bias=False) # 256->128

        
        
    def forward(self, x, mask=None, return_attn=False):
        b, n, c, h, m, eps = *x.shape, self.heads, self.num_pathways, self.eps
        x_raw = x # use for residual connection
        q = self.q(x)
        kv = self.kv(x).reshape(b, -1, 2, c//2).permute(2, 0, 1, 3) 
        k, v = kv[0], kv[1]
        q = q.reshape(b, 1, -1, c//2)
        k = k.reshape(b, 1, -1, c//2)
        v = v.reshape(b, 1, -1, c//2)

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :self.num_pathways, :]  # bs x head x num_pathways x dim
        q_histology = q[:, :, self.num_pathways:, :]  # bs x head x num_patches x dim

        k_pathways = k[:, :, :self.num_pathways, :] 
        k_histology = k[:, :, self.num_pathways:, :]

        v_pathways = v[:, :, :self.num_pathways, :]
        v_histology = v[:, :, self.num_pathways:, :]

        attn_histology = (q_histology @ k_pathways.transpose(-2, -1)).softmax(dim=-1)  
        attn_pathways = (q_pathways @ k_histology.transpose(-2, -1)).softmax(dim=-1) 
        out_h = attn_histology @ v_pathways 
        out_p = attn_pathways @ v_histology 


        xh = attn_histology @ out_p 
        xh = xh.permute(0, 2, 1, 3).reshape(b, -1, c//2)

        xp = attn_pathways @ out_h 
        xp = xp.permute(0, 2, 1, 3).reshape(b, -1, c//2) 

        x_cat = torch.cat((xp, xh), dim=-2) 

        if self.residual:
            # x_raw: 256, x: 128
            x_raw = self.residual_trans(x_raw).reshape(b, -1, c//2) 
            x = self.residual_weight * x_raw + x_cat 

        if return_attn:  
            return x, attn_pathways.squeeze().detach().cpu(), attn_histology.squeeze().detach().cpu(), out_p.squeeze().detach().cpu(), out_h.squeeze().detach().cpu(), xp.squeeze().detach().cpu(), xh.squeeze().detach().cpu()

        return x

class DualAttentionLoraV2(nn.Module):
    def __init__(
        self,
        dim = 256,
        dim_head = 64,
        heads = 8,
        residual = True,
        res_conv2d = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        # dropout = 0.,
        num_pathways = 281,
        attn_drop=0.,
        proj_drop=0.,
        r=8,
        alpha=16,
        dropout=0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head 

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.q = LoRALinear(dim, dim//2, r=r, alpha=alpha, dropout=dropout)
        self.kv = nn.Linear(dim, dim, bias=False)# 256->256

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.residual = residual

        if self.residual:
            self.residual_weight = nn.Parameter(torch.ones(1))  
            self.residual_gate = nn.Parameter(torch.zeros(1, 1, dim//2))  # shape: [1, 1, 128]
            self.gate_fc = nn.Sequential(
                                nn.Linear(dim//2, 192),  # or c_cat, depending on what to condition on
                                nn.Sigmoid())

            self.residual_trans = nn.Linear(dim, dim//2, bias=False) # 256->128
            self.residual_proj = nn.Linear(dim, dim//2)  # 384 → 192
   
        
    def forward(self, x, mask=None, return_attn=False):
        b, n, c, h, m, eps = *x.shape, self.heads, self.num_pathways, self.eps
        x_raw = x # use for residual connection
        q = self.q(x)
        kv = self.kv(x).reshape(b, -1, 2, c//2).permute(2, 0, 1, 3) 
        k, v = kv[0], kv[1]
        q = q.reshape(b, 1, -1, c//2)
        k = k.reshape(b, 1, -1, c//2)
        v = v.reshape(b, 1, -1, c//2)

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :self.num_pathways, :]  # bs x head x num_pathways x dim
        q_histology = q[:, :, self.num_pathways:, :]  # bs x head x num_patches x dim

        k_pathways = k[:, :, :self.num_pathways, :] 
        k_histology = k[:, :, self.num_pathways:, :]

        v_pathways = v[:, :, :self.num_pathways, :]
        v_histology = v[:, :, self.num_pathways:, :]

        attn_histology = (q_histology @ k_pathways.transpose(-2, -1)).softmax(dim=-1)  
        attn_pathways = (q_pathways @ k_histology.transpose(-2, -1)).softmax(dim=-1) 
        out_h = attn_histology @ v_pathways 
        out_p = attn_pathways @ v_histology 


        xh = attn_histology @ out_p 
        xh = xh.permute(0, 2, 1, 3).reshape(b, -1, c//2) 

        xp = attn_pathways @ out_h 
        xp = xp.permute(0, 2, 1, 3).reshape(b, -1, c//2) 

        x_cat = torch.cat((xp, xh), dim=-2) 

        if self.residual:
            # x_raw: 256, x: 128
            x_raw_proj = self.residual_proj(x_raw).reshape(b, -1, c//2)   
            gate = self.gate_fc(x_raw_proj.mean(dim=1, keepdim=True))  
            x = gate * x_raw_proj +  (1 - gate) * x_cat 

        else:
            x = x_cat 

        if return_attn:  
            return x, attn_pathways.squeeze().detach().cpu(), attn_histology.squeeze().detach().cpu(), out_p.squeeze().detach().cpu(), out_h.squeeze().detach().cpu(), xp.squeeze().detach().cpu(), xh.squeeze().detach().cpu()

        return x


class DualAttentionV2(nn.Module):
    def __init__(
        self,
        dim = 256,
        dim_head = 64,
        heads = 8,
        residual = True,
        res_conv2d = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        # dropout = 0.,
        num_pathways = 281,
        attn_drop=0.,
        proj_drop=0.,
        dropout=0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head # # 8*64=512

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.q = nn.Linear(dim, dim//2, bias=False)
        self.kv = nn.Linear(dim, dim, bias=False)# 256->256

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)

        self.residual = residual
        if self.residual:
            self.residual_weight = nn.Parameter(torch.ones(1))  
            self.residual_gate = nn.Parameter(torch.zeros(1, 1, dim//2))  
            self.gate_fc = nn.Sequential(
                                nn.Linear(dim//2, 192), 
                                nn.Sigmoid())

            self.residual_trans = nn.Linear(dim, dim//2, bias=False) # 256->128
            self.residual_proj = nn.Linear(dim, dim//2)  # 384 → 192
        
    def forward(self, x, mask=None, return_attn=False):
        b, n, c, h, m, eps = *x.shape, self.heads, self.num_pathways, self.eps

        x_raw = x # use for residual connection
        q = self.q(x)
        kv = self.kv(x).reshape(b, -1, 2, c//2).permute(2, 0, 1, 3) 
        k, v = kv[0], kv[1]

        q = q.reshape(b, 1, -1, c//2)
        k = k.reshape(b, 1, -1, c//2)
        v = v.reshape(b, 1, -1, c//2)

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :self.num_pathways, :]  # bs x head x num_pathways x dim
        q_histology = q[:, :, self.num_pathways:, :]  # bs x head x num_patches x dim

        k_pathways = k[:, :, :self.num_pathways, :] # torch.Size([1, 1, 331, 128])
        k_histology = k[:, :, self.num_pathways:, :]

        v_pathways = v[:, :, :self.num_pathways, :]
        v_histology = v[:, :, self.num_pathways:, :]

        attn_histology = (q_histology @ k_pathways.transpose(-2, -1)).softmax(dim=-1)  # (4096, 331)
        attn_pathways = (q_pathways @ k_histology.transpose(-2, -1)).softmax(dim=-1) # (331, 4096)
        out_h = attn_histology @ v_pathways # (4096, 128)
        out_p = attn_pathways @ v_histology # (331, 128)


        xh = attn_histology @ out_p # torch.Size([1, 8, 4096, 32])
        xh = xh.permute(0, 2, 1, 3).reshape(b, -1, c//2) # # (1, 4096, 128)

        xp = attn_pathways @ out_h # torch.Size([1, 8, 331, 32])
        xp = xp.permute(0, 2, 1, 3).reshape(b, -1, c//2) # # (1, 331, 128)

        x_cat = torch.cat((xp, xh), dim=-2) # torch.Size([1, 4427, 128])

        if self.residual:
            # x_raw: 256, x: 128
            x_raw_proj = self.residual_proj(x_raw).reshape(b, -1, c//2)   #  torch.Size([1, 4118, 192])
            gate = self.gate_fc(x_raw_proj.mean(dim=1, keepdim=True))  # (B, 1, 192)
            x = gate * x_raw_proj +  (1 - gate) * x_cat  # torch.Size([1, 4118, 192])

        else:
            x = x_cat # （384）

        if return_attn:  
            # return three matrices
            return x, attn_pathways.squeeze().detach().cpu(), attn_histology.squeeze().detach().cpu(), out_p.squeeze().detach().cpu(), out_h.squeeze().detach().cpu(), xp.squeeze().detach().cpu(), xh.squeeze().detach().cpu()

        return x




class UniAttentionLoraV2(nn.Module):
    def __init__(
        self,
        dim = 256,
        dim_head = 64,
        heads = 8,
        residual = True,
        res_conv2d = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        # dropout = 0.,
        num_pathways = 281,
        attn_drop=0.,
        proj_drop=0.,
        r=8,
        alpha=16,
        dropout=0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head # # 8*64=512

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.q = LoRALinear(dim, dim//2, r=r, alpha=alpha, dropout=dropout)
        self.kv = nn.Linear(dim, dim, bias=False)# 256->256

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        
        self.residual = residual
        if self.residual:
            self.residual_weight = nn.Parameter(torch.ones(1))  
            self.residual_gate = nn.Parameter(torch.zeros(1, 1, dim//2))  # shape: [1, 1, 128]
            self.gate_fc = nn.Sequential(
                                nn.Linear(dim//2, 192),  # or c_cat, depending on what to condition on
                                nn.Sigmoid())

            self.residual_trans = nn.Linear(dim, dim//2, bias=False) # 256->128
            self.residual_proj = nn.Linear(dim, dim//2)  # 384 → 192

        
    def forward(self, x, mask=None, return_attn=False):
        b, n, c, h, m, eps = *x.shape, self.heads, self.num_pathways, self.eps
        x_raw = x # use for residual connection
        q = self.q(x)
        kv = self.kv(x).reshape(b, -1, 2, c//2).permute(2, 0, 1, 3) 
        q = q.reshape(b, 1, -1, c//2)
        k = k.reshape(b, 1, -1, c//2)
        v = v.reshape(b, 1, -1, c//2)

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        q_pathways = q[:, :, :self.num_pathways, :]  # bs x head x num_pathways x dim
        q_histology = q[:, :, self.num_pathways:, :]  # bs x head x num_patches x dim

        k_pathways = k[:, :, :self.num_pathways, :] # torch.Size([1, 1, 331, 128])
        k_histology = k[:, :, self.num_pathways:, :]

        v_pathways = v[:, :, :self.num_pathways, :]
        v_histology = v[:, :, self.num_pathways:, :]

        attn_histology = (q_histology @ k_pathways.transpose(-2, -1)).softmax(dim=-1)  # (4096, 331)
        attn_pathways = (q_pathways @ k_histology.transpose(-2, -1)).softmax(dim=-1) # (331, 4096)
        out_h = attn_histology @ v_pathways # torch.Size([1, 1, 4096, 192])
        out_p = attn_pathways @ v_histology # torch.Size([1, 1, 36, 192])
        x_cat = torch.cat((out_h, out_p), dim=-2).squeeze(0) # torch.Size([1, 1, 4132, 192])--> torch.Size([1, 4132, 192])

        if self.residual:
            # x_raw: 256, x: 128
            x_raw_proj = self.residual_proj(x_raw).reshape(b, -1, c//2)   #  torch.Size([1, 4118, 192])
            gate = self.gate_fc(x_raw_proj.mean(dim=1, keepdim=True))  # (B, 1, 192)
            x = gate * x_raw_proj +  (1 - gate) * x_cat  # torch.Size([1, 4118, 192])

        else:
            x = x_cat # （384）

        if return_attn:  
            # return three matrices
            return x, attn_pathways.squeeze().detach().cpu(), attn_histology.squeeze().detach().cpu(), out_p.squeeze().detach().cpu(), out_h.squeeze().detach().cpu()
        return x


class DualAttentionLoraLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        residual=True,
        # dropout=0.,
        num_pathways = 281,
        r=8,
        alpha=16,
        dropout=0.1
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.attn = DualAttentionLoraV2(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways,
            r=r,
            alpha=alpha,
            # dropout=dropout
        )

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, attn_histology, out_p, out_h, xp, xh = self.attn(x=self.norm(x), mask=mask, return_attn=True)
            return x, attn_pathways, attn_histology, out_p, out_h, xp, xh
        else:
            x = self.attn(x=self.norm(x), mask=mask)

        return x


class DualAttentionLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        residual=True,
        # dropout=0.,
        num_pathways = 281,
        dropout=0.1
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.attn = DualAttentionV2(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways,
            # dropout=dropout
        )

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, attn_histology, out_p, out_h, xp, xh = self.attn(x=self.norm(x), mask=mask, return_attn=True)
            return x, attn_pathways, attn_histology, out_p, out_h, xp, xh
        else:
            x = self.attn(x=self.norm(x), mask=mask)

        return x




class UniAttentionLoraLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        residual=True,
        # dropout=0.,
        num_pathways = 281,
        r=8,
        alpha=16,
        dropout=0.1
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.attn = UniAttentionLoraV2(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways,
            r=r,
            alpha=alpha,
            # dropout=dropout
        )

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, attn_histology, out_p, out_h, xp, xh = self.attn(x=self.norm(x), mask=mask, return_attn=True)
            return x, attn_pathways, attn_histology, out_p, out_h, xp, xh
        else:
            x = self.attn(x=self.norm(x), mask=mask)

        return x

