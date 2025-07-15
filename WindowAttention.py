import jittor as jt
from jittor import nn
import numpy as np
from scipy.stats import truncnorm

def trunc_normal_jt(var, mean=0., std=1., a=-2., b=2.):
    """jittor version of trunc_normal."""

    low = (a - mean) / std
    high = (b - mean) / std
    np_array = truncnorm.rvs(low, high, loc=mean, scale=std, size=var.shape).astype(np.float32)
    var.assign(jt.array(np_array))
    return var

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = (
            jt.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = jt.arange(self.window_size[0])
        coords_w = jt.arange(self.window_size[1])
        coords = jt.stack(jt.meshgrid([coords_h, coords_w]))
        coords_flatten = jt.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self._relative_position_index = relative_position_index

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_jt(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = jt.matmul(q, k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)

        x = jt.matmul(attn, v).permute(0, 2, 1, 3).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"
    
    def flops(self, N):
        '''calculate flops for 1 window with token length of N'''
        flops = 0
        flops += N * self.dim * 3 * self.dim # qkv = self.qkv(x)
        flops += self.num_heads * N * (self.dim // self.num_heads) * N # attn = jt.matmul(q, k)
        flops += self.num_heads * N * N * (self.dim // self.num_heads) # v = jt.matmul(attn, v)
        flops += N * self.dim * self.dim # x = self.proj(x)
        return flops

if __name__ == "__main__":

    window_attention = WindowAttention(dim=64, window_size=(7, 7), num_heads=8)
    print(window_attention)
    print("Flops for N=49:", window_attention.flops(49))