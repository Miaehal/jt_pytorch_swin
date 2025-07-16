import jittor as jt
from jittor import nn

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def execute(self, x):
        # x: (B, H*W, C)
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, "input resolution must be divisible by 2"

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C)
        x = jt.concat([x0, x1, x2, x3], dim=-1) # (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C)  # (B, H/2*W/2, 4*C)

        x = self.norm(x)  # (B, H/2*W/2, 4*C)
        x = self.reduction(x)  # (B, H/2*W/2, 2*C)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

if __name__ == "__main__":
    
    input_resolution = (4, 4)
    dim = 64
    patch_merging_layer = PatchMerging(input_resolution, dim)
    
    x = jt.randn(1, 16, dim)  # B, H*W, C
    output = patch_merging_layer(x)
    print("Output shape:", output.shape)  # (1, 4, 128)