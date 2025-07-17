import jittor as jt

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows

x = jt.randn(4, 56, 56, 3)
window_size = 7
windows = window_partition(x, window_size)
print(windows.shape)

# Note!!! We need H,W // window_size == 0.It needs to be processed in the data file.