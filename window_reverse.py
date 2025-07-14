import jittor as jt

def window_reverse(windows, window_size, H, W):
    B = windows.shape[0] // (H // window_size * W // window_size)
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x

windows = jt.randn(256, 7, 7, 3)
window_size = 7
H, W = 56, 56
windows = window_reverse(windows, window_size, H, W)
print(windows.shape)

# If data satisfied the condition H,W // window_size == 0, it will be processed correctly.(Note in the window_partition.py)