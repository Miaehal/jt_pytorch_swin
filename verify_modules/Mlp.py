import jittor as jt
from jittor import nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

if __name__ == "__main__":
    # Example usage
    model = Mlp(in_features=128, hidden_features=256, out_features=64)
    x = jt.randn(10, 128)  # Batch of 10 samples with 128 features each
    output = model(x)
    print(output.shape)  # Should print: (10, 64)
