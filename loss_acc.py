import jittor as jt
from jittor import nn

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing value should be in the range [0.0, 1.0)."
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def execute(self, x: jt.Var, target: jt.Var) -> jt.Var:
        num_classes = x.shape[-1]
        assert x.shape[0] == target.shape[0], "Batch size of inputs and targets must match."
        log_probs = nn.log_softmax(x, dim=-1)
        zeros = jt.zeros_like(x)
        target_reshaped = target.unsqueeze(1)
        # Create one-hot encoded targets with smoothing
        one_hot_targets = zeros.scatter(1, target_reshaped, jt.array(1.0))
        soft_targets = one_hot_targets * self.confidence + \
                       (1 - one_hot_targets) * (self.smoothing / (num_classes - 1))
        loss = (-soft_targets * log_probs).sum(dim=-1)
        return loss.mean()

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = jt.topk(output, maxk, dim=1, largest=True, sorted=True)
    pred = pred.transpose(1, 0)
    correct = pred.equal(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k * (100.0 / batch_size))
    return res