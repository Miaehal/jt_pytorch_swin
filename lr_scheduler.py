import jittor as jt
from jittor.lr_scheduler import CosineAnnealingLR, StepLR

class WarmupScheduler:
    def __init__(self, optimizer, base_scheduler, warmup_steps, warmup_lr_init, base_lr):
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_lr_init = warmup_lr_init
        self.base_lr = base_lr
        self.step_count = 0
        self.optimizer.lr = self.warmup_lr_init

    def step(self):
        self.step_count += 1
        if self.warmup_steps > 0 and self.step_count <= self.warmup_steps:
            lr = self.warmup_lr_init + (self.base_lr - self.warmup_lr_init) * (self.step_count / self.warmup_steps)
            self.optimizer.lr = lr
        else:
            if self.base_scheduler is not None:
                self.base_scheduler.step()
        
        return self.optimizer.lr

class LinearLRScheduler:
    def __init__(self,
                 optimizer: jt.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t: int = 0,
                 warmup_lr_init: float = 0.):
        
        self.optimizer = optimizer
        self.t_initial = t_initial
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        # base lr
        self.base_lr = self.optimizer.lr
        # Lower limit of the lr
        self.lr_min = self.base_lr * lr_min_rate
        # count
        self.step_count = 0
        self.optimizer.lr = self.warmup_lr_init

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_t:
            new_lr = self.warmup_lr_init + (self.base_lr - self.warmup_lr_init) * (self.step_count / self.warmup_t)
        else:
            t = self.step_count - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            
            if total_t > 0:
                new_lr = self.base_lr - ((self.base_lr - self.lr_min) * (t / total_t))
            else:
                new_lr = self.base_lr
        
        self.optimizer.lr = max(self.lr_min, new_lr)
        return self.optimizer.lr

def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)

    base_lr = optimizer.lr

    lr_scheduler = None

    if config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=config.TRAIN.MIN_LR / config.TRAIN.BASE_LR if config.TRAIN.BASE_LR > 0 else 0.0,
            warmup_t=warmup_steps,
            warmup_lr_init=config.TRAIN.WARMUP_LR
        )
        return lr_scheduler
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=(num_steps - warmup_steps) if config.TRAIN.LR_SCHEDULER.WARMUP_PREFIX else num_steps,
            eta_min=config.TRAIN.MIN_LR
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLR(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE
        )
    else:
        raise ValueError(f"Unsupported LR scheduler: {config.TRAIN.LR_SCHEDULER.NAME}")
    
    lr_scheduler = WarmupScheduler(
        optimizer,
        lr_scheduler,
        warmup_steps=warmup_steps,
        warmup_lr_init=config.TRAIN.WARMUP_LR,
        base_lr=base_lr
    )

    return lr_scheduler