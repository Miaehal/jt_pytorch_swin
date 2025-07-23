from functools import partial
import jittor as jt

def build_optimizer(config, model, is_pretrain=False):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = jt.optim.SGD(parameters, lr=config.TRAIN.BASE_LR, momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                               weight_decay=config.TRAIN.WEIGHT_DECAY, nesterov=True)
    elif opt_lower == 'adamw':
        optimizer = jt.optim.AdamW(parameters, lr=config.TRAIN.BASE_LR, eps=config.TRAIN.OPTIMIZER.EPS,
                              betas=config.TRAIN.OPTIMIZER.BETAS, weight_decay=config.TRAIN.WEIGHT_DECAY)

    return optimizer

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if param.is_stop_grad():
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin