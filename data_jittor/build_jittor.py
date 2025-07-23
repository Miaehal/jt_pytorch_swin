import os
import random
import jittor as jt
from PIL import Image

class RandomErasing:
    def __init__(self, prob=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0):
        """The same as the default parameters of timm"""
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        for _ in range(10):
            area = img.shape[1] * img.shape[2]
            target_area = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
            target_area *= area

            aspect_ratio = random.random() * (self.ratio[1] - self.ratio[0]) + self.ratio[0]
            # Calculate the height and width of the rectangle
            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))
            if w < img.shape[2] and h < img.shape[1]:
                x1 = int(random.randint(0, img.shape[2] - w))
                y1 = int(random.randint(0, img.shape[1] - h))
                # Ensure the coordinates are within bounds
                img[:, y1:y1+h, x1:x1+w] = self.value
                return img
        return img

def build_loader_jittor(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print("successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print("successfully build val dataset")

    data_loader_train = dataset_train.set_attrs(
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        drop_last=True
    )
    data_loader_val = dataset_val.set_attrs(
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        drop_last=False
    )
    return dataset_train, dataset_val, data_loader_train, data_loader_val

def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'cats_vs_dogs':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = jt.dataset.ImageFolder(root, transform=transform)
        num_classes = 2
    else:
        raise NotImplementedError("We only support cats_vs_dogs Now.")
    return dataset, num_classes

def _string_to_interp_mode(mode):
    if mode == 'bicubic':
        return Image.BICUBIC
    elif mode == 'lanczos':
        return Image.LANCZOS
    elif mode == 'hamming':
        return Image.HAMMING
    else:
        return Image.BILINEAR

def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    interp_mode = _string_to_interp_mode(config.DATA.INTERPOLATION)
    if is_train:
        transform_list = [
            jt.transform.RandomResizedCrop(
                config.DATA.IMG_SIZE, 
                interpolation=interp_mode
            ),
            jt.transform.RandomHorizontalFlip(),
            jt.transform.ColorJitter(
                brightness=config.AUG.COLOR_JITTER, 
                contrast=config.AUG.COLOR_JITTER, 
                saturation=config.AUG.COLOR_JITTER
            ),
            jt.transform.ToTensor(),
            jt.transform.ImageNormalize(mean=config.DATA.IMAGENET_DEFAULT_MEAN, std=config.DATA.IMAGENET_DEFAULT_STD)
        ]
        random_erasing = RandomErasing(
            prob=config.AUG.REPROB,
            scale=config.AUG.SCALE,
            ratio=config.AUG.RATIO
        )
        transform_list.append(random_erasing)
        if not resize_im:
            # RandomCrop
            transform_list[0] = jt.nn.ZeroPad2d(padding=4)
            transform_list.insert(1, jt.transform.RandomCrop(config.DATA.IMG_SIZE))
        return jt.transform.Compose(transform_list)

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(jt.transform.Resize(size, interp_mode))
            t.append(jt.transform.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(jt.transform.Resize(config.DATA.IMG_SIZE, interp_mode))

    t.append(jt.transform.ToTensor())
    t.append(jt.transform.ImageNormalize(mean=config.DATA.IMAGENET_DEFAULT_MEAN, std=config.DATA.IMAGENET_DEFAULT_STD))
    return jt.transform.Compose(t)