import os
import numpy as np
import jittor as jt

class RandomErasing:
    def __init__(self, prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        """The same as the default parameters of timm"""
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        if jt.random() > self.prob:
            return img
        for _ in range(10):
            area = img.shape[1] * img.shape[2]
            target_area = jt.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
            target_area *= area

            aspect_ratio = jt.rand() * (self.ratio[1] - self.ratio[0]) + self.ratio[0]
            # Calculate the height and width of the rectangle
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            if w < img.shape[2] and h < img.shape[1]:
                x1 = int(jt.rand() * (img.shape[2] - w))
                y1 = int(jt.rand() * (img.shape[1] - h))
                # Ensure the coordinates are within bounds
                img[:, y1:y1+h, x1:x1+w] = self.value
                return img
        return img

def build_loader(config):
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
        dataset = jt.datasets.ImageFolder(root, transform=transform)
        nb_classes = 2
    else:
        raise NotImplementedError("We only support cats_vs_dogs Now.")
    return dataset, nb_classes

def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # Data augmentation
    if is_train:
        transform_list = [
            jt.transform.RandomResizedCrop(
                config.DATA.IMG_SIZE, 
                interpolation=config.DATA.INTERPOLATION
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
            mode=config.AUG.REMODE,
            max_count=config.AUG.RECOUNT
        )
        transform_list.append(random_erasing)
        if not resize_im:
            # RandomCrop
            transform_list[0] = jt.transform.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return jt.transform.Compose(transform_list)

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(jt.transform.Resize(size, interpolation=config.DATA.INTERPOLATION))
            t.append(jt.transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(jt.transform.Resize(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE,
                    interpolation=config.DATA.INTERPOLATION))

        t.append(jt.transform.ToTensor())
        t.append(jt.transform.ImageNormalize(mean=config.DATA.IMAGENET_DEFAULT_MEAN, std=config.DATA.IMAGENET_DEFAULT_STD))
    return jt.transform.Compose(t)