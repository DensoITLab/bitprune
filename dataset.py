import torch
import os
from torchvision import datasets, transforms
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
)

train_transforms_cifar = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms_cifar = transforms.Compose(
    [
        transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

def build_dataset(is_train, cfg):
    if cfg.dataset.name == 'CIFAR100':
        transform = build_transform(is_train, cfg)
        dataset = datasets.CIFAR100(cfg.dataset.path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif cfg.dataset.name == 'cifar100':
        if is_train:
            transform =  train_transforms_cifar
        else:
            transform =  test_transforms_cifar
        dataset = datasets.CIFAR100(cfg.dataset.path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif cfg.dataset.name == 'CIFAR10':
        transform = build_transform(is_train, cfg)
        dataset = datasets.CIFAR10(cfg.dataset.path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif cfg.dataset.name == 'cifar10':
        if is_train:
            transform =  train_transforms_cifar
        else:
            transform =  test_transforms_cifar
        dataset = datasets.CIFAR10(cfg.dataset.path, train=is_train, transform=transform, download=True)
        nb_classes = 10    
    elif cfg.dataset.name == 'IMNET':
        print("reading from datapath", cfg.dataset.path)
        transform = build_transform(is_train, cfg)
        root = os.path.join(cfg.dataset.path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif cfg.dataset.name == "image_folder":
        root = cfg.dataset.path if is_train else cfg.dataset.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = cfg.dataset.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
   
    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes

# adopted from https://github.com/facebookresearch/ConvNeXt/blob/33440594b4221b713d493ce11f33b939c4afd696/datasets.py
def build_transform(is_train, cfg):
    imagenet_default_mean_and_std = cfg.dataset.imagenet_default_mean_and_std
    resize_im = cfg.dataset.input_size > 32
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=cfg.dataset.input_size,
            is_training=True,
            color_jitter=cfg.dataset.color_jitter,
            auto_augment=cfg.dataset.aa,
            interpolation=cfg.dataset.train_interpolation,
            re_prob=cfg.dataset.reprob,
            re_mode=cfg.dataset.remode,
            re_count=cfg.dataset.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                cfg.dataset.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if cfg.dataset.input_size >= 384:  
            t.append(
            transforms.Resize((cfg.dataset.input_size, cfg.dataset.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {cfg.dataset.input_size} size input images...")
        else:
            if cfg.dataset.crop_pct<=0:
                cfg.dataset.crop_pct = 224 / 256
            size = int(cfg.dataset.input_size / cfg.dataset.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(cfg.dataset.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def getLoader(cfg):
    dataset_train, nb_classes  = build_dataset(True, cfg)
    dataset_val, nb_classes  = build_dataset(False, cfg)
    cfg.dataset.input_shape = [1, 3, cfg.dataset.input_size, cfg.dataset.input_size]
   
    num_tasks = get_world_size()
    global_rank = get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=cfg.misc.seed,
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=cfg.optim.batch_size,
        num_workers=cfg.hardware.num_cpu_workers,
        pin_memory=cfg.dataset.pin_mem,
        drop_last=True,
        )
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=cfg.optim.batch_size,
            num_workers=cfg.hardware.num_cpu_workers,
            pin_memory=cfg.dataset.pin_mem,
            drop_last=False
        )
    return data_loader_train, data_loader_val
