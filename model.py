import torchvision
import torch.nn as nn

import torchvision.models as M

def get_model(cfg):
    pretrained = cfg.model.pretrained==1
    if cfg.dataset.name in ['CIFAR10', 'cifar10']:
        return create_cifar_model(pretrained=pretrained, num_classes=10)
    elif cfg.dataset.name in ['CIFAR100', 'cifar100']:
        return create_cifar_model(pretrained=pretrained, num_classes=100)
    elif cfg.dataset.name =='IMNET':
        return create_imagenet_model(pretrained=pretrained)
            
def create_cifar_model(pretrained, num_classes):
    if pretrained:
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(512,num_classes)
    else:
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.conv1.is_input = True
    model.maxpool = nn.Identity()
    return model

def create_imagenet_model(pretrained):
    if pretrained==1:
        model=torchvision.models.convnext_base(weights = M.ConvNeXt_Base_Weights.DEFAULT)
    else:
        model=torchvision.models.convnext_base()
    model.features[0][0].is_input = True
    
    return model