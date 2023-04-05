from logging import error
import torch.nn as nn
from core.layers.bconv2d import BConv2d
from timm.models.layers import trunc_normal_, DropPath

# Ref
# https://discuss.pytorch.org/t/how-can-i-replace-an-intermediate-layer-in-a-pre-trained-network/3586/5


def convert_layers(model, cfg):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if module is not None and len(list(module.children())) > 0 and type(module) != nn.Conv2d:
            model._modules[name], num_converted = convert_layers(module, cfg)

        if type(module) == nn.Conv2d:
            in_channels, out_channels = module.in_channels, module.out_channels
            out_channels = out_channels//cfg.model.width
            if not hasattr(module, 'is_input'):
                in_channels = in_channels//cfg.model.width
            module_new = BConv2d(in_channels, out_channels, module.kernel_size, module.stride,
                                             module.padding, module.dilation, module.groups,
                                             module.bias is not None, cfg, is_input = hasattr(module, 'is_input')) 

            
            module_new.is_conv = True
            if cfg.model.pretrained==1:
                # module_new.fweight  = module.weight
                # module_new.bias     = module.bias
                module_new.fweight.data.copy_(module.weight.data)
                module_new.bias.data.copy_(module.bias.data)
                # module_new.calibrate_masked_weight()
                module_new.set_scale(type='absmax')

            # module_new.loss_type = loss_type
            model._modules[name] = module_new
            
            conversion_count += 1

        if type(module) == nn.BatchNorm2d:
            # print(module)
            num_features = module.num_features
            num_features = num_features//cfg.model.width
            model._modules[name] = nn.BatchNorm2d(num_features) 
            conversion_count += 1
            
        if type(module) == nn.Linear:
            if  module.out_features==1000 or module.out_features==100 or module.out_features==10:
                in_features, out_features = module.in_features, module.out_features
                # print(out_features)
                in_features = in_features//cfg.model.width
                module_new = nn.Linear(in_features, out_features)
            else:
                module_new = BConv2d(module.in_features, module.out_features, [1,1], [1,1],
                                                [0,0], [1,1], 1,
                                                module.bias is not None, cfg, is_input = False) 

                module_new.is_conv = False
                if cfg.model.pretrained==1:
                    module_new.fweight.data.copy_(module.weight[:,:,None,None].data)
                    module_new.bias.data.copy_(module.bias.data)
                    module_new.set_scale(type='absmax')
                    # module_new.calibrate_masked_weight()

                # module_new.name = module.name

            model._modules[name] = module_new
            
            conversion_count += 1
            
        if hasattr(module, "drop_path"):
            module.drop_path = DropPath(cfg.optim.drop_path) if cfg.optim.drop_path > 0. else nn.Identity()

        if type(module) == nn.GELU:
            model._modules[name] = nn.ReLU()
  
            

    return model, conversion_count

if __name__ == '__main__':
    print('__main__')