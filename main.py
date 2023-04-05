import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import tqdm
import torch.nn as nn
from logger import STATS
from core.bnet import BNet
from model import get_model
import argconfig
import dataset
import timm

def extract_batch(batch):
    x, y = batch
    x, y = x.to('cuda'), y.to('cuda')
    return x, y
    
def main():
    cfg = argconfig.load()
    train_loader, val_loader = dataset.getLoader(cfg)
    cfg.freeze()
    
    model   = get_model(cfg)
    net     = BNet(model, cfg)
    
    # Setup Optimizer
    wd0_para = []
    wd1_para = []
    for name, value in net.model.named_parameters():
        if "scale" in name:
            wd0_para += [value]
        elif 'alpha' in name:
            wd0_para += [value]
        elif "fweight" in name:
            if cfg.optim.enable_decay:
                wd1_para += [value]
            else:
                wd0_para += [value]
        else:
            wd1_para += [value]

    if cfg.optim.optimizer=='SGD':
        optimizer = torch.optim.SGD(wd1_para,lr=cfg.optim.lr_core, momentum=cfg.optim.momentum, weight_decay=cfg.optim.weight_decay)
    elif cfg.optim.optimizer=='AdamW':
        optimizer = optim.AdamW(wd1_para, lr=cfg.optim.lr_core, weight_decay=cfg.optim.weight_decay)

    optimizer.add_param_group({"params": wd0_para, 'lr':cfg.optim.lr_mask,'weight_decay':0.0}) #   BASE_LR: 0.00004, ADAMW  0.001

    if cfg.optim.scheduler=='OneCycleLR':
        steps_per_epoch = len(train_loader)
        scheduler = lr_scheduler.OneCycleLR(optimizer,[cfg.optim.lr_core, cfg.optim.lr_mask], epochs=cfg.optim.epochs, steps_per_epoch=steps_per_epoch)
    elif cfg.optim.scheduler=='ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.optim.gamma)
    elif cfg.optim.scheduler=='CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optim.T_max)
    elif cfg.optim.scheduler=='CosineDecay':
        scheduler = timm.scheduler.CosineLRScheduler(optimizer, t_initial=cfg.optim.epochs,  warmup_lr_init=cfg.optim.lr_core/1e3,  warmup_t=10)

    if cfg.model.pretrained==2 and cfg.optim.spr_w>0:
        pretrain_path = cfg.model.pretrain_path
        net.load_state_dict(torch.load(pretrain_path), strict=True)
        print('loading '+ pretrain_path)
    # Move to GPU
    if torch.cuda.is_available():
        net = net.cuda()
        num_gpu = list(range(torch.cuda.device_count()))

    stats = STATS(cfg)
    if cfg.optim.smoothing>0:
        criterion = timm.loss.LabelSmoothingCrossEntropy(smoothing=cfg.optim.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()


    # Dummy forward for initializing LSQ, We do NOT need this when using pre-trained model from same bit
    if (cfg.model.pretrained==2 and cfg.optim.spr_w>0) or cfg.model.act_bit<=0:
        net.set_init_state(-1)
    else:
        net.set_init_state(0)
        dummy(cfg, stats, criterion, net, optimizer, scheduler, train_loader)
        net.set_init_state(1)
    
    # Parallel GPUs
    net = torch.nn.DataParallel(net, device_ids=cfg.hardware.gpu_device)
    
    for epoch in range(cfg.optim.epochs):
        stats.epoch = epoch
        stats.init_meter('train')
        train(cfg, stats, criterion, net, optimizer, scheduler, train_loader)  
        stats.save(cfg.misc.log_name)
        stats.disp()
                
        stats.init_meter('val')      
        evaluate(cfg, stats, criterion, net, optimizer, scheduler, val_loader)
        stats.save()
        stats.disp()
    
        if stats.epoch% 10 == 0:
            torch.save(net.module.state_dict(), cfg.model_path_final)
        if stats.epoch==stats.log.best.epoch:
            torch.save(net.module.state_dict(), cfg.model_path_best)
                
def dummy(cfg, stats, criterion, net, optimizer, scheduler, loader):
    net.train()
    net.reset_hook()
    with torch.no_grad():
        for iter, batch in tqdm.tqdm(enumerate(loader)):
            x, _ = extract_batch(batch)
            _ = net(x)


def train(cfg, stats, criterion, net, optimizer, scheduler, loader):
    net.train()

    lamda_c = cfg.optim.lamda_ini
    net.module.set_lamda(lamda_c)

    for iter, batch in tqdm.tqdm(enumerate(loader)):
        optimizer.zero_grad()
        x, labels = extract_batch(batch)
       
        net.module.register_rand_hook(3)
        
        outputs = net(x)
        ce_loss = criterion(outputs, labels)

        
        layer_stats = net.module.aggregate_dss()  
        stats.update_meter(ce_loss, outputs, labels, optimizer.param_groups[0]['lr'], layer_stats)
    
        if cfg.optim.spr_w==0:
            loss = ce_loss
        else:
            loss = ce_loss + cfg.optim.spr_w*layer_stats.op_loss.to('cuda:0')
        loss.backward()
        
        if iter % 16 == 0:
            stats.disp()
            
        optimizer.step()
        if isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step() 

    
    if not isinstance(scheduler, lr_scheduler.OneCycleLR):
        if isinstance(scheduler, timm.scheduler.CosineLRScheduler):
            scheduler.step(stats.epoch)
        else:
            scheduler.step()
    

def evaluate(cfg, stats, criterion, net, optimizer, scheduler, loader):
    net.eval()
    for iter, batch in tqdm.tqdm(enumerate(loader)):
        with torch.no_grad():
            x, labels = extract_batch(batch)
            net.module.register_all_hook()
            outputs = net(x)
            ce_loss = criterion(outputs, labels)
            layer_stats = net.module.aggregate_dss()    
        stats.update_meter(ce_loss, outputs, labels, optimizer.param_groups[0]['lr'], layer_stats)


if __name__ == "__main__":
    main()