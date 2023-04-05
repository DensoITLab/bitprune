from easydict import EasyDict as edict
import numpy as np
import pickle
import os
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

class STATS():
    def __init__(self, config):
        self.cfg = config
        log = edict()
        log.train =  edict()
        log.val =  edict()
        log.test =  edict()
        

        def init_loss(e_obj):
            e_obj.ce_loss         = np.zeros(config.optim.epochs)
            e_obj.acc             = np.zeros(config.optim.epochs)
            e_obj.acc5            = np.zeros(config.optim.epochs)
            e_obj.mac             = np.zeros(config.optim.epochs)
            e_obj.sac             = np.zeros(config.optim.epochs)
            e_obj.mac2            = np.zeros(config.optim.epochs)
            e_obj.sac2            = np.zeros(config.optim.epochs)
            e_obj.op_loss         = np.zeros(config.optim.epochs)
            e_obj.avg_wgt         = np.zeros(config.optim.epochs)
            e_obj.avg_act         = np.zeros(config.optim.epochs)
            e_obj.avg_bit         = np.zeros(config.optim.epochs)
            e_obj.lr              = np.zeros(config.optim.epochs)
        init_loss(log.train)
        init_loss(log.val)
        init_loss(log.test)
        
        log.best = edict()
        log.best.acc = 0.0
        log.best.ce_loss = 0.0
        log.best.itr = 0
        log.best.epoch = -1

        self.log = log
        self.epoch = 0
            
    def init_meter(self, phase):
        self.phase = phase
        if phase=='train':
            print("<<<<<<<<< start epoch{:3d} {} {}".format(self.epoch,  self.cfg.dataset.name, self.cfg.cfg_name + ' ' +self.cfg.file_name))

        self.ce_loss_meter      = AverageMeter()
        self.acc_meter          = AverageMeter()
        self.acc5_meter         = AverageMeter()
        self.mac_meter          = AverageMeter()
        self.sac_meter          = AverageMeter()
        self.mac2_meter         = AverageMeter()
        self.sac2_meter         = AverageMeter()
        self.op_loss_meter      = AverageMeter()
        self.avg_wgt_meter      = AverageMeter()
        self.avg_act_meter      = AverageMeter()
        self.avg_bit_meter      = AverageMeter()
        self.ce_loss_meter      = AverageMeter()
        self.lr_meter           = AverageMeter()

    def update_meter(self, ce_loss, outputs, labels, lr, layer_stats):
        acc1, acc5 = self.__accuracy(outputs, labels, topk=(1, 5))
        
        self.ce_loss_meter.update(ce_loss.item())
        self.acc_meter.update(acc1.item())
        self.acc5_meter.update(acc5.item())
        self.op_loss_meter.update(layer_stats.op_loss.item())
        self.mac_meter.update(layer_stats.mac)
        self.sac_meter.update(layer_stats.sac)
        self.mac2_meter.update(layer_stats.mac2)
        self.sac2_meter.update(layer_stats.sac2)
        self.avg_wgt_meter.update(layer_stats.avg_wgt)
        self.avg_act_meter.update(layer_stats.avg_act)
        self.avg_bit_meter.update(layer_stats.avg_bit)   
        self.lr_meter.update(lr)   


    def save(self, logname='log2'):
        self.log[self.phase].ce_loss[self.epoch]    = self.ce_loss_meter.average()
        self.log[self.phase].acc[self.epoch]        = self.acc_meter.average()
        self.log[self.phase].acc5[self.epoch]       = self.acc5_meter.average()
        self.log[self.phase].mac[self.epoch]        = self.mac_meter.average()
        self.log[self.phase].sac[self.epoch]        = self.sac_meter.average()
        self.log[self.phase].mac2[self.epoch]       = self.mac2_meter.average()
        self.log[self.phase].sac2[self.epoch]       = self.sac2_meter.average()
        self.log[self.phase].op_loss[self.epoch]    = self.op_loss_meter.average()
        self.log[self.phase].avg_wgt[self.epoch]    = self.avg_wgt_meter.average()
        self.log[self.phase].avg_act[self.epoch]    = self.avg_act_meter.average()
        self.log[self.phase].avg_bit[self.epoch]    = self.avg_bit_meter.average()
        self.log[self.phase].lr[self.epoch]         = self.lr_meter.average()
        
        
        if self.phase=='val' and self.log[self.phase].acc[self.epoch]>self.log.best.acc:
            self.log.best.acc = self.log[self.phase].acc[self.epoch]
            self.log.best.epoch = self.epoch
            print(" Best updated!  epoch{:3d}, acc: {:0.3f}, ".format(self.log.best.epoch, self.log.best.acc))
                  
        with open(os.path.join(self.cfg.stats_dir, logname), 'wb') as handle:
            pickle.dump(self.log, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # print("finished epoch{:3d} {}>>>>>>>>>>".format(self.epoch, self.phase,))
        
    def disp(self):
        print_scale_op_loss    = 1e-3
        print_scale_mac        = 1e9
        
        print(" {}, lr:{:0.5f} epoch{:3d}, acc: {:0.3f}, acc5: {:0.3f}, ce: {:2.3f}, spr: {:2.3f},  mac: {:2.3f}, {:2.3f}, sac: {:2.3f},{:2.3f}  avg_bit: {:0.4f}, avg_wgt: {:0.4f}, avg_act: {:0.4f}".format(
            self.phase,  
            self.lr_meter.average(), 
            self.epoch, 
            self.acc_meter.average(), 
            self.acc5_meter.average(), 
            self.ce_loss_meter.average(),  
            self.op_loss_meter.average()/print_scale_op_loss, 
            self.mac_meter.average()/print_scale_mac, 
            self.mac2_meter.average()/print_scale_mac, 
            self.sac_meter.average()/print_scale_mac,
            self.sac2_meter.average()/print_scale_mac, 
            self.avg_bit_meter.average(),  
            self.avg_wgt_meter.average(),  
            self.avg_act_meter.average())
        )
        
        
    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(1.0 / batch_size))
            return res