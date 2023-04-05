from numpy.core.numeric import Inf
from core.layers.bconv2d import BConv2d
from core.model_converter import convert_layers
import torch.nn as nn
import numpy as np
import torch
import random
from easydict import EasyDict as edict

class BNet(nn.Module):
    def __init__(self, model, cfg):        
        super(BNet, self).__init__()
        self.cfg = cfg
        self.selected_out =[]
        self.fhooks = []
        self.DSS_cand=[]
        self.selected_idx = []

        model, _ = convert_layers(model, cfg)
        self.model = model 
                    
        model.eval()
        dummy_input = torch.randn(cfg.dataset.input_shape)
        self.prepare_hook()
        self.compute_dense_syn_cnt(dummy_input)
        
        
    def forward(self, x):
        x = self.model(x)
        return x
        
    # hooks
    def prepare_hook(self):
        DSS_cand = []

        print('<<<<>>>>')
        idx = 0
        names = []
        th_cand = []
        for name, module in self.named_modules():
            if isinstance(module, BConv2d):
                module.name = name
                module.idx = idx            
                DSS_cand.append(module)
                names.append(name)
                idx+=1
        self.DSS_cand = DSS_cand
        self.th_cand = th_cand
        # print(DSS_cand)
        return

    def reset_hook(self):
        for fhook in self.fhooks:
            fhook.remove()
        self.fhooks.clear()
        self.selected_out.clear()
        
        for module in self.DSS_cand:
            if isinstance(module, BConv2d):
                module.selected_itr = []
                module.hook_count = 0
                

    def register_idx_hook(self, idx=[]):
        self.max_hook = 3
        for fhook in self.fhooks:
            fhook.remove()
        self.fhooks.clear()
        self.selected_out.clear()
        self.selected_idx = idx
        for idx_ in self.selected_idx:
            self.fhooks.append(self.DSS_cand[idx_].register_forward_hook(self.forward_hook(idx_)))


    def register_rand_hook(self, n_sample=1):
        self.max_hook = Inf
        for fhook in self.fhooks:
            fhook.remove()
        self.fhooks.clear()
        self.selected_out.clear()
        self.selected_idx.clear()
        for idx, module in enumerate(self.DSS_cand):
            module.is_selected=False
        
        for itr in range(n_sample):
            idx_ = random.choices(range(len(self.DSS_cand)), weights=self.dense_syn_cnt)[0]
            if idx_ not in self.selected_idx:
                self.selected_idx.append(idx_)
                self.DSS_cand[idx_].is_selected=True
                # self.selected_idx = np.random.randint(low=0, high=len(self.DSS_cand), size=1)[0]
                # print('register_rand_hook %d  %d '%(self.selected_idx, len(self.fhooks)))
                self.fhooks.append(self.DSS_cand[idx_].register_forward_hook(self.forward_hook(idx_)))

    def register_all_hook(self):
        self.max_hook = Inf
        for fhook in self.fhooks:
            fhook.remove()
        self.fhooks.clear()
        self.selected_out.clear()

        for idx, module in enumerate(self.DSS_cand):
            self.fhooks.append(module.register_forward_hook(self.forward_hook(idx)))

    def forward_hook(self, selected_idx):
        def hook(module, input, output):
            module.hook_count+=1
            if isinstance(module, BConv2d):
                module.compute_spr_loss(input[0])
                module.layer_stats.selected_idx = selected_idx
                self.selected_out.append(module)
        return hook

    def set_lamda(self, lamda):
        for idx, module in enumerate(self.DSS_cand):
            module.lamda = (lamda+0.001)
            

    def compute_dense_syn_cnt(self, input):
        for fhook in self.fhooks:
            fhook.remove()
        self.fhooks.clear()
        self.selected_out.clear()

        # print('register_all_hook %d '%(len(self.fhooks)))
        for idx, module in enumerate(self.DSS_cand):
            self.fhooks.append(module.register_forward_hook(self.forward_hook(idx)))

        self(input)
    
        dense_syn_cnt = np.zeros(len(self.DSS_cand))
        dense_act_cnt = np.zeros(len(self.DSS_cand))
        for module in self.selected_out:
            stats = module.layer_stats      
            dense_syn_cnt[stats.selected_idx]+=stats.dense_syn_cnt
            dense_act_cnt[stats.selected_idx]+=stats.dense_act_cnt

            
        self.dense_syn_cnt = dense_syn_cnt
        print('Dense SYN CNT (10^9): %f'%(dense_syn_cnt.sum()))
        print('Dense ACT CNT (10^9): %f'%(dense_act_cnt.sum()))

    def set_init_state(self, init_state):
        for module in self.DSS_cand:
             module.act_lsq.set_init_state(init_state)
                

    def aggregate_dss(self):
        # accumlate stats
        layer_stats_all = edict()
        layer_stats_all.op_loss     = 0.0
        layer_stats_all.mac         = 0.0
        layer_stats_all.sac         = 0.0
        layer_stats_all.mac2        = 0.0
        layer_stats_all.sac2        = 0.0
        layer_stats_all.bit_cnt     = 0.0
        layer_stats_all.wgt_cnt     = 0.0
        layer_stats_all.act_cnt     = 0.0
        
        layer_stats_all.dense_act_cnt     = 0.0
        layer_stats_all.dense_syn_cnt     = 0.0
        layer_stats_all.dense_wgt_cnt     = 0.0
        
        for module in self.selected_out:
            stats = module.layer_stats      
            layer_stats_all.op_loss     +=stats.op_loss.to('cuda:0')
            layer_stats_all.mac         +=stats.mac
            layer_stats_all.sac         +=stats.sac
            layer_stats_all.mac2        +=stats.mac2
            layer_stats_all.sac2        +=stats.sac2
            layer_stats_all.bit_cnt     +=stats.bit_cnt
            layer_stats_all.wgt_cnt     +=stats.wgt_cnt
            layer_stats_all.act_cnt     +=stats.act_cnt
            
            layer_stats_all.dense_act_cnt     +=stats.dense_act_cnt
            layer_stats_all.dense_syn_cnt     +=stats.dense_syn_cnt
            layer_stats_all.dense_wgt_cnt     +=stats.dense_wgt_cnt

        # Normalize 
        syn_scale = (np.sum(self.dense_syn_cnt)/layer_stats_all.dense_syn_cnt)
        layer_stats_all.op_loss /= layer_stats_all.dense_syn_cnt
        layer_stats_all.mac     *=  syn_scale
        layer_stats_all.sac     *=  syn_scale
        layer_stats_all.mac2    *=  syn_scale
        layer_stats_all.sac2    *=  syn_scale
        layer_stats_all.avg_bit = layer_stats_all.bit_cnt/layer_stats_all.dense_wgt_cnt 
        layer_stats_all.avg_wgt = layer_stats_all.wgt_cnt/layer_stats_all.dense_wgt_cnt
        layer_stats_all.avg_act = layer_stats_all.act_cnt/layer_stats_all.dense_act_cnt
        return layer_stats_all    