from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as transforms
import torch.nn.init as init
import math
import os
    
# Adopted form https://github.com/hustzxd/LSQuantization/blob/master/lsq.py
def ste_round_(x, add_noise=False):
    eps = (x.round() - x).detach()
    # eps[eps.isnan()] = 0
    if add_noise==True and 0:
        pm = -eps.sgn()
        noise = pm*torch.randint_like(x,0,2)
        return x + eps + noise
    else:
        return x + eps
    
class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g

class Floor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.floor()

    @staticmethod
    def backward(ctx, g):
        return g


# Code adopted from https://github.com/hustzxd/EfficientPyTorch/tree/1fcb533c7bfdafba4aba8272f1e0c34cbde91309
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

class ActLSQ(nn.Module):
    def __init__(self, nbits=4):
        super(ActLSQ, self).__init__()
        self.nbits = nbits
        if self.nbits == 0:
            self.register_parameter('alpha', None)
            return
        requires_grad = nbits<=8
        self.register_buffer('init_state', torch.zeros(1))
        self.alpha = nn.Parameter(torch.Tensor(1), requires_grad=requires_grad)
        self.itr = 0
        self.running_mean = 0

    def forward(self, x):
        if self.alpha is None:
            return x

        Qp = 2 ** self.nbits - 1
        if self.init_state == 0:
            self.running_mean+=x.abs().mean()
            self.itr+=1
            return x
        
        # return x
        g = 1.0 / math.sqrt(x.numel() * Qp)
        alpha = grad_scale(self.alpha, g)
        x = round_pass((x / alpha).clamp(None, Qp)) * alpha
        return x
        
    def set_init_state(self, init_state):
        Qp = 2 ** self.nbits - 1
        if self.alpha is None:
            return 
            
        if init_state==0:
            self.init_state.fill_(0)
            self.running_mean=0
        elif init_state==1:
            self.init_state.fill_(1)
            self.alpha.data.copy_(2 * self.running_mean /self.itr / math.sqrt(Qp))
            # print(self.alpha)
        else:
            pass

def bit2min(bit):
    return -2**((bit-1))

def bit2max(bit):
    return 2**((bit-1))

def binary(x, bit):
    mask = 2**torch.arange(bit).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def compute_wgt_vec(bit):
    return torch.linspace(bit2min(bit), bit2max(bit), 2**bit+1)

def compute_bit_cost(bit):
    wgt_vec = compute_wgt_vec(bit)
    bit_cost = []
    for v_ in wgt_vec:
        tmp = binary((v_.abs()).byte(), bit)
        bit_cost.append(tmp.sum())
        
    bit_cost = torch.stack(bit_cost).float()
    
    # make dist matrix    
    bit_cost_m = bit_cost.view([2**bit+1,1]).sub(bit_cost.view([1,2**bit+1]))
    return bit_cost, bit_cost_m

def compute_wgt_cost(bit, pnorm):
    wgt_vec = compute_wgt_vec(bit)
    wgt_cost = torch.sgn(wgt_vec)*torch.pow(wgt_vec.abs(), pnorm)
    wgt_cost_m = -wgt_cost.view([-1,1]).sub(wgt_cost.view([1,-1])).abs()
    return wgt_cost, wgt_cost_m

def compute_cost(bit, pnorm, lamda):
    bit_cost, bit_cost_m = compute_bit_cost(bit)
    wgt_cost, wgt_cost_m = compute_wgt_cost(bit, pnorm)

    cost = lamda*bit_cost_m + wgt_cost_m
    tgt_bin = cost.argmax(dim=1)

    # Copy target value's cost, and add the distance to the 
    wgt_cost_new = torch.zeros_like(wgt_cost)
    wgt_cost_new = -wgt_cost_m[range(len(wgt_cost_new)), tgt_bin]
    cost = bit_cost[tgt_bin]  + wgt_cost_new
    return cost

def w2cost(w, cost, bit):
    w_ = w.add(2**(bit-1)).round()
    return cost[w_.long()].float()

def abs_binalize(x):
    return x.abs().gt(0).to(x)

########################################################################
# Bit-Pruning Conv2d
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
class BConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=[0,0], dilation=1, groups=1, bias=True, cfg=None, is_input=False):    
        super(BConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.cfg = cfg

        if not isinstance(stride, (tuple, list)):
            stride = (stride, stride)

        self.stride = stride        
        w_init_alg = 'kaiming_uniform' # 'kaiming_uniform', 'kaiming_normal'
        self.register_parameter("fweight", nn.Parameter(torch.zeros(out_channels, in_channels//groups, kernel_size[0], kernel_size[1])))

        if w_init_alg=='kaiming_normal':
            init.kaiming_normal_(self.fweight)
        elif w_init_alg=='kaiming_uniform':
            init.kaiming_uniform_(self.fweight)
    

        self.mask_th = 0.0
        if bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros(out_channels)))
        else:
            self.bias = None
            
        self.reg_type = 3
        self.hook_count = 0
        self.lamda = (self.cfg.optim.lamda_ini+0.001)
        
        self.thd_neg = bit2min(cfg.model.wgt_bit)
        self.thd_pos = bit2max(cfg.model.wgt_bit)

        bit_cost, bit_cost_m = compute_bit_cost(cfg.model.wgt_bit)
        wgt_cost, wgt_cost_m = compute_wgt_cost(cfg.model.wgt_bit, self.cfg.optim.wgt_p_norm)

        self.register_parameter("bit_cost", nn.Parameter(bit_cost,      requires_grad=False))
        self.register_parameter("bit_cost_m", nn.Parameter(bit_cost_m,  requires_grad=False))  
        self.register_parameter("wgt_cost", nn.Parameter(wgt_cost,      requires_grad=False))
        self.register_parameter("wgt_cost_m", nn.Parameter(wgt_cost_m,  requires_grad=False))
        
        if self.cfg.optim.lamda_ini>0:
            cost = compute_cost(self.cfg.model.wgt_bit, self.cfg.optim.wgt_p_norm, self.lamda+0.001)
        else:
            cost = self.bit_cost
        self.register_parameter("cost", nn.Parameter(cost,  requires_grad=False))
                
        self.act_lsq = ActLSQ(0 if is_input else cfg.model.act_bit)

        self.is_selected = False
        self.layer_stats = edict()
        self.layer_stats.dense_act_cnt = 1
        self.layer_stats.dense_syn_cnt = 1
        self.layer_stats.dense_wgt_cnt = torch.numel(self.fweight)
        
        self.register_parameter("scale", nn.Parameter(torch.ones([out_channels, 1, 1, 1])))
        self.set_scale()
            
    def set_scale(self, type='absmax'):
        with torch.no_grad():
            v_max =  self.fweight.abs().flatten(1).max(dim=1)[0]
            v_std  = 3*self.fweight.flatten(1).std(dim=1)
                
            if type=='absmax':
                scale = v_max.div(self.thd_pos)
            else:
                scale = torch.minimum(v_max, v_std).div(self.thd_pos)
    
        self.scale.data.copy_(scale.view([self.out_channels, 1, 1, 1]).data)

    def count_bit(self, x, cost, bit):
        hist = torch.histogram(x.cpu(), bins=2**bit+1, range=(bit2min(bit)-0.5, bit2max(bit)+0.5), density=False)
        bit_cnt = hist[0].mul(cost.cpu().squeeze()).sum()
        return bit_cnt, self.layer_stats.dense_wgt_cnt - hist[0][2**(bit-1)]

    def calibrate_masked_weight(self):
        self.fweight.data = self.fweight.data + torch.sign(self.fweight.data)*self.mask_th

    def compute_out_shape(self,x):
        b, c, h, w = x.shape
        h_ = np.floor((h+2*self.padding[0]-1*(self.kernel_size[0]-1)-1)/self.stride[0]+1)-(self.dilation[0]-1)*2
        w_ = np.floor((w+2*self.padding[1]-1*(self.kernel_size[1]-1)-1)/self.stride[1]+1)-(self.dilation[0]-1)*2
        
        self.view4 = (-1, c*np.prod(self.kernel_size), int(h_), int(w_))
        self.view5 = (-1, c, np.prod(self.kernel_size), int(h_), int(w_))     
        return self.view4, self.view5

    def get_dense_syn_cnt(self):
        return self.dense_syn_cnt
    
    def tgt_loss(self,  x_unfold, w, mask_act=False, use_correction=False):
        tgt = self.compute_prox()
        
        if use_correction:
            with torch.no_grad():
                d_cost = self.compute_dcost()
            d = F.softshrink((tgt-w), lambd=0.5).mul(d_cost).div(6).div(256)
        else:
            d = F.softshrink(tgt-w, lambd=0.5).div(256)        
        
        if mask_act:
            L = self.act_spr_loss(x_unfold, d)
        else:
            L  = self.wgt_spr_loss(d)
        return L
                

    def bilinear_loss(self, x_unfold, w, mask_act=False):
        bit = self.cfg.model.wgt_bit
        cost = self.cost

        pi = F.pad(torch.tensor(cost).to(w.device).float(), (1,1), mode='constant', value=cost[0])

        pi = pi.div(pi.sum())  

        w = w.add(1.0 + 2**(bit-1))
        
        x_floor = w.floor().long()
        v_floor = pi[x_floor]
        
        x_ceil = w.ceil().long()
        v_ceil = pi[x_ceil] 
        
        d_floor     = w - w.floor()
        d_ceil      = (1-d_floor)

        v = v_floor*d_ceil + v_ceil*d_floor
        
        if mask_act:
            L = torch.einsum('oi, bimn->b',  v.flatten(1), x_unfold).mean() 
        else:
            H, W = x_unfold.shape[-2:]
            L  = v.sum()
        return L
    
    def act_spr_loss(self, x, weight):
        if self.groups>1:
            x = x.view(self.view5)
        else:
            x = x.view(self.view4)
        # d = d.view([out_shape[0],  -1, out_shape[-2]* out_shape[-1]])
        weight = weight.flatten(1)
        if self.reg_type==3:
            # Square-Hoyer
            if self.groups>1:
                sum1 = torch.einsum('oi, boimn->b',  weight.abs(), x)
                sum2 = torch.einsum('oi, boimn->b',  weight.abs(), x**2)
            else:
                sum1 = torch.einsum('oi, bimn->b',  weight.abs(), x)
                sum2 = torch.einsum('oi, bimn->b',  weight.abs(), x**2)
            dss  = (sum1**2)/(sum2+1e-06)
        elif self.reg_type==4:
            # Group-Square-Hoyer
            sum1 = torch.einsum('oi, bimn->bi',  weight.abs(), x)
            sum2 = torch.einsum('oi, bimn->bi',  weight.abs(), x**2)
            dss  = ((sum1**2)/(sum2+1e-06)).sum(1)
        elif self.reg_type==1:
            # 1 norm
            dss = torch.einsum('oi, bimn->b',  weight.abs(), x)
        return dss.mean()
    
    def calc_l1_and_zero_ratio(self, weights, scale):
        x = Round.apply(weights.abs() / 2 ** (scale - 8))

        b1 = Floor.apply(x/64)
        b2 = Floor.apply((x-b1.detach()*64)/16)
        b3 = Floor.apply((x-b1.detach()*64-b2.detach()*16)/4)
        b4 = x-b1.detach()*64-b2.detach()*16-b3.detach()*4
        
        l1_norm = b1.abs().sum() + b2.abs().sum() + b3.abs().sum() + b4.abs().sum()
        return l1_norm

    def wgt_spr_loss(self, weight):
        # d = d.view([out_shape[0],  -1, out_shape[-2]* out_shape[-1]])
        if self.reg_type==3:
            # Square-Hoyer
            sum1 = weight.abs().flatten(0).sum(dim=0)
            sum2 = (weight**2).flatten(0).sum(dim=0)
            L  = (sum1**2)/(sum2+1e-06)
        elif self.reg_type==4:
            # Group-Square-Hoyer
            sum1 = weight.abs().flatten(1).sum(dim=1)
            sum2 = (weight**2).flatten(1).sum(dim=1)
            L  = (sum1**2)/(sum2+1e-06)
            L = L.sum()
        elif self.reg_type==1:
            # 1 norm
            L = weight.abs().sum()
        return L.mean()  
        
    def compute_spr_loss(self, x):
        if not self.is_conv:
            x = x.permute([0,3,1,2])
        
        x = self.act_lsq(x)
        # x = x.relu()
        
        self.compute_out_shape(x)
        weight_s = self.scale_w(self.fweight)
        weight_r = self.round_w(weight_s, prep=True)
        weight_q = self.qunatize_w(weight_r, prep=True)
        weight_b = abs_binalize(weight_r) # no-grad

        # Unfold
        if self.is_conv:
            I_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride, dilation=self.dilation).view(self.view4)
        else:
            I_unfold = x

        bit_cnt, wgt_cnt = self.count_bit(weight_s, self.bit_cost, self.cfg.model.wgt_bit)
        
        space_scale = np.prod(self.view4[-2:])/self.groups
        
        op_loss = torch.tensor([0.0]).to(x)
        if self.cfg.optim.loss_type[:3]=='act':
            if self.cfg.optim.loss_type in ['act_bilinear']:
                op_loss  = self.bilinear_loss(I_unfold, weight_s,  mask_act=True)
                op_loss = op_loss.mul(32) # magic number to make the loss comparable to other losses
            elif self.cfg.optim.loss_type in ['act_tgt']:
                op_loss  = self.tgt_loss(I_unfold, weight_s, mask_act=True, use_correction=self.cfg.optim.use_correction)
                op_loss = op_loss.mul(256) # magic number to make the loss comparable to other losses
            elif self.cfg.optim.loss_type in ['act_slice']:
                op_loss  = self.calc_l1_and_zero_ratio(self.fweight, self.scale)
                op_loss = op_loss.mul(256) # magic number to make the loss comparable to other losses
            elif self.cfg.optim.loss_type in ['act_naive']:
                op_loss = self.act_spr_loss(I_unfold, weight_q)
            else:
                os.error('unsupported loss_type ' + self.cfg.optim.loss_type)
        elif self.cfg.optim.loss_type[:3]=='wgt':
            if self.cfg.optim.loss_type in ['wgt_bilinear']:
                op_loss  = self.bilinear_loss(I_unfold, weight_s, mask_act=False)
                op_loss = op_loss.mul(space_scale).mul(32) # magic number to make the loss comparable to other losses
            elif self.cfg.optim.loss_type in ['wgt_tgt']:
                op_loss  = self.tgt_loss(I_unfold, weight_s, mask_act=False).mul(space_scale)
                op_loss = op_loss.mul(space_scale).mul(4) # magic number to make the loss comparable to other losses
            elif self.cfg.optim.loss_type in ['wgt_slice']:
                op_loss  = self.calc_l1_and_zero_ratio(self.fweight, self.scale)
                op_loss = op_loss.mul(256) # magic number to make the loss comparable to other losses
            elif self.cfg.optim.loss_type in ['wgt_naive']:
                op_loss = self.wgt_spr_loss(weight_q).mul(space_scale)
            else:
                os.error('unsupported loss_type ' + self.cfg.optim.loss_type)
        else:
            os.error('unsupported loss_type ' + self.cfg.optim.loss_type)

        with torch.no_grad():
            w_cost  = w2cost(weight_r, self.bit_cost.to(x.device), self.cfg.model.wgt_bit)
            if self.groups>1:
                mac     = torch.einsum('oi, boimn->b',  weight_b.flatten(1), (I_unfold.view(self.view5)!=0).to(x)).mean()
                sac     = torch.einsum('oi, boimn->b',  w_cost.flatten(1), (I_unfold.view(self.view5)!=0).to(x)).mean()
            else:
                mac     = torch.einsum('oi, bimn->b',  weight_b.flatten(1), (I_unfold!=0).to(x)).mean()
                sac     = torch.einsum('oi, bimn->b',  w_cost.flatten(1), (I_unfold!=0).to(x)).mean()
            mac2    = wgt_cnt.mul(space_scale)
            sac2    = bit_cnt.mul(space_scale)
            act_cnt = torch.count_nonzero(x)/x.shape[0]
            
        assert(~torch.isnan(op_loss))
        self.layer_stats.op_loss     = op_loss
        self.layer_stats.mac         = mac.item()
        self.layer_stats.sac         = sac.item()
        self.layer_stats.mac2        = mac2.item() # weight sparsity only
        self.layer_stats.sac2        = sac2.item() # weight sparsity only
        self.layer_stats.bit_cnt     = bit_cnt.item()
        self.layer_stats.wgt_cnt     = wgt_cnt.item()
        self.layer_stats.act_cnt     = act_cnt.item()
      
    # https://pmelchior.net/blog/proximal-matrix-factorization-in-pytorch.html
    def solve_argmin(self, lamda=1.001):
        cost = lamda*self.bit_cost_m + self.wgt_cost_m
        tgt_bin = cost.argmax(dim=1)
        return tgt_bin
    
    def compute_prox(self):
        min_idx  = self.solve_argmin(self.lamda)
        rw = self.round_w((self.fweight)).sub(self.thd_neg).long()        
        return min_idx[rw].float().add(self.thd_neg)
  
    def compute_dcost(self):
        min_idx  = self.solve_argmin(self.lamda)
        rw = self.round_w((self.fweight)).sub(self.thd_neg).long()
        d_cost = self.bit_cost[rw] - self.bit_cost[min_idx][rw]
        return d_cost.float()
    
          
    def project_prox(self, x):
        with torch.no_grad():
            # Project to proximal point
            fweight_        = self.compute_prox().mul(self.scale)
            output_prox     = F.conv2d(x, fweight_, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation) # 'zeros'
        output_org      = F.conv2d(x, self.qunatize_w(self.fweight), bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation) # 'zeros'

        eps = (output_prox - output_org).detach()
        return output_org + eps
    
    # LSQ Qauntization 
    def qunatize_w(self,w, prep=False):
        if prep:
            return w.mul(self.scale)
        else:
            return self.round_w(w).mul(self.scale)
    def round_w(self,w, prep=False):
        if prep:
            return ste_round_(w)
        else:
            return ste_round_(self.scale_w(w))
    def scale_w(self,w):
        return torch.clamp(w.div(self.scale), self.thd_neg-0.5,  self.thd_pos+0.5)
    
    
    def forward(self, x):
        x = self.act_lsq(x)        
        self.layer_stats.dense_act_cnt=np.prod([*x.shape[1:]])
        if self.is_conv:
            self.layer_stats.dense_syn_cnt=np.prod([*self.fweight.shape])*np.prod([*x.shape[2:]])/(np.prod(self.stride))/(self.groups)
        else:
            self.layer_stats.dense_syn_cnt=np.prod([*self.fweight.shape])*np.prod([*x.shape[1:3]])/(np.prod(self.stride))
        
        # Experimental code (Project to proximal point in forward pass)
        if self.is_selected and self.cfg.optim.loss_type in ['wgt_prox']:
            return self.project_prox(x)
        
        if self.cfg.optim.loss_type[:4]=='fp32':
            fweight_ = self.fweight
        else:
            fweight_ = self.qunatize_w(self.fweight)
        
        # Main convolution
        if self.is_conv:
            output = F.conv2d(x, fweight_, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups) 
        else:
            output = F.linear(x, fweight_[:,:,0,0], bias=self.bias)
            
        return output