from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.utils import to_list
from pytorch_forecasting.data.encoders import EncoderNormalizer, GroupNormalizer, MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric, MultiLoss, QuantileLoss
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
import numpy as np
from numpy.lib.function_base import iterable
import pandas as pd
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.parsing import AttributeDict, get_init_args
import scipy.stats
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from pytorch_forecasting.utils import (
    OutputMixIn,
    apply_to_list,
    create_mask,
    get_embedding_size,
    groupby_apply,
    move_to_device,
    to_list,
)

from losses.crf_loss import CrfLoss
from cus_utils.visualization import VisUtil
from tft.class_define import CLASS_VALUES
from .embedding import embed
from .parameters import *
from custom_model.crf import CRF
from custom_model.beamsearch import *
from cus_utils.utils_crf import maskset
from cus_utils.dataloader import data
from tft.class_define import CLASS_VALUES, EOS_IDX, WEOS_IDX, SOS_IDX, PAD_IDX
    
class Seq2Seq(nn.Module):
    def __init__(
        self,
        hidden_size: int = 16,
        qcut_len: int = 15,
        num_classes:int = 6,
        device=None,
        **kwargs,
    ):
        super().__init__()
        
        self.device = device
        self.hidden_size = hidden_size
        # 使用encoder,decoder方式,注意输入长度应该包括前后标志符号以及pad符号
        cti_size = qcut_len + 3
        self.enc = encoder(cti_size, num_classes,device=device)
        self.dec = decoder(num_classes,device=device)
        # self = self.cuda(device=self.device)  

    def ext_properties(self,**kwargs):
        self.viz = kwargs['viz']
        self.fig_save_path = kwargs['fig_save_path']
        if self.viz:
            self.viz_util = VisUtil()
            
    def configure_optimizers(self):
        """
        自定义优化器及学习率
        """
        
        # 默认使用adam优化器
        lrs = self.hparams.learning_rate
        if isinstance(lrs, (list, tuple)):
            lr = lrs[0]
        else:
            lr = lrs        
        ignored_params = list(map(id, self.loss.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.parameters())
        
        optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': self.loss.parameters(), 'lr': lr*10}], lr,weight_decay=0)
        # Assuming optimizer has two groups.
        lambda1 = lambda epoch: 0.9 ** epoch
        lambda2 = lambda epoch: 0.9 ** epoch
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        scheduler_config = {
            "scheduler": scheduler,
            "monitor": "val_loss",  # Default: val_loss
            "interval": "epoch",
            "frequency": 1,
            "strict": False,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
                      
    def forward(self, x: Dict[str, torch.Tensor],y: Tuple[torch.Tensor, torch.Tensor]):
        """
        input dimensions: n_samples x time x variables
        """
        
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]    
        encoder_target = x["encoder_target"].cuda(device=self.device)
        x_cat = x["encoder_cat"].cuda(device=self.device)
        x_cont = x["encoder_cont"].cuda(device=self.device)
        # 拼接离散变量和连续变量转换后的离散变量
        xc = x_cont #torch.cat((x_cat,x_cont),2)
        target = y[0].cuda(device=self.device)
        # 标签添加结束标志
        target = F.pad(target, (0,1), value=WEOS_IDX) 
        loss = 0
        mask,lens = self.base(xc, encoder_target, target)
        # 使用1作为初始值
        yi = torch.LongTensor([1] * target.size(0)).cuda(device=self.device)
        results = []
        for t in range(target.size(1)):          
            yo = self.dec(yi.unsqueeze(1), mask)
            results.append(yo)
            yi = target[:, t] # teacher forcing

            loss += F.nll_loss(yo, yi, ignore_index = PAD_IDX)        
        data = {"results":results,"target":target,"inputs_xc":xc,"inputs_xw":encoder_target}
        return loss,data
    
    def base(self, xc, xw, y0):
        # “词向量”添加起結束符號
        xw = F.pad(xw, (0,1,0,0), value=WEOS_IDX)  
        # “字符向量”添加起始和結束符號
        xc = F.pad(xc, (1,0,0,0,0,0), value=SOS_IDX) 
        xc = F.pad(xc, (0,1,0,0,0,0), value=EOS_IDX) 
        # xc多补充一个end行
        xc = F.pad(xc, (0,0,0,1,0,0), value=PAD_IDX)  
        b = y0.size(0) # batch size
        self.zero_grad()
        mask, lens = maskset(xw)
        lens = lens.cpu()
        # 传递连续值和离散值
        self.dec.M = self.enc(b, xc,xw, lens)
        self.dec.hidden = self.enc.hidden
        self.dec.attn.Va = zeros(b, 1, HIDDEN_SIZE)   
        return mask,lens 

    def val(self,  x: Dict[str, torch.Tensor],y: Tuple[torch.Tensor, torch.Tensor]):
        itw = list(CLASS_VALUES.keys())
        encoder_target = x["encoder_target"].cuda(device=self.device)
        x_cat = x["encoder_cat"].cuda(device=self.device)
        x_cont = x["encoder_cont"].cuda(device=self.device)       
        # 拼接离散变量和连续变量转换后的离散变量
        xc = x_cont # torch.cat((x_cat,x_cont),2)
        target = y[0].cuda(device=self.device)
        target = F.pad(target, (0,1), value=WEOS_IDX)  
        mask,lens = self.base(xc, encoder_target, target)
        b = target.size(0)
        t = 0
        # 使用1作为初始值
        yi = torch.LongTensor([[1]] * b).cuda(device=self.device)     
        eos = [False for _ in encoder_target]
        batch = data()   
        batch.idx = list(range(len(encoder_target)))
        batch.xc = xc
        batch.xw = encoder_target
        batch.y1 = [[] for _ in range(b)]
        batch.prob = [torch.FloatTensor([0]).cuda(device=self.device) for _ in range(b)]
        batch.attn = [[["", *batch.x1[i], EOS]] for i in batch.idx]
        while t < MAX_LEN and sum(eos) < len(eos):
            yo = self.dec(yi, mask)
            args = (self.dec, batch, itw, eos, lens, yo)
            yi = beam_search(*args, t) if BEAM_SIZE > 1 else greedy_search(*args)
            t += 1
        results = []
        for i, (x0, y0, y1) in enumerate(zip(xc, target, batch.y1)):
            if not i % BEAM_SIZE: # use the best candidate from each beam
                y1 = [itw[y] for y in y1[:-1]]
                results.append((x0, y0, y1))        
        return results
       
class encoder(nn.Module):
    def __init__(self, cti_size, wti_size,device=None):
        super().__init__()
        self.hidden = None # encoder hidden states

        # architecture
        self.embed = embed(ENC_EMBED, cti_size, wti_size,device=device)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim,
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self = self.cuda(device=device)
        
    def init_state(self, b): # initialize RNN states
        n = NUM_LAYERS * NUM_DIRS
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h) # hidden state
        if RNN_TYPE == "LSTM":
            cs = zeros(n, b, h) # LSTM cell state
            return (hs, cs)
        return hs

    def forward(self, b, xc, xw, lens):
        self.hidden = self.init_state(b)
        x = self.embed(xc,xw)
        x = nn.utils.rnn.pack_padded_sequence(x, lens, enforce_sorted=False,batch_first = True)
        h, _ = self.rnn(x, self.hidden)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        return h
    
class decoder(nn.Module):
    def __init__(self, wti_size,device=None):
        super().__init__()
        self.M = None # source hidden states
        self.hidden = None # decoder hidden states

        # architecture
        self.embed = embed(DEC_EMBED, 0, wti_size,device=device)
        self.rnn = getattr(nn, RNN_TYPE)(
            input_size = self.embed.dim + HIDDEN_SIZE, # input feeding
            hidden_size = HIDDEN_SIZE // NUM_DIRS,
            num_layers = NUM_LAYERS,
            bias = True,
            batch_first = True,
            dropout = DROPOUT,
            bidirectional = (NUM_DIRS == 2)
        )
        self.attn = attn()
        self.out = nn.Linear(HIDDEN_SIZE, wti_size)
        self.softmax = nn.LogSoftmax(1)
        self = self.cuda(device=device)

    def forward(self, y1, mask):
        x = self.embed(None, y1)
        x = torch.cat((x, self.attn.Va), 2) # input feeding
        h, _ = self.rnn(x, self.hidden)
        h = self.attn(h, self.M, mask)
        h = self.out(h).squeeze(1)
        y = self.softmax(h)     
        return y

class attn(nn.Module):
    def __init__(self):
        super().__init__()

        # architecture
        self.Wa = None # attention weights
        self.Wc = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE)
        self.Va = None # attention vector

    def align(self, ht, hs, mask):
        a = ht.bmm(hs.transpose(1, 2)) # [B, 1, H] @ [B, H, L] = [B, 1, L]
        a = F.softmax(a.masked_fill(mask.unsqueeze(1), -10000), 2)
        return a # attention weights

    def forward(self, ht, hs, mask):
        self.Wa = self.align(ht, hs, mask)
        c = self.Wa.bmm(hs) # context vector [B, 1, L] @ [B, L, H] = [B, 1, H]
        self.Va = torch.tanh(self.Wc(torch.cat((c, ht), 2)))
        return self.Va # attention vector    
    