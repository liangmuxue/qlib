import torch
import torch.nn as nn
from TorchCRF import CRF

from pytorch_forecasting.metrics import MultiHorizonMetric

class CrfLoss(MultiHorizonMetric):
    """使用CRF模型进行loss计算
    """

    def __init__(self, num_classes,device=None,**kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        # self.device = device
        self.crf = CRF(num_labels=num_classes)

    def loss(self, inputs, targets):
        """
        Args:
            inputs 输入,shape为(batch_size,pred_size,output_size)
            targets 标签,shape为(batch_size,pred_size)
        """
        
        mask = (torch.ones(targets.shape)>0).to(self.device)      
        loss = self.crf(inputs, targets,mask) 
        loss = -1 * loss
        return loss
    
    def to_prediction(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        重载预测转换方法,以适应时序标注模式
        """
        pred_labels = self.crf.decode(y_pred) 
        pred_labels = torch.Tensor(pred_labels).cuda()
        return pred_labels
    
    def compute_acc(self,y_pred, targets):
        """
        计算准确率
        """   
        pred_labels = self.to_prediction(y_pred)
        comp = torch.eq(pred_labels, targets)
        # 分别比较每个标签,取得准确率
        acc = comp.sum()/(targets.shape[0]*targets.shape[1])
        # 计算相对准确率
        reduce = torch.abs(pred_labels - targets)
        reduce = reduce.sum()
        acc_relative = 1- reduce/targets.sum()
        return acc,acc_relative
        
    