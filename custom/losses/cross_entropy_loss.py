import torch
import torch.nn as nn

from pytorch_forecasting.metrics import Metric

class CrossEntropyLoss(Metric):
    r"""Cross entropy loss with label smoothing regularizer.
    
    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by
    
    .. math::
        \begin{equation}
        (1 - \eps) \times y + \frac{\eps}{K},
        \end{equation}

    where :math:`K` denotes the number of classes and :math:`\eps` is a weight. When
    :math:`\eps = 0`, the loss function reduces to the normal cross entropy.
    
    Args:
        num_classes (int): number of classes.
        eps (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, num_classes, device=None,eps=0.1, use_gpu=True, label_smooth=True):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps if label_smooth else 0
        # self.device = device,
        self.use_gpu = use_gpu
        self.logsoftmax = nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=None,
            ignore_index=-100,
            reduction="mean",
        )        

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        
        
        log_probs = self.logsoftmax(inputs)
        zeros = torch.zeros(log_probs.size())
        t = targets[0]
        if isinstance(t,torch.Tensor):
            t = targets[0].data.cpu().type(torch.int64)
        else:
            t = targets[0][0].data.cpu().type(torch.int64).unsqueeze(-1)
        if self.use_gpu:
            t = t.cuda()            
        loss = self.criterion(inputs, t.squeeze(-1))
        return loss
        targets = zeros.scatter_(1, t, 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.eps) * targets + self.eps / self.num_classes
        return (-targets * log_probs).mean(0).sum()
