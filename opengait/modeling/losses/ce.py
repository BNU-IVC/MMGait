import torch.nn.functional as F

from .base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False):
        super(CrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy

    def forward(self, logits, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        n, c, p = logits.size()
        logits = logits.float()
        labels = labels.unsqueeze(1)
        if self.label_smooth:
            loss = F.cross_entropy(
                logits*self.scale, labels.repeat(1, p), label_smoothing=self.eps)
        else:
            loss = F.cross_entropy(logits*self.scale, labels.repeat(1, p))
        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info


import torch
class MaskCrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False):
        super(MaskCrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy

    def forward(self, logits, labels, modal_valid=None):
        """
            logits: [n, c, p]
            labels: [n]
            modal_valid: [n] boolean tensor, True for valid samples
        """
        n, c, p = logits.size()
        logits = logits.float()
        labels = labels.unsqueeze(1)
        
        if modal_valid is None:
            modal_valid = torch.ones(n, dtype=torch.bool, device=logits.device)
        # else:
        #     # 将 list 转换为 tensor
        #     modal_valid = torch.tensor(modal_valid, dtype=torch.bool, device=logits.device)
        
        
        # 只计算有效样本的 loss
        if modal_valid.sum() == 0:
            # 如果没有有效样本，返回零loss
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            self.info.update({'loss': loss.detach().clone()})
            if self.log_accuracy:
                self.info.update({'accuracy': torch.tensor(0.0)})
            return loss, self.info
        
        # 筛选有效样本
        valid_logits = logits[modal_valid]  # [n_valid, c, p]
        valid_labels = labels[modal_valid]  # [n_valid, 1]
        
        if self.label_smooth:
            loss = F.cross_entropy(
                valid_logits*self.scale, valid_labels.repeat(1, p), 
                label_smoothing=self.eps)
        else:
            loss = F.cross_entropy(
                valid_logits*self.scale, valid_labels.repeat(1, p))
        
        self.info.update({'loss': loss.detach().clone()})
        
        if self.log_accuracy:
            pred = valid_logits.argmax(dim=1)  # [n_valid, p]
            accu = (pred == valid_labels).float().mean()
            self.info.update({'accuracy': accu})
        
        return loss, self.info