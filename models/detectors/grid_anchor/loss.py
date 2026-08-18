"""
损失函数
"""
import torch
import yaml
from torch import nn
import torch.nn.functional as F



class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        with open("models/detectors/grid_anchor/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        self.nc = config["nc"]  # 类别数

        # 超参数
        # 各个损失权重
        self.lambda_obj = config["lambda_obj"]# 正样本置信度权重
        self.lambda_noobj = config["lambda_noobj"]# 负样本置信度权重
        self.lambda_lct_loss = config["lambda_lct_loss"]# 位置损失权重
        self.lambda_conf_loss = config["lambda_conf_loss"]# 置信度损失权重
        self.lambda_cls_loss = config["lambda_cls_loss"]# 类别损失权重
        # Focal Loss 超参数
        self.gamma = config["gamma"]          # 调制因子
        self.alpha = config["alpha"]         # 正样本权重（用于置信度 Focal Loss）
        # 多类别 Focal Loss 的 alpha 可设为类别频率的逆，这里简化为标量3.0
        self.cls_alpha = config["cls_alpha"]

    def forward(self, predict, target):
        """
        predict, target: [B, 13, 13, 5, 5+nc]
        predict[..., 0:4] : tx, ty, tw, th (未归一化的偏移量)
        predict[..., 4]   : 置信度 (sigmoid 后的概率)
        predict[..., 5:]  : 类别 logits (未经过 softmax)
        target[..., 0:4]  : 真实偏移量
        target[..., 4]    : 真实置信度 (0/1)
        target[..., 5:]   : 真实类别 one-hot
        """
        obj_mask = target[..., 4] > 0.5  # 正样本掩码 [B,13,13,5]
        if obj_mask.sum() == 0:
            return torch.tensor(0.0, device=predict.device)

        noobj_mask = ~obj_mask # 负样本掩码

        # -------------------- 1. 位置损失 (SmoothL1Loss) --------------------
        # 仅正样本参与计算
        loc_pred = predict[..., 0:4][obj_mask]   # [N, 4]
        loc_target = target[..., 0:4][obj_mask]  # [N, 4]
        lct_loss = F.smooth_l1_loss(loc_pred, loc_target, beta=1.0)

        # -------------------- 2. 置信度损失 (Binary Focal Loss) --------------------
        # 输入 predict[...,4] 已经是 sigmoid 概率 (0~1)
        conf_pred = predict[..., 4]              # [B,13,13,5]
        conf_target = target[..., 4]             # [B,13,13,5]

        # Binary Focal Loss 公式: FL = -alpha_t * (1-p_t)^gamma * log(p_t)
        # 其中二分类的 p_t = p  if y=1 else 1-p
        pt = conf_pred * conf_target + (1 - conf_pred) * (1 - conf_target)  # 对正确类别的预测概率
        focal_weight = (1 - pt) ** self.gamma
        # alpha_t: 正样本 alpha, 负样本 1-alpha
        alpha_t = self.alpha * conf_target + (1 - self.alpha) * (1 - conf_target)
        bce = F.binary_cross_entropy(conf_pred, conf_target, reduction='none')  # 普通 BCE
        focal_bce = alpha_t * focal_weight * bce

        # 分别聚合正负样本
        conf_pos = focal_bce[obj_mask].sum()
        conf_neg = focal_bce[noobj_mask].sum()
        N = obj_mask.numel()  # 总样本数
        conf_loss = (self.lambda_obj * conf_pos + self.lambda_noobj * conf_neg) / N

        # -------------------- 3. 类别损失 (Multi-class Focal Loss) --------------------
        # 仅正样本参与，N为正样本个数
        cls_pred = predict[..., 5:5+self.nc][obj_mask]    # [N, C] logits
        cls_target = target[..., 5:5+self.nc][obj_mask]   # [N, C] one-hot

        # 将 one-hot 转为类别索引，如(0, 1, 0, 2, 1...)
        cls_target_idx = cls_target.argmax(dim=-1) # [N]

        # Multi-class Focal Loss
        ce_loss = F.cross_entropy(cls_pred, cls_target_idx, reduction='none')  # [N]，交叉熵损失
        pt = torch.exp(-ce_loss)  # [N]，对正确类别的预测概率
        focal_weight = (1 - pt) ** self.gamma # [N]，难度权重
        # 这里 cls_alpha 可设为标量或每个类别的权重向量，简化为标量 1.0
        cls_loss = (self.cls_alpha * focal_weight * ce_loss).mean()

        # -------------------- 4. 总损失 --------------------
        total_loss = (self.lambda_lct_loss * lct_loss +
                        self.lambda_conf_loss * conf_loss +
                        self.lambda_cls_loss * cls_loss)
        return total_loss
