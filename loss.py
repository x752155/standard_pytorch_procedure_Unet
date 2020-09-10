import torch
from data_loader import  *

import torch.nn as nn

from  config import *
class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, activation='sigmoid'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation


    def forward(self, y_pr, y_gt):
        print("forward")
        print(self.diceCoeffv2(y_pr, y_gt, activation=self.activation) )
        return 1 - self.diceCoeffv2(y_pr, y_gt, activation=self.activation)

    def diceCoeff(self,pred, gt, smooth=1e-5, activation='sigmoid'):
        r""" computational formula：
            dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
        """

        if activation is None or activation == "none":
            activation_fn = lambda x: x
        elif activation == "sigmoid":
            activation_fn = nn.Sigmoid()
        elif activation == "softmax2d":
            activation_fn = nn.Softmax2d()
        else:
            raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

        pred = activation_fn(pred)

        N = gt.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)

        intersection = (pred_flat * gt_flat).sum(1)
        unionset = pred_flat.sum(1) + gt_flat.sum(1)
        loss = (2 * intersection + smooth) / (unionset + smooth)

        return loss.sum() / N

    def diceCoeffv2(self,pred, gt, eps=1e-5, activation='sigmoid'):
        r""" computational formula：
            dice = (2 * tp) / (2 * tp + fp + fn)
        """

        if activation is None or activation == "none":
            activation_fn = lambda x: x
        elif activation == "sigmoid":
            activation_fn = nn.Sigmoid()
        elif activation == "softmax2d":
            activation_fn = nn.Softmax2d()
        else:
            raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

        pred = activation_fn(pred)

        N = gt.size(0)  # batch_size
        pred_flat = pred.view(N, -1)    # 把剩下的部分拉成一维的
        gt_flat = gt.view(N, -1)    # 标签值也是

        tp = torch.sum(gt_flat * pred_flat, dim=1)  #相乘代表相交
        fp = torch.sum(pred_flat, dim=1) - tp
        fn = torch.sum(gt_flat, dim=1) - tp
        loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        return loss.sum() / N


    # v2的另一种代码写法
    def diceCoeffv3(self,pred, gt, eps=1e-5, activation='sigmoid'):
        r""" computational formula：
            dice = (2 * tp) / (2 * tp + fp + fn)
        """

        if activation is None or activation == "none":
            activation_fn = lambda x: x
        elif activation == "sigmoid":
            activation_fn = nn.Sigmoid()
        elif activation == "softmax2d":
            activation_fn = nn.Softmax2d()
        else:
            raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

        pred = activation_fn(pred)

        N = gt.size(0)
        pred_flat = pred.view(N, -1)
        gt_flat = gt.view(N, -1)

        tp = torch.sum((pred_flat != 0) * (gt_flat != 0), dim=1)
        fp = torch.sum((pred_flat != 0) * (gt_flat == 0), dim=1)
        fn = torch.sum((pred_flat == 0) * (gt_flat != 0), dim=1)
        # 转为float，以防long类型之间相除结果为0
        loss = (2 * tp + eps).float() / (2 * tp + fp + fn + eps).float()

        return loss.sum() / N

def mask_to_onehot(mask, palette):
        """
        Converts a segmentation mask (H, W, C) to (K,H, W, ) where the last dim is a one
        hot encoding vector, C is usually 1, and K is the number of segmented class.
        eg:
        mask:单通道的标注图像
        palette:[[0],[1],[2],[3],[4],[5]]
        """
        # mask = mask.numpy()
        # palette = palette.numpy()
        semantic_map = []
        for colour in palette:
            equality = np.equal(mask, colour)
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
        # torch.from_numpy(semantic_map)
        return semantic_map

def onehot_to_mask(mask, palette):  #
        """
        Converts a mask (H, W, K) to (H, W, C)
        K is the number of segmented class,C is usually 1
        """
        # mask = mask.numpy()
        # palette = palette.numpy()
        x = np.argmax(mask, axis=-1)
        colour_codes = np.array(palette)
        x = np.uint8(colour_codes[x.astype(np.uint8)])
        # torch.from_numpy(x)
        return x
