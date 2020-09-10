import numpy as np
import cv2
import config
import os
import  torch
from loss import *
from resnet import *

# if __name__ == '__main__':
#      resnet_model = resnet50(pretrained=True)
#      rmodel = models.resnet50(pretrained=True)
#      img = cv2.imread(config.route_img+'5.tif')
#      img = [img]
#      img =torch.tensor(img).permute(0,3,1,2)
#      img =torch.tensor(img)
#      img = img.byte()
#      print(img.size())
#      mmm = rmodel(img.float())
#      print(mmm)
    #
    # # shape = torch.Size([2, 3, 4, 4])
    # # 模拟batch_size = 2
    # '''
    # 1 0 0= bladder
    # 0 1 0 = tumor
    # 0 0 1= background
    # '''
    # pred = torch.Tensor([[
    #     [[0, 1, 0, 0],
    #      [1, 0, 0, 1],
    #      [1, 0, 0, 1],
    #      [0, 1, 1, 0]],
    #     [[0, 0, 0, 0],
    #      [0, 0, 0, 0],
    #      [0, 1, 1, 0],
    #      [0, 0, 0, 0]],
    #     [[1, 0, 1, 1],
    #      [0, 1, 1, 0],
    #      [0, 0, 0, 0],
    #      [1, 0, 0, 1]]],
    #     [
    #         [[0, 1, 0, 0],
    #          [1, 0, 0, 1],
    #          [1, 0, 0, 1],
    #          [0, 1, 1, 0]],
    #         [[0, 0, 0, 0],
    #          [0, 0, 0, 0],
    #          [0, 1, 1, 0],
    #          [0, 0, 0, 0]],
    #         [[1, 0, 1, 1],
    #          [0, 1, 1, 0],
    #          [0, 0, 0, 0],
    #          [1, 0, 0, 1]]]
    # ])
    #
    # gt = torch.Tensor([[
    #     [[0, 1, 1, 0],
    #      [1, 0, 0, 1],
    #      [1, 0, 0, 1],
    #      [0, 1, 1, 0]],
    #     [[0, 0, 0, 0],
    #      [0, 0, 0, 0],
    #      [0, 1, 1, 0],
    #      [0, 0, 0, 0]],
    #     [[1, 0, 0, 1],
    #      [0, 1, 1, 0],
    #      [0, 0, 0, 0],
    #      [1, 0, 0, 1]]],
    #     [
    #         [[0, 1, 1, 0],
    #          [1, 0, 0, 1],
    #          [1, 0, 0, 1],
    #          [0, 1, 1, 0]],
    #         [[0, 0, 0, 0],
    #          [0, 0, 0, 0],
    #          [0, 1, 1, 0],
    #          [0, 0, 0, 0]],
    #         [[1, 0, 0, 1],
    #          [0, 1, 1, 0],
    #          [0, 0, 0, 0],
    #          [1, 0, 0, 1]]]
    # ])
    #
    # lo = SoftDiceLoss()
    # dice1 = lo.diceCoeff(pred[:, 0:1, :], gt[:, 0:1, :], activation=None)
    # dice2 = lo.diceCoeffv2(pred[:, 0:1, :], gt[:, 0:1, :], activation=None)
    # dice3 = lo.diceCoeffv3(pred[:, 0:1, :], gt[:, 0:1, :], activation=None)
    # dice4 = lo.forward(pred[:, 0:1, :], gt[:, 0:1, :])
    # dice5 = 1- dice1
    # print(dice1, dice2, dice3,dice4,dice5)
    # print(gt.size())
