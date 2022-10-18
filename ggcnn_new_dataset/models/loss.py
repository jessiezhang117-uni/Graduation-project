import math
import time
import torch
import torch.nn.functional as F



def bin_focal_loss(pred, target, gamma=2, alpha=0.6):
    """
    基于二值交叉熵的focal loss    
    增加难分类样本的损失
    增加负样本
    
    pred:   (N, C, H, W)
    target: (N, C, H, W)
    width: 是否输入的是抓取宽度

    对于抓取点和抓取角，置信度越大，alpha越大。置信度=0，alpha=0.4; 置信度=1，alpha=0.6。 
    对于抓取宽度，y=0时，alpha=0.4，其他alpha都等于0.6
    """
    n, c, h, w = pred.size()

    _loss = -1 * target * torch.log(pred + 1e-7) - (1 - target) * torch.log(1 - pred+1e-7)    # (N, C, H, W)
    _gamma = torch.abs(pred - target) ** gamma

    zeros_loc = torch.where(target == 0)
    _alpha = torch.ones_like(pred) * alpha
    _alpha[zeros_loc] = 1 - alpha

    loss = _loss * _gamma * _alpha
    loss = loss.sum() / (n*c*h*w)
    return loss


def focal_loss(net, x, y_pos, y_cos, y_sin, y_wid):
    """
    计算 focal loss
    params:
        net: 网络
        x:     网络输入图像   (batch, 1,   h, w)
        y_pos: 抓取点标签图   (batch, 1,   h, w)
        y_cos: 抓取cos标签图   (batch, 1,   h, w)
        y_sin: 抓取sin标签图   (batch, 1,   h, w)
        y_wid: 抓取宽度标签图   (batch, 1,   h, w)
    """

    # 获取网络预测
    pred_pos, pred_cos, pred_sin, pred_wid = net(x)         # shape 同上

    pred_pos = torch.sigmoid(pred_pos)
    loss_pos = bin_focal_loss(pred_pos, y_pos, alpha=0.9) * 10

    pred_cos = torch.sigmoid(pred_cos)
    loss_cos = bin_focal_loss(pred_cos, (y_cos+1)/2, alpha=0.9) * 10

    pred_sin = torch.sigmoid(pred_sin)
    loss_sin = bin_focal_loss(pred_sin, (y_sin+1)/2, alpha=0.9) * 10

    pred_wid = torch.sigmoid(pred_wid)
    loss_wid = bin_focal_loss(pred_wid, y_wid, alpha=0.9) * 10


    return {
        'loss': loss_pos + loss_cos + loss_sin + loss_wid,
        'losses': {
            'loss_pos': loss_pos,
            'loss_cos': loss_cos,
            'loss_sin': loss_sin,
            'loss_wid': loss_wid,
        },
        'pred': {
            'pred_pos': pred_pos, 
            'pred_cos': pred_cos, 
            'pred_sin': pred_sin, 
            'pred_wid': pred_wid, 
        }
    }



def get_pred(net, xc):
    net.eval()
    with torch.no_grad():
        pred_pos, pred_cos, pred_sin, pred_wid = net(xc)
        
        pred_pos = torch.sigmoid(pred_pos)
        pred_cos = torch.sigmoid(pred_cos)
        pred_sin = torch.sigmoid(pred_sin)
        pred_wid = torch.sigmoid(pred_wid)

    return pred_pos, pred_cos, pred_sin, pred_wid
