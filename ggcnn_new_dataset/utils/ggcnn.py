import cv2
import os
import torch
import time
import math
from skimage.feature import peak_local_max
import numpy as np
from models.common import post_process_output
from models.loss import get_pred
from models import get_network


def input_img(img, out_size=300):
    """
    对图像进行裁剪，保留中间(320, 320)的图像
    Crop the images
    :param file: rgb file
    :return: input tensor, left top coordinate 
    """

    assert img.shape[0] >= out_size and img.shape[1] >= out_size, 'input depth image shape must bigger or equal to (300, 300)'

    # 裁剪中间图像块
    crop_x1 = int((img.shape[1] - out_size) / 2)
    crop_y1 = int((img.shape[0] - out_size) / 2)
    crop_x2 = crop_x1 + out_size
    crop_y2 = crop_y1 + out_size
    img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    # normalize
    img = np.clip(img - img.mean(), -1., 1.)

    tensor = torch.from_numpy(img[np.newaxis, np.newaxis, :, :])  # np to tensor

    return tensor, crop_x1, crop_y1


def arg_thresh(array, thresh):
    """
    获取array中大于thresh的二维索引
    :param array: 二维array
    :param thresh: float阈值
    :return: array shape=(n, 2)
    """
    res = np.where(array > thresh)
    rows = np.reshape(res[0], (-1, 1))
    cols = np.reshape(res[1], (-1, 1))
    locs = np.hstack((rows, cols))
    for i in range(locs.shape[0]):
        for j in range(locs.shape[0])[i+1:]:
            if array[locs[i, 0], locs[i, 1]] < array[locs[j, 0], locs[j, 1]]:
                locs[[i, j], :] = locs[[j, i], :]

    return locs


class GGCNN:
    def __init__(self, model, device, network='ggcnn2'):
        self.t = 0
        self.num = 0
        self.device = device
        # 加载模型
        print('>> loading network')
        ggcnn = get_network(network)
        self.net = ggcnn()
        self.net.load_state_dict(torch.load(model, map_location=self.device), strict=True)   # True:完全吻合，False:只加载键值相同的参数，其他加载默认值。
        self.net = self.net.to(device)
        print('>> load done')

    def fps(self):
        return 1.0 / (self.t / self.num)

    def predict(self, img, mode, thresh=0.3, peak_dist=1):
        """
        预测抓取模型
        :param img: 输入图像 np.array (h, w, 3)
        :param thresh: 置信度阈值
        :param peak_dist: 置信度筛选峰值
        :return:
            pred_grasps: list([row, col, angle, width])
            crop_x1
            crop_y1
        """
        # 预测
        input, self.crop_x1, self.crop_y1 = input_img(img)
        
        t1 = time.time()
        # 预测
        self.pos_out, self.cos_out, self.sin_out, self.wid_out = get_pred(self.net, input.to(self.device))
        t2 = time.time() - t1

        # 后处理
        pos_pred, ang_pred, wid_pred = post_process_output(self.pos_out, self.cos_out, self.sin_out, self.wid_out)
        if mode == 'peak':
            # 置信度峰值 抓取点
            pred_pts = peak_local_max(pos_pred, min_distance=peak_dist, threshold_abs=thresh)
        elif mode == 'all':
            # 超过阈值的所有抓取点
            pred_pts = arg_thresh(pos_pred, thresh=thresh)
        elif mode == 'max':
            # 置信度最大的点
            loc = np.argmax(pos_pred)
            row = loc // pos_pred.shape[0]
            col = loc % pos_pred.shape[0]
            pred_pts = np.array([[row, col]])
        else:
            raise ValueError

        # 绘制预测的抓取
        pred_grasps = []
        for idx in range(pred_pts.shape[0]):
            row, col = pred_pts[idx]
            angle = (ang_pred[row, col] + 2 * math.pi) % math.pi
            width = wid_pred[row, col]
            row += self.crop_y1
            col += self.crop_x1

            pred_grasps.append([row, col, angle, width])

        self.t += t2
        self.num += 1

        return pred_grasps, self.crop_x1, self.crop_y1
