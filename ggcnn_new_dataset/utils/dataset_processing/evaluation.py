import cv2
import math
from skimage.draw import polygon
from skimage.feature import peak_local_max
from torch.jit import Error
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('/home/wangdx/research/grasp_detection/sim_grasp/sgdn/')
from utils.data.structure.grasp import GRASP_WIDTH_MAX



def length(pt1, pt2):
    """
    计算两点间的欧氏距离
    :param pt1: [row, col]
    :param pt2: [row, col]
    :return:
    """
    return pow(pow(pt1[0] - pt2[0], 2) + pow(pt1[1] - pt2[1], 2), 0.5)


def diff_angle_bin(pred_angle_bin, label_angle_bins, thresh_bin=1):
    """
    判断预测的抓取角类别与抓取角标签之差是否小于阈值

    :param pred_angle_bin: 预测的抓取角类别 
    :param label_angle_bins: 一维数组 array (k, )  标注的抓取角标签概率

    :return: 
        pred_success: 预测的抓取角类别与抓取角标签之差是否小于等于阈值  
        labels: 满足抓取角类别差值小于等于阈值的抓取角类别标签
    """
    label_bins = np.argwhere(label_angle_bins == 1) # shape=(n,1)
    label_bins = np.reshape(label_bins, newshape=(label_bins.shape[0],))
    label_bins = list(label_bins)

    pred_success = False
    labels = []
    angle_k = label_angle_bins.shape[0] # 17

    for label_bin in label_bins:
        if abs(label_bin - pred_angle_bin) <= thresh_bin\
            or abs(label_bin - pred_angle_bin) >= (angle_k-thresh_bin):
            pred_success = True
            labels.append(label_bin)

    return pred_success, labels


def diff(k, label):
    """
    计算cls与label的差值
    :param k: int 不大于label的长度
    :param label: 一维数组 array (k, )  label为多标签的标注类别
    :return: min_diff: 最小的差值 int    clss_list: 角度GT的类别 len=1/2/angle_k
    """
    clss = np.argwhere(label == 1)
    clss = np.reshape(clss, newshape=(clss.shape[0],))
    clss_list = list(clss)
    min_diff = label.shape[0] + 1

    for cls in clss_list:
        min_diff = min(min_diff, abs(cls - k))

    return min_diff, clss_list


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


def rect_loc(row, col, angle, height, bottom):
    """
    计算矩形的四个角的坐标[row, col]
    :param row:矩形中点 row
    :param col:矩形中点 col
    :param angle: 抓取角 弧度
    :param height: 抓取宽度
    :param bottom: (三角形的底)
    :param angle_k: 抓取角分类数
    :return:
    """
    xo = np.cos(angle)
    yo = np.sin(angle)

    y1 = row + height / 2 * yo
    x1 = col - height / 2 * xo
    y2 = row - height / 2 * yo
    x2 = col + height / 2 * xo

    return np.array(
        [
         [y1 - bottom/2 * xo, x1 - bottom/2 * yo],
         [y2 - bottom/2 * xo, x2 - bottom/2 * yo],
         [y2 + bottom/2 * xo, x2 + bottom/2 * yo],
         [y1 + bottom/2 * xo, x1 + bottom/2 * yo],
         ]
    ).astype(np.int)


def polygon_iou(polygon_1, polygon_2):
    """
    计算两个多边形的IOU
    :param polygon_1: [[row1, col1], [row2, col2], ...]
    :param polygon_2: 同上
    :return:
    """
    rr1, cc1 = polygon(polygon_2[:, 0], polygon_2[:, 1])
    rr2, cc2 = polygon(polygon_1[:, 0], polygon_1[:, 1])

    try:
        r_max = max(rr1.max(), rr2.max()) + 1
        c_max = max(cc1.max(), cc2.max()) + 1
    except:
        return 0

    canvas = np.zeros((r_max, c_max))
    canvas[rr1, cc1] += 1
    canvas[rr2, cc2] += 1
    union = np.sum(canvas > 0)
    if union == 0:
        return 0
    intersection = np.sum(canvas == 2)
    return intersection / union


def calcAngle2(angle):
    """
    根据给定的angle计算与之反向的angle
    :param angle: 弧度
    :return: 弧度
    """
    return angle + math.pi - int((angle + math.pi) // (2 * math.pi)) * 2 * math.pi


def evaluation(pred_pos, pred_angle, pred_width, target_pos, target_angle, target_width):
    """
    评估预测结果
    :param pred_success:    预测抓取置信度     (h, w)
    :param pred_angle_bins: 预测抓取角         (h, w)
    :param pred_width:      预测抓取宽度       (h, w)

    :param target_success:      抓取点标签      (1, 1, h, w)
    :param target_angle_bins:   抓取角标签      (1, bin, h, w)
    :param target_width:        抓取宽度标签    (1, bin, h, w)

    :return:
        0或1, 0-预测错误, 1-预测正确

        与任意一个label同时满足以下两个条件，认为预测正确：
        1、抓取点距离小于5像素
        2、偏转角小于等于30°
        3、抓取宽度之比小于0.8
    """
    # 阈值
    thresh_pos = 0.3
    thresh_pt = 5
    thresh_angle = 30.0 / 180 * math.pi
    thresh_width = 0.8

    # label预处理
    target_pos   = target_pos.cpu().numpy().squeeze()         # (h, w)
    target_angle = target_angle.cpu().numpy().squeeze()     # (h, w)
    target_width  = target_width.cpu().numpy().squeeze() * GRASP_WIDTH_MAX    # (h, w)  

    # 当标签都是负样本
    # 预测成功率都是负样本时，判为正确
    # 预测成功率有正样本时，判为错误
    if np.max(target_pos) < 1:
        return 1

    # 当最高置信度小于阈值时，判为检测失败
    if np.max(pred_pos) < thresh_pos:
        return 0

    # 获取最高置信度的抓取点
    loc = np.argmax(pred_pos)
    pred_pt_row = loc // pred_pos.shape[0]          # 预测的抓取点
    pred_pt_col = loc % pred_pos.shape[0]           # 预测的抓取点
    pred_angle = (pred_angle[pred_pt_row, pred_pt_col] + 2 * math.pi) % math.pi  # 预测的抓取角 弧度
    pred_width = pred_width[pred_pt_row, pred_pt_col]      # 预测的抓取宽度 像素

    # 以p为中点，搜索正方形内的label
    H, W = pred_pos.shape
    search_l = max(pred_pt_col - thresh_pt, 0)
    search_r = min(pred_pt_col + thresh_pt, W-1)
    search_t = max(pred_pt_row - thresh_pt, 0)
    search_b = min(pred_pt_row + thresh_pt, H-1)

    for target_row in range(search_t, search_b+1):
        for target_col in range(search_l, search_r+1):
            if target_pos[target_row, target_col] != 1.0:
                continue

            # 抓取点
            if length([target_row, target_col], [pred_pt_row, pred_pt_col]) > thresh_pt:
                continue
            
            # 抓取角
            label_angle = (target_angle[target_row, target_col] + 2*math.pi) % math.pi
            if abs(label_angle - pred_angle) > thresh_angle\
                and abs(label_angle - pred_angle) < (math.pi-thresh_angle):
                continue
            # 抓取宽度
            label_width = target_width[target_row, target_col]  # 抓取宽度标签
            if (pred_width / label_width) >= thresh_width and (pred_width / label_width) <= 1./thresh_width:
                return 1
    return 0
