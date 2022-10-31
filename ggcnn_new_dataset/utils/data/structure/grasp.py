import numpy as np
import cv2
import math
import scipy.io as scio
from ggcnn_new_dataset.utils.dataset_processing import mmcv


GRASP_WIDTH_MAX = 200.0


class GraspMat:
    def __init__(self, file):
        self.grasp = scio.loadmat(file)['A']   # (3, h, w)

    def height(self):
        return self.grasp.shape[1]

    def width(self):
        return self.grasp.shape[2]

    def crop(self, bbox):
        """
        crop self.grasp

        args:
            bbox: list(x1, y1, x2, y2)
        """
        self.grasp = self.grasp[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def rescale(self, scale, interpolation='nearest'):
        """
        resize
        """
        ori_shape = self.grasp.shape[1]
        self.grasp = np.stack([
            mmcv.imrescale(grasp, scale, interpolation=interpolation)
            for grasp in self.grasp
        ])
        new_shape = self.grasp.shape[1]
        ratio = new_shape / ori_shape
        # resize the grasp width
        self.grasp[2, :, :] = self.grasp[2, :, :] * ratio

    def rotate(self, rota):
        """
        rotate 
        rota: angle
        """
        self.grasp = np.stack([mmcv.imrotate(grasp, rota) for grasp in self.grasp])
        rota = rota / 180. * np.pi
        self.grasp[1, :, :] -= rota
        self.grasp[1, :, :] = self.grasp[1, :, :] % (np.pi * 2)
        self.grasp[1, :, :] *= self.grasp[0, :, :]

    def _flipAngle(self, angle_mat, confidence_mat):
        """
        flip angle
        Args:
            angle_mat: (h, w) radians
            confidence_mat: (h, w) 
        Returns:
        """
        # all 
        angle_out = (angle_mat // math.pi) * 2 * math.pi + math.pi - angle_mat
        # not grasp are goes to 0
        angle_out = angle_out * confidence_mat
        angle_out = angle_out % (2 * math.pi)

        return angle_out

    def flip(self, flip_direction='horizontal'):
        """
        horizontal flip
        """
        assert flip_direction in ('horizontal', 'vertical')

        self.grasp = np.stack([
            mmcv.imflip(grasp, direction=flip_direction)
            for grasp in self.grasp
        ])
        self.grasp[1, :, :] = self._flipAngle(self.grasp[1, :, :], self.grasp[0, :, :])

    def encode(self):
        """
        (4, H, W) -> (angle_cls+2, H, W)
        """
        self.grasp[1, :, :] = (self.grasp[1, :, :] + 2 * math.pi) % math.pi
        
        self.grasp_point = self.grasp[0, :, :]
        self.grasp_cos = np.cos(self.grasp[1, :, :] * 2)
        self.grasp_sin = np.sin(self.grasp[1, :, :] * 2)
        self.grasp_width = self.grasp[2, :, :]
    
