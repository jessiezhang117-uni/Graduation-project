import numpy as np
import cv2
import torch
import torch.utils.data
import math
import random
import os
import copy
import glob
import sys
sys.path.append('./ggcnn_new_dataset')
from utils.data.structure.img import DepthImage
from utils.data.structure.grasp import GraspMat
from utils.dataset_processing import mmcv



class GraspDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, output_size=360, include_depth=True, include_rgb=False, argument=False):
        """
        :param output_size: int 
        :param include_depth: bool
        :param include_rgb: bool
        """

        self.output_size = output_size
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.argument = argument

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

        graspf = glob.glob(os.path.join(file_path, '*grasp.mat'))
        graspf.sort()
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        depthf = [f.replace('grasp.mat', 'd.tiff') for f in graspf]
        rgbf = [f.replace('grasp.mat', 'r.png') for f in graspf]

        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def __getitem__(self, idx):
        image = DepthImage(self.depth_files[idx])
        label = GraspMat(self.grasp_files[idx])
        # data argument
        if self.argument:
            # resize
            scale = np.random.uniform(0.9, 1.1)
            image.rescale(scale)
            label.rescale(scale)
            # rotate
            rota = 30
            rota = np.random.uniform(-1 * rota, rota)
            image.rotate(rota)
            label.rotate(rota)
            # crop
            dist = 30
            crop_bbox = image.crop(self.output_size, dist)
            label.crop(crop_bbox)
            # flip
            flip = True if np.random.rand() < 0.5 else False
            if flip:
                image.flip()
                label.flip()
        else:
            # crop
            crop_bbox = image.crop(self.output_size)
            label.crop(crop_bbox)

        image.normalize()
        label.encode()

        img = self.numpy_to_torch(image.img)
        grasp_point = self.numpy_to_torch(label.grasp_point)
        grasp_cos = self.numpy_to_torch(label.grasp_cos)
        grasp_sin = self.numpy_to_torch(label.grasp_sin)
        grasp_width = self.numpy_to_torch(label.grasp_width)

        return img, (grasp_point, grasp_cos, grasp_sin, grasp_width)


    def __len__(self):
        return len(self.grasp_files)


if __name__ == '__main__':
    dataset_path = './dataset/cornell'
    # 加载训练集
    print('Loading Dataset...')
    train_dataset = GraspDataset(dataset_path, start=0.0, end=0.2, argument=True, output_size=300)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1)

    print('>> dataset: {}'.format(len(train_data)))

    count = 0
    max_w = 0
    for x, y in train_data:
        count += 1
        np.squeeze
        img = x.cpu().numpy().squeeze()    # (h, w)
        img = np.array(img, dtype=np.float) # m
        img_color = mmcv.depth2Gray3(img)

        pos_img = y[0].cpu().numpy().squeeze()
        wid_img = y[3].cpu().numpy().squeeze()      
        ang_img = (torch.atan2(y[2], y[1]) / 2.0).cpu().numpy().squeeze() # radians

        grasps = []
        rows, cols = np.where(pos_img == 1)    # grasp point
        for i in range(rows.shape[0]):
            row, col = rows[i], cols[i]
            width = wid_img[row, col] * 200
            angle = (ang_img[row, col] + 2 * math.pi) % math.pi
            print('angle = ', angle)
            grasps.append([row, col, angle, width])

        img_color_region = img_color.copy()
        img_color_line = img_color.copy()
        im_grasp_region = mmcv.drawGrasps(img_color_region, grasps, mode='region')
        im_grasp_line = mmcv.drawGrasps(img_color_line, grasps, mode='line')
        cv2.imwrite('./dataset/cornell' + str(count) + '_region.png', im_grasp_region)
        cv2.imwrite('./dataset/cornell' + str(count) + '_line.png', im_grasp_line)