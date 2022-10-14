import os
import cv2
import sys
import torch
import glob
import numpy as np
import tensorboardX
from torchsummary import summary


class Saver:
    def __init__(self, path, logdir, modeldir, imgdir, net_desc):
        self.path = path            # save path
        self.logdir = logdir        # tensorboard file folder
        self.modeldir = modeldir    # model file folder
        self.imgdir = imgdir        # img file folder
        self.net_desc = net_desc

    def save_summary(self):
        """
        save to tensorboard
        """
        save_folder = os.path.join(self.path, self.logdir, self.net_desc)    # output/tensorboard/....
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        return tensorboardX.SummaryWriter(save_folder)

    def save_arch(self, net, shape):
        """
        save network architecture self.path/arch.txt
        """
        f = open(os.path.join(self.path, 'arch.txt'), 'w')
        sys.stdout = f
        summary(net, shape)  
        sys.stdout = sys.__stdout__
        f.close()

    def save_model(self, net, model_name):
        """
        save model
        """
        model_path = os.path.join(self.path, self.modeldir, self.net_desc)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # torch.save(net, os.path.join(model_path, model_name))
        torch.save(net.state_dict(), os.path.join(model_path, model_name))

    def remove_model(self, num):
        """
        remove extra model
        """
        # search saved model
        model_path = os.path.join(self.path, self.modeldir, self.net_desc)
        models = glob.glob(model_path + '/*_.pth')
        models.sort()
        if len(models) > num:
            for file in models[:len(models)-num]:
                os.remove(file)

    def save_img(self, epoch, idx, imgs):
        """
        save predicted image
        """
        able_out_1_255 = imgs[1].copy() * (255 / imgs[1].max())
        able_out_1_255 = able_out_1_255.astype(np.uint8)

        able_y, _, _ = imgs[2]
        able_y = able_y.cpu().numpy().squeeze()  # (1, 1, 320, 320) -> (320, 320)

        able_y_255 = able_y.copy() * 255
        able_y_255 = able_y_255.astype(np.uint8)

        save_folder = os.path.join(self.path, self.imgdir, self.net_desc)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        able_out_1_filename = os.path.join(save_folder, '{}_{}_{:03d}.jpg'.format('able1', idx, epoch))
        able_y_filename = os.path.join(save_folder, '{}_{}.jpg'.format('abley', idx))

        cv2.imwrite(able_out_1_filename, able_out_1_255)
        cv2.imwrite(able_y_filename, able_y_255)
