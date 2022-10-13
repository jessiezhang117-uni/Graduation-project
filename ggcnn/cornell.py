import torch
import glob
import os
import numpy as np

from grasp import Grasps
from image import Image,DepthImage

class Cornell(torch.utils.data.Dataset):
    def __init__(self,file_dir,include_depth=True,include_rgb=True,start = 0.0,end = 1.0,output_size = 300):
        '''
        :file_dir : str, file path of cornell dataset
        :include_depth : bool
        :include_rgb   : bool
        :start,end : float,for train and test dataset split 
        ''' 
        super(Cornell,self).__init__()

        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.output_size = output_size
        # load dataset with given path 
        graspf = glob.glob(os.path.join(file_dir,'*','pcd*cpos.txt'))
        graspf.sort()
        
        
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('Cannot find dataset, check the file path{}'.format(file_dir))
        
        rgbf = [filename.replace('cpos.txt','r.png') for filename in graspf]
        depthf = [filename.replace('cpos.txt','d.tiff') for filename in graspf]

        # get dataset with given splitting parameters
        self.graspf = graspf[int(l*start):int(l*end)]
        self.rgbf = rgbf[int(l*start):int(l*end)]
        self.depthf = depthf[int(l*start):int(l*end)]

    @staticmethod
    def numpy_to_torch(s):
        '''
        :convert numpy array to torch tensor
        :s   :numpy ndarray
        :return     :tensor
        '''
        if len(s.shape) == 2: #expand channel dimension
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
    

        
    def _get_crop_attrs(self,idx):
        '''
        :read center for multi rectangles, calculate the left coordinate for cropping based on output size
        :idx :int,
        '''
        grasp_rectangles = Grasps.load_from_cornell_files(self.graspf[idx])
        center = grasp_rectangles.center
        # prevent the corner points exceed the boundary
        left = max(0, min(center[0] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[1] - self.output_size // 2, 480 - self.output_size))
        
        return center,left,top


    def get_rgb(self,idx):

        rgb_img = Image.from_file(self.rgbf[idx])
        rgb_img.normalize()
        center,left,top = self._get_crop_attrs(idx)
        rgb_img.crop((left,top),(left+self.output_size,top+self.output_size))
        rgb_img.resize((self.output_size, self.output_size))
        
        return rgb_img.img

    def get_depth(self,idx):

        depth_img = DepthImage.from_file(self.depthf[idx])
        depth_img.normalize()
        center,left,top = self._get_crop_attrs(idx)
        depth_img.crop((left,top),(left+self.output_size,top+self.output_size))
        depth_img.resize((self.output_size, self.output_size))

        return depth_img.img
    
    def get_grasp(self,idx):

        grs = Grasps.load_from_cornell_files(self.graspf[idx])
        grs.offset((-(grs.center[0]-self.output_size//2),-(grs.center[1]-self.output_size//2)))
        pos_img,angle_img,width_img = grs.generate_img(shape = (480,640))
        

        pos_img,angle_img,width_img = grs.generate_img(shape = (self.output_size,self.output_size))

        return pos_img,angle_img,width_img

    def __getitem__(self,idx):
        if self.include_depth:
            depth_img = self.get_depth(idx)
            x = self.numpy_to_torch(depth_img)
        
        if self.include_rgb:
            rgb_img = self.get_rgb(idx)
            #channel-first
            if rgb_img.shape[2] == 3:
                rgb_img = np.moveaxis(rgb_img,2,0)
            x = self.numpy_to_torch(rgb_img)
        if self.include_depth and self.include_rgb:# four channels
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img,0),rgb_img),0
                )
            )
            
   
        pos_img,angle_img,width_img = self.get_grasp(idx)
        # mapping the angles
        cos_img = self.numpy_to_torch(np.cos(2*angle_img))
        sin_img = self.numpy_to_torch(np.sin(2*angle_img))
        
        pos_img = self.numpy_to_torch(pos_img)
        
        # limiting and mapping the width to [0,1]
        width_img = np.clip(width_img, 0.0, 150.0)/150.0
        width_img = self.numpy_to_torch(width_img)
        
        return x,(pos_img,cos_img,sin_img,width_img)
    

    def __len__(self):
        return len(self.graspf)

