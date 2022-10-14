import torch
import glob
import os
import numpy as np
import random
from grasp import Grasps
from image import Image,DepthImage

class Cornell(torch.utils.data.Dataset):
    def __init__(self,file_dir,include_depth=True,include_rgb=True,start = 0.0,end = 1.0,random_rotate=False,random_zoom=False,output_size = 300):
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
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom

        # load dataset with given path 
        graspf = glob.glob(os.path.join(file_dir,'*','pcd*cpos.txt'))
        graspf.sort()
        
        
        l = len(graspf)
        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path{}'.format(file_dir))
        
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


    def get_rgb(self,idx,rot=0,zoom=1.0,normalise=True):

        rgb_img = Image.from_file(self.rgbf[idx])
        rgb_img.normalize()
        center,left,top = self._get_crop_attrs(idx)
        # first rotate then crop and zoom, finally resize
        rgb_img.rotate(rot,center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2,0,1))
        
        return rgb_img.img

    def get_depth(self,idx,rot=0,zoom=1.0):

        depth_img = DepthImage.from_file(self.depthf[idx])
        depth_img.normalize()
        center,left,top = self._get_crop_attrs(idx)
        depth_img.rotate(rot,center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))

        return depth_img.img
    
    def get_grasp(self,idx,rot=0,zoom=1.0):

        grs = Grasps.load_from_cornell_files(self.graspf[idx])
        center,left,top = self._get_crop_attrs(idx)
        #rotate,offset then zoom
        grs.rotate(rot,center)
        grs.offset((-left,-top))
        grs.zoom(zoom,(self.output_size//2,self.output_size//2))
    
        pos_img,angle_img,width_img = grs.generate_img(shape = (self.output_size,self.output_size))

        return pos_img,angle_img,width_img

    def get_raw_grasps(self,idx,rot,zoom):

        raw_grasps = Grasps.load_from_cornell_files(self.graspf[idx])
        center, left, top = self._get_crop_attrs(idx)
        raw_grasps.rotate(rot,center)
        raw_grasps.offset((-left,-top))
        raw_grasps.zoom(zoom,(self.output_size//2,self.output_size//2))

        
        return raw_grasps

    def __getitem__(self,idx):

        if self.random_rotate:
            rotations = [0,np.pi/2,2*np.pi/2,3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0
        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5,1.0)
        else:
            zoom_factor = 1.0

        if self.include_depth:
            depth_img = self.get_depth(idx,rot=rot,zoom=zoom_factor)
            x = self.numpy_to_torch(depth_img)
        # load rgb images
        if self.include_rgb:
            rgb_img = self.get_rgb(idx,rot=rot,zoom=zoom_factor)
            # channel-first
            if rgb_img.shape[2] == 3:
                rgb_img = np.moveaxis(rgb_img,2,0)
            x = self.numpy_to_torch(rgb_img)
        if self.include_depth and self.include_rgb:# four channels
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img,0),rgb_img),0
                )
            )
            

        pos_img,angle_img,width_img = self.get_grasp(idx,rot=rot,zoom=zoom_factor)
        # mapping the angles
        cos_img = self.numpy_to_torch(np.cos(2*angle_img))
        sin_img = self.numpy_to_torch(np.sin(2*angle_img))
        
        pos_img = self.numpy_to_torch(pos_img)
        
        # limiting and mapping the width to [0,1]
        width_img = np.clip(width_img, 0.0, 150.0)/150.0
        width_img = self.numpy_to_torch(width_img)
        
        return x,(pos_img,cos_img,sin_img,width_img),idx,rot,zoom_factor
    

    def __len__(self):
        return len(self.graspf)

