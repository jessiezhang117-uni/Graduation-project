from imageio.v2 import imread
import numpy as np
from skimage.transform import resize

class Image:
    def __init__(self,img):
        self.img = img
    
    @classmethod
    def from_file(cls,file_path):
        return cls(imread(file_path))
    
    def img_format(self):
        pass
    
    def normalize(self):
        self.img = self.img.astype('float32')/255.0
        self.img = self.img-self.img.mean()
        
    def crop(self,top_left,bottom_right):
        self.img = self.img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    def resize(self,shape):
        if self.img.shape == shape:
            return
        self.img = resize(self.img, shape, preserve_range=True).astype(self.img.dtype)
    

class DepthImage(Image):
    def __init__(self,img):
        super(DepthImage,self).__init__(img)
    
    @classmethod
    def from_file(cls,file_path):
        return cls(imread(file_path))
    
    def normalize(self):
        self.img = self.img.astype('float32')/255.0
        self.img = self.img -self.img.mean()

    def crop(self,top_left,bottom_right):
        self.img = self.img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    def resize(self,shape):
        if self.img.shape == shape:
            return
        self.img = resize(self.img, shape, preserve_range=True).astype(self.img.dtype)
    