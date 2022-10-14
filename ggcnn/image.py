from imageio.v2 import imread
import numpy as np
from skimage.transform import resize,rotate

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
        self.img = self.img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    def resize(self,shape):
        if self.img.shape == shape:
            return
        self.img = resize(self.img, shape, preserve_range=True).astype(self.img.dtype)
    
    def rotate(self,angle,center=None):
        if center is not None:
            center = (center[1],center[0])
        self.img = rotate(self.img,angle/np.pi*180,center = center,mode = 'symmetric',preserve_range=True).astype(self.img.dtype)

    def zoom(self,factor):
    
        sr = int(self.img.shape[0] * (1 - factor)) // 2
        sc = int(self.img.shape[1] * (1 - factor)) // 2

        origin_shape = self.img.shape
        self.img = self.img[sr:self.img.shape[0] - sr, sc: self.img.shape[1] - sc].copy()
        self.img = resize(self.img,origin_shape,mode='symmetric',preserve_range=True).astype(self.img.dtype)

    def normalise(self):
        self.img = self.img.astype(np.float32)/255.0
        self.img -= self.img.mean()


class DepthImage(Image):
    def __init__(self,img):
        super(DepthImage,self).__init__(img)
    
