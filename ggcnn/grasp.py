import numpy as np
from skimage.draw import polygon


def str2num(point):
    x,y = point.split()
    x,y = int(round(float(x))),int(round(float(y)))
    
    return np.array([x,y])



class Grasp:
    def __init__(self,points):
        '''
        :points : 2darry,[[x1,y1],[x2,y2],[x3,y3],[x4,x4]]
        '''
        self.points = points
       
    @property
    def center(self):
        center = np.mean(self.points,axis = 0).astype(np.int)
        return center
    
    @property
    def width(self):
        dx = self.points[0][0] - self.points[1][0]
        dy = self.points[0][1] - self.points[1][1]
        
        return np.sqrt(dx**2+dy**2)
    
    @property
    def length(self):
        dx = self.points[1][0] - self.points[2][0]
        dy = self.points[1][1] - self.points[2][1]
        
        return np.sqrt(dx**2+dy**2)
    
    @property
    def angle(self):

        dx = self.points[0][0] - self.points[1][0]
        dy = self.points[0][1] - self.points[1][1]
        
        return (np.arctan2(-dy,dx) + np.pi/2) % np.pi - np.pi/2

    def polygon_coords(self, shape=None):
        return polygon(self.points[:, 0], self.points[:, 1], shape)
    
    def compact_polygon_coords(self,shape):
        '''
        return all the coordinates inside the grasp rectangle
        :shape    : tuple, optional.Image shape which is used to determine the maximum extent of output pixel coordinates.
        :ndarray  : rr,cc row,column coordinates
        ''' 
        return Grasp_cpaw(self.center, self.angle, self.length, self.width/3).as_gr.polygon_coords(shape)

    def offset(self, offset):
        """
        :offset: array [y, x] 
        """
        self.points += np.array(offset).reshape((1, 2))
    
    def rotate(self,angle,center):
        R = np.array(
            [
                [np.cos(angle),np.sin(angle)],
                [-1*np.sin(angle),np.cos(angle)]
            ]
        )
        c = np.array(center).reshape((1,2))
        self.points = ((np.dot(R,(self.points-c).T)).T+c).astype(np.int)
    
    def zoom(self,factor,center):
        T = np.array(
            [
                [1/factor,0],
                [0,1/factor]
            ]
        )
        c = np.array(center).reshape((1,2))
        
        self.points = ((np.dot(T,(self.points - c).T)).T+c).astype(np.int)
        



class Grasps:
    def __init__(self,grs = None):

        if grs:
            self.grs = grs
        else:
            self.grs = []
    

    def __getattr__(self, attr):
        """
        当用户调用某一个Grasps类中没有的属性时，查找iGrasp类中有没有这个函数，有的话就对Grasps类中的每个Grasp对象调用它。
        这里是直接从ggcnn里面搬运过来的，高端操作
        """
        if hasattr(Grasp, attr) and callable(getattr(Grasp, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)
    


    @classmethod
    def load_from_cornell_files(cls,cornell_grasp_files):
   
        grasp_rectangles = []
        with open(cornell_grasp_files,'r') as f:
            while True:
                grasp_rectangle = []
                point0 = f.readline().strip()
                if not point0:
                    break
                point1,point2,point3 = f.readline().strip(),f.readline().strip(),f.readline().strip()
                if point0[0] == 'N': # for points that is NaN
                    break
                grasp_rectangle = np.array([str2num(point0),
                               str2num(point1),
                               str2num(point2),
                               str2num(point3)])
                grasp_rectangles.append(Grasp(grasp_rectangle))

            return cls(grasp_rectangles)
        
    def generate_img(self,pos = True,angle = True,width = True,shape = (480,640)):
        if pos:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            angle_out = np.zeros(shape)
        else:
            angle_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None
        
        for gr in self.grs:
            rr,cc = gr.compact_polygon_coords(shape)
            
            if pos:
                pos_out[cc,rr] = 1.0
            if angle:
                angle_out[cc,rr] = gr.angle
            if width:
                width_out[cc,rr] = gr.width

        return pos_out,angle_out,width_out
    
    @property
    def points(self):
        points = []
        for gr in self.grs:
            points.append(gr.points)
        return points

    @property
    def center(self):
        centers = []
        for gr in self.grs:
            centers.append(gr.center)
        center = np.mean(np.array(centers),axis = 0).astype(np.uint32)
        return center

class Grasp_cpaw:
    def __init__(self,center, angle, length=60, width=30):
        self.center = center
        self.angle = angle   
        self.length = length
        self.width = width
        
    @property
    def as_gr(self):
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)
        
        y1 = self.center[0] - self.width / 2 * xo
        x1 = self.center[1] + self.width / 2 * yo
        y2 = self.center[0] + self.width / 2 * xo
        x2 = self.center[1] - self.width / 2 * yo
        
        return Grasp(np.array(
            [
             [y1 - self.length/2 * yo, x1 - self.length/2 * xo],
             [y2 - self.length/2 * yo, x2 - self.length/2 * xo],
             [y2 + self.length/2 * yo, x2 + self.length/2 * xo],
             [y1 + self.length/2 * yo, x1 + self.length/2 * xo],
             ]
        ).astype(np.float))