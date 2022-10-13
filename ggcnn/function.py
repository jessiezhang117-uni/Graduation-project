import torch
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.draw import polygon
import numpy as np

from grasp import Grasp_cpaw

def post_process(pos_img,cos_img,sin_img,width_img):
    '''
    get angle_img and add gaussian filter
    '''
    q_img = pos_img.cpu().data.numpy().squeeze()
    ang_img = (torch.atan2(sin_img,cos_img)/2.0).cpu().data.numpy().squeeze()
    width_img = width_img.cpu().data.numpy().squeeze()

    # batch_size should be 1    
    q_img_g = gaussian(q_img,2.0,preserve_range=True)
    ang_img_g = gaussian(ang_img,2.0,preserve_range=True)
    width_img_g = gaussian(width_img,1.0,preserve_range=True)

    return q_img_g,ang_img_g,width_img_g

def detect_grasps(q_out,ang_out,wid_out=None,num_grasp=1):
    '''
    get the most confident grasp from the predicted mapping images
    '''
    grasps_pre = []
    local_max = peak_local_max(q_out,min_distance=20,threshold_abs =0.1,num_peaks=1)
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)
        grasp_angle = ang_out[grasp_point]
        grasp_width = wid_out[grasp_point]
        g = Grasp_cpaw(grasp_point,grasp_angle,grasp_width)
        if wid_out is not None:
            g.width = wid_out[grasp_point]*150
            g.length = g.width/2
        grasps_pre.append(g)
    
    return grasps_pre

def max_iou(grasp_pre,grasps_true):
    '''
    calculate maximum iou
    '''
    grasp_pre = grasp_pre.as_gr
    max_iou = 0
    for grasp_true in grasps_true.grs:
        if iou(grasp_pre,grasp_true) > max_iou:
            max_iou = iou(grasp_pre,grasp_true)
    return max_iou

def iou(grasp_pre,grasp_true):
    '''
    calculate iou for two rectangles
    '''
    # get the are of two rectangles
    rr1, cc1 = grasp_pre.polygon_coords() #convert center and angle to four points
    rr2, cc2 = polygon(grasp_true.points[:,0],grasp_true.points[:,1])

    # read max position of the two rectangles
    r_max = max(rr1.max(),rr2.max())+1
    c_max = max(cc1.max(),cc2.max())+1

    #load canvas 
    canvas = np.zeros((r_max,c_max))
    canvas[rr1,cc1]+=1
    canvas[rr2,cc2]+=1

    union =np.sum(canvas>0)

    if union ==0:
        return 0
    
    intersection = np.sum(canvas==2)
    
    return intersection/union
