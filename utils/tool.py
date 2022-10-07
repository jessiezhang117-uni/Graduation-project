import math
import cv2
import os
import zipfile
import numpy as np


def quaternion_to_rotation_matrix(q):  # x, y ,z ,w
    rot_matrix = np.array(
        [[1.0 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[0] * q[1] - q[3] * q[2]), 2 * (q[3] * q[1] + q[0] * q[2])],
         [2 * (q[0] * q[1] + q[3] * q[2]), 1.0 - 2 * (q[0] * q[0] + q[2] * q[2]), 2 * (q[1] * q[2] - q[3] * q[0])],
         [2 * (q[0] * q[2] - q[3] * q[1]), 2 * (q[1] * q[2] + q[3] * q[0]), 1.0 - 2 * (q[0] * q[0] + q[1] * q[1])]],
        dtype=np.float)
    return rot_matrix


def getTransfMat(offset, rotate):
    mat = np.array([
        [rotate[0, 0], rotate[0, 1], rotate[0, 2], offset[0]], 
        [rotate[1, 0], rotate[1, 1], rotate[1, 2], offset[1]], 
        [rotate[2, 0], rotate[2, 1], rotate[2, 2], offset[2]],
        [0, 0, 0, 1.] 
    ])
    return mat

def depth2Gray(im_depth):
    """
    depth image to gray image
    """
    # 16 bits to 8 bits
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('Rendering error ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max
    return (im_depth * k + b).astype(np.uint8)

def depth2Gray3(im_depth):
    """
    depth image to 3 channel gray image
    (h, w, 3)
    """
    # 16 bits to 8 bits
    x_max = np.max(im_depth)
    x_min = np.min(im_depth)
    if x_max == x_min:
        print('Rendering error ...')
        raise EOFError
    
    k = 255 / (x_max - x_min)
    b = 255 - k * x_max

    ret = (im_depth * k + b).astype(np.uint8)
    ret = np.expand_dims(ret, 2).repeat(3, axis=2)
    return ret


def distancePt(pt1, pt2):
    """
    calculate distance between two points
    pt: [row, col] 或 [x, y]
    return: float
    """
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

def distancePt3d(pt1, pt2):
    """
    calculate distance between three points
    pt: [x, y, z]
    return: float
    """
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2) ** 0.5


def calcAngleOfPts(pt1, pt2):
    """
    calculate angle between two points 
    
    pt: [x, y] 2D coordinates
    
    return: radians
    """
    dy = pt2[1] - pt1[1]
    dx = pt2[0] - pt1[0]
    return (math.atan2(dy, dx) + 2*math.pi) % (2*math.pi)
    

def radians_TO_angle(radians):
    """
    radians to angle
    """
    return 180 * radians / math.pi

def angle_TO_radians(angle):
    """
    angle to radians
    """
    return math.pi * angle / 180

def depth3C(depth):
    """
    depth to 3 channel np.uint8类型
    """
    depth_3c = depth[..., np.newaxis]
    depth_3c = np.concatenate((depth_3c, depth_3c, depth_3c), axis=2)
    return depth_3c.astype(np.uint8)

def zip_file(filedir):
    """
    zip file
    """
    file_news = filedir + '.zip'
    # overwrite exist file
    if os.path.exists(file_news):
        os.remove(file_news)

    z = zipfile.ZipFile(file_news,'w',zipfile.ZIP_DEFLATED) 
    for dirpath, dirnames, filenames in os.walk(filedir):
        fpath = dirpath.replace(filedir,'') 
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename),fpath+filename)
    z.close()


def unzip(file_name):
    """
    unzip file
    """
    zip_ref = zipfile.ZipFile(file_name) 
    os.mkdir(file_name.replace(".zip","")) 
    zip_ref.extractall(file_name.replace(".zip","")) 
    zip_ref.close() 