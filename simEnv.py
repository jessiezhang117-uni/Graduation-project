import pybullet as p
import pybullet_data
import time
import math
import os
import glob
import random
import cv2
import shutil
import numpy as np
import scipy.io as scio
import sys
import scipy.stats as ss
import skimage.transform as skt
sys.path.append('/home/delta/Documents/Graduation_project_Jie/Simulation')


# Image size
IMAGEWIDTH = 640
IMAGEHEIGHT = 480

nearPlane = 0.01
farPlane = 10

fov = 60    # vertical angle -> image height: tan(30) * 0.7 *2 = 0.8082903m
aspect = IMAGEWIDTH / IMAGEHEIGHT



def imresize(image, size, interp="nearest"):
    skt_interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5
    }
    if interp in ("lanczos", "cubic"):
        raise ValueError("'lanczos' and 'cubic'"
                         " interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation '{}' not"
                                      " supported.".format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size '{}'.".format(type(size)))

    return skt.resize(image,
                      output_shape,
                      order=skt_interp_map[interp],
                      anti_aliasing=False,
                      mode="constant")

def inpaint(img, missing_value=0):
    """
    Inpaint missing values in depth image.
    :param missing_value: Value to fill in teh depth image.
    """
    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (img == missing_value).astype(np.uint8)

    # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
    scale = np.abs(img).max()
    img = img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
    img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)

    # Back to original size and value range.
    img = img[1:-1, 1:-1]
    img = img * scale

    return img


class SimEnv(object):
    def __init__(self, bullet_client, path, gripperId=None):
        """
        path: URDF file path 
        """
        self.p = bullet_client
        # self.p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # maxNumCmdPer1ms: add 1ms sleep if the number of commands executed exceed this threshold
        # solverResidualThreshold: velocity threshold
        # eneableFileCaching: set to 0 to disable file caching
        self.p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000, solverResidualThreshold=0, enableFileCaching=0)
        self.p.resetDebugVisualizerCamera(cameraDistance=1.7, cameraYaw=78, cameraPitch=-24, cameraTargetPosition=[0, 0, 0])
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())  
        self.planeId = self.p.loadURDF("plane.urdf", [0, 0, 0])     
        # self.trayId = self.p.loadURDF('objects_model/tray/tray_small.urdf', [0, 0, -0.007])
        self.p.setGravity(0, 0, -10) 
        self.flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES # cache and re-use graphics shapes, improve loading performance
        self.gripperId = gripperId

        # load camera
        self.movecamera(0, 0)
        self.projectionMatrix = self.p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
       
        # read object_list.txt
        list_file = os.path.join(path, 'object_list.txt')
        if not os.path.exists(list_file):
            raise shutil.Error
        self.urdfs_list = []
        with open(list_file, 'r') as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                self.urdfs_list.append(os.path.join(path, line[:-1]+'.urdf') )

        self.num_urdf = 0
        self.urdfs_id = []  
        self.objs_id = []   
        self.EulerRPList = [[0, 0], [math.pi/2, 0], [-1*math.pi/2, 0], [math.pi, 0], [0, math.pi/2], [0, -1*math.pi/2]]

    
    def _urdf_nums(self):
        return len(self.urdfs_list)
    

    def movecamera(self, x, y, z=0.7):
        """
        move camera to target position
        x, y: world coordinates x,y
        """
        self.viewMatrix = self.p.computeViewMatrix(cameraEyePosition=[x, y, z],cameraTargetPosition=[x, y, 0], cameraUpVector=[0, 1, 0])   # camera height fix to 0.7m



    # load single URDF obj
    def loadObjInURDF(self, urdf_file, idx, render_n=0):
        """
        urdf_file: urdf file 
        idx: object id
        render_n: to get object orientation 
        """
        # get object file
        if idx >= 0:
            self.urdfs_filename = [self.urdfs_list[idx]]
            self.objs_id = [idx]
        else:
            self.urdfs_filename = [urdf_file]
            self.objs_id = [-1]
        self.num_urdf = 1


        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []

        # orientation
        # baseEuler = [self.EulerRPList[render_n][0], self.EulerRPList[render_n][1], 0]
        # baseEuler = [self.EulerRPList[render_n][0], self.EulerRPList[render_n][1], random.uniform(0, 2*math.pi)]
        baseEuler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
        baseOrientation = self.p.getQuaternionFromEuler(baseEuler)
        # baseOrientation = [0, 0, 0, 1]    # fixed orientation

        # randomalize position
        pos = 0.1
        #basePosition = [random.uniform(1 * pos, pos), random.uniform(1 * pos, pos), random.uniform(0.1, 0.4)] 
        basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), random.uniform(0.1,0.4)] 
        # basePosition = [0.05, -0.1, 0.05] # fixed position 

        # load objects
        urdf_id = self.p.loadURDF(self.urdfs_filename[0], basePosition, baseOrientation)    

        # get xyz coordinates and scaling information
        inf = self.p.getVisualShapeData(urdf_id)[0]

        self.urdfs_id.append(urdf_id)
        self.urdfs_xyz.append(inf[5]) 
        self.urdfs_scale.append(inf[3][0]) 
    
    
    # load multiple objects
    def loadObjsInURDF(self, idx, num):
        """

        num: objects number 
        idx: starting id
            idx is negative randomly load the num of objects
            idx is non negative start to load the num of objects from the id
        """
        assert idx >= 0 and idx < len(self.urdfs_list)
        self.num_urdf = num

        # load objects file
        if (idx + self.num_urdf - 1) > (len(self.urdfs_list) - 1):     
            self.urdfs_filename = self.urdfs_list[idx:]
            self.urdfs_filename += self.urdfs_list[:2*self.num_urdf-len(self.urdfs_list)+idx]
            self.objs_id = list(range(idx, len(self.urdfs_list)))
            self.objs_id += list(range(self.num_urdf-len(self.urdfs_list)+idx))
        else:
            self.urdfs_filename = self.urdfs_list[idx:idx+self.num_urdf]
            self.objs_id = list(range(idx, idx+self.num_urdf))
        
        # print('self.urdfs_filename = \n', self.urdfs_filename)
        print('self.objs_id = \n', self.objs_id)

        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []
        for i in range(self.num_urdf):
            # randomlize position
            pos = 0.1
            #basePosition = [random.uniform(pos, 1), 0, random.uniform(0.2, 0.3)]
            basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), random.uniform(0.2,0.3)]  
            #basePosition = [0.5, 0, 0] # fixed position

            # randomlize orientation
            baseEuler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
            baseOrientation = self.p.getQuaternionFromEuler(baseEuler)
            # baseOrientation = [0, 0, 0, 1]    # fixed orientation
            
            # load objects
            urdf_id = self.p.loadURDF(self.urdfs_filename[i], basePosition, baseOrientation)    
            # allow collision between the obj and robot
            # default no collision
            if self.gripperId is not None:
                self.p.setCollisionFilterPair(urdf_id, self.gripperId, -1, 0, 1)
                self.p.setCollisionFilterPair(urdf_id, self.gripperId, -1, 1, 1)
                self.p.setCollisionFilterPair(urdf_id, self.gripperId, -1, 2, 1)

            # get xyz and scale info
            inf = self.p.getVisualShapeData(urdf_id)[0]

            self.urdfs_id.append(urdf_id)
            self.urdfs_xyz.append(inf[5]) 
            self.urdfs_scale.append(inf[3][0]) 
            
            t = 0
            while True:
                p.stepSimulation()
                t += 1
                if t == 120:
                    break


    def evalGrasp(self, z_thresh):
        """
        evaluate whether grasp succeed
        if z coordinate > z_thresh then successful grasp
        """
        for i in range(self.num_urdf):
            offset, _ =  self.p.getBasePositionAndOrientation(self.urdfs_id[i])
            if offset[2] >= z_thresh:
                return True
        print('!!!!!!!!!!!!!!!!!!!!! fail !!!!!!!!!!!!!!!!!!!!!')
        return False

    def evalGraspAndRemove(self, z_thresh):
        """
        evaluate whether grasp succeed and remove the grasped object
        """
        for i in range(self.num_urdf):
            offset, _ =  self.p.getBasePositionAndOrientation(self.urdfs_id[i])
            if offset[2] >= z_thresh:
                self.removeObjInURDF(i)
                return True
        print('!!!!!!!!!!!!!!!!!!!!! fail !!!!!!!!!!!!!!!!!!!!!')
        return False
    

    def resetObjsPoseRandom(self):
        """
        reset the object pose 
        path: object pose .mat file
        """
        # read .mat file
        for i in range(self.num_urdf):
            pos = 0.1
            basePosition = [random.uniform(-1 * pos, pos), random.uniform(-1 * pos, pos), random.uniform(0.3, 0.6)]
            baseEuler = [random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi), random.uniform(0, 2*math.pi)]
            baseOrientation = self.p.getQuaternionFromEuler(baseEuler)
            self.p.resetBasePositionAndOrientation(self.urdfs_id[i], basePosition, baseOrientation)

            t = 0
            while True:
                p.stepSimulation()
                t += 1
                if t == 120:
                    break


    def removeObjsInURDF(self):
        """
        remove all the objects
        """
        for i in range(self.num_urdf):
            self.p.removeBody(self.urdfs_id[i])
        self.num_urdf = 0
        self.urdfs_id = []
        self.urdfs_xyz = []
        self.urdfs_scale = []
        self.urdfs_filename = []
        self.objs_id = []

    def removeObjInURDF(self, i):
        """
        remove target object
        """
        self.num_urdf -= 1
        self.p.removeBody(self.urdfs_id[i])
        self.urdfs_id.pop(i)
        self.urdfs_xyz.pop(i)
        self.urdfs_scale.pop(i)
        self.urdfs_filename.pop(i)
        self.objs_id.pop(i)


    def renderCameraDepthImage(self):
        """
        rendering setting 
        """
        # rendering image
        img_camera = self.p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_camera[0]      # width of the image, in pixels
        h = img_camera[1]      # height of the image, in pixels
        dep = img_camera[3]    # depth data

        # depth image
        depth = np.reshape(dep, (h, w))  # [40:440, 120:520]
        A = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane * nearPlane
        B = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * farPlane
        C = np.ones((IMAGEHEIGHT, IMAGEWIDTH), dtype=np.float64) * (farPlane - nearPlane)
        # im_depthCamera = A / (B - C * depth)  # unit: m
        im_depthCamera = np.divide(A, (np.subtract(B, np.multiply(C, depth))))  # unit: m
        return im_depthCamera

    def renderCameraMask(self):
        """
        rendering setting 
        """
        # rendering image
        img_camera = self.p.getCameraImage(IMAGEWIDTH, IMAGEHEIGHT, self.viewMatrix, self.projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        w = img_camera[0]      # width of the image, in pixels
        h = img_camera[1]      # height of the image, in pixels
        # rgba = img_camera[2]    # color data RGB
        # dep = img_camera[3]    # depth data
        mask = img_camera[4]    # mask data

        # get mask image
        im_mask = np.reshape(mask, (h, w)).astype(np.uint8)
        im_mask[im_mask > 2] = 255
        return im_mask


    def gaussian_noise(self, im_depth):
        """
        add gaussian noise to images, refer to Dex-Net
        im_depth: float depth image unit :m
        """
        gamma_shape = 1000.00
        gamma_scale = 1 / gamma_shape
        gaussian_process_sigma = 0.002  # 0.002
        gaussian_process_scaling_factor = 8.0   # 8.0

        im_height, im_width = im_depth.shape
        
        # 1
        # mult_samples = ss.gamma.rvs(gamma_shape, scale=gamma_scale, size=1) # generate a random number close to 1，shape=(1,)
        # mult_samples = mult_samples[:, np.newaxis]
        # im_depth = im_depth * np.tile(mult_samples, [im_height, im_width])  # convert mult_samples to same size as camera_depth
        
        # 2
        gp_rescale_factor = gaussian_process_scaling_factor     # 4.0
        gp_sample_height = int(im_height / gp_rescale_factor)   # im_height / 4.0
        gp_sample_width = int(im_width / gp_rescale_factor)     # im_width / 4.0
        gp_num_pix = gp_sample_height * gp_sample_width     # im_height * im_width / 16.0
        gp_sigma = gaussian_process_sigma
        gp_noise = ss.norm.rvs(scale=gp_sigma, size=gp_num_pix).reshape(gp_sample_height, gp_sample_width)  # generate (mean is 0，variance is scale) gp_num_pix ，and reshape
        # print('maximum error:', gp_noise.max())
        gp_noise = imresize(gp_noise, gp_rescale_factor, interp="bicubic")  # resize，bicubic为双三次插值算法
        # gp_noise[gp_noise < 0] = 0
        # camera_depth[camera_depth > 0] += gp_noise[camera_depth > 0]
        im_depth += gp_noise

        return im_depth

    def add_noise(self, img):
        """
        add gaussian noise
        """
        img = self.gaussian_noise(img)    # add gaussian noise
        # inpainting image 
        # img = inpaint(img, missing_value=0)
        return img
