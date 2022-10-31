import pybullet as p
import pybullet_data
import time
import math
import cv2
import os
import numpy as np
import sys
import scipy.io as scio
sys.path.append('/home/delta/Documents/Graduation_project_Jie/Simulation')
from simEnv import SimEnv
from utils import tool
import drv_sim_grasp as DrvSim
from utils.camera import Camera
from ggcnn_new_dataset.ggcnn import GGCNNNet,drawGrasps,drawRect,getGraspDepth

FINGER_L1 = 0.015
FINGER_L2 = 0.005

def run(database_path,start_idx,objs_num):
    cid = p.connect(p.GUI)
    robot = DrvSim.DrvSimAuto(p, [0, -0.6, 0])  # init grasping 
    env = SimEnv(p, database_path, robot.robotId) # init simulation env
    camera = Camera()
    # ggcnn = GGCNNNet('robotic-grasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97',device='cpu')

    success_grasp = 0
    sum_grasp = 0
    tt=5
    # load muptiple objects
    env.loadObjsInURDF(start_idx, objs_num)
    t = 0
    continue_fail = 0
    while True:
        # wait for the object to be stable
        for _ in range(240*5):
            p.stepSimulation()

        # render depth images
        camera_depth = env.renderCameraDepthImage()
        camera_depth = env.add_noise(camera_depth)

        # predict grasp
        row,col,grasp_angle,grasp_width_pixels = ggcnn.predict(camera_depth,input_size=300)
        grasp_width = camera.pixels_TO_length(grasp_width_pixels,camera_depth[row,col])

        grasp_x,grasp_y,grasp_z = camera.img2world([col,row],camera_depth[row,col])
        finger_l1_pixels = camera.length_TO_pixels(FINGER_L1,camera_depth[row,col])
        finger_l2_pixels = camera.length_TO_pixels(FINGER_L2,camera_depth[row,col])
        grasp_depth = getGraspDepth(camera_depth,row,col,grasp_angle,grasp_width_pixels,finger_l1_pixels,finger_l2_pixels)
        grasp_z = max(0.7-grasp_depth,0)

        print('*' * 100)
        print('grasp pose:')
        print('grasp_x = ', grasp_x)
        print('grasp_y = ', grasp_y)
        print('grasp_z = ', grasp_z)
        print('grasp_depth = ', grasp_depth)
        print('grasp_angle = ', grasp_angle)
        print('grasp_width = ', grasp_width)
        print('*' * 100)

        # show grasp points
        im_rgb = tool.depth2Gray3(camera_depth)
        im_grasp = drawGrasps(im_rgb,[[row,col,grasp_angle,grasp_width_pixels]],mode='line')
        cv2.imshow('im_grasp',im_grasp)
        cv2.waitKey(30)

        t = 0
        while True:
            p.stepSimulation()
            t +=1
            if t % tt==0:
                time.sleep(1./240.)

            if robot.step([grasp_x,grasp_y,grasp_z],grasp_angle,grasp_width/2):
                t=0
                break

        sum_grasp +=1
        if env.evalGraspAndRemove(z_thresh=0.2):
            success_grasp+=1
            continue_fail = 0
            if env.num_urdf == 0:
                p.disconnect()
                return success_grasp,sum_grasp
            
        else:
            continue_fail +=1
            if continue_fail == 5:
                p.disconnect()
                return success_grasp,sum_grasp
        
        robot.setArmPos([0.5,-0.6,0.2])

if __name__ == "__main__":
    start_idx = 0      
    objs_num = 5       
    database_path = '/home/delta/Documents/Graduation_project_Jie/Simulation/objects_model/objs'
    success_grasp,all_grasp = run(database_path, start_idx, objs_num)
    print('\n>>>>>>>>>>>>>>>>>>>> Success Rate: {}/{}={}'.format(success_grasp, all_grasp, success_grasp/all_grasp))     
    print('\n>>>>>>>>>>>>>>>>>>>> Percent Cleared: {}/{}={}'.format(success_grasp, objs_num, success_grasp/objs_num))  