import pybullet as p
import pybullet_data
import time
import math
import cv2
import os
import numpy as np
import sys
import scipy.io as scio
from pybullet_grasp.simEnv import SimEnv
import utils.tool as tool
import pybullet_grasp.drv_sim_grasp as DrvSim
from utils.camera import Camera


def run(database_path, start_idx, objs_num):
    cid = p.connect(p.GUI)  
    robot = DrvSim.DrvSimAuto(p, [0, -0.6, 0])  # init grasping 
    env = SimEnv(p, database_path, robot.robotId) # init simulation env

    tt = 1
    # load muptiple objects
    env.loadObjsInURDF(start_idx, objs_num)
    t = 0
    continue_fail = 0
    while True:
        # wait for the object to be stable
        for _ in range(240*5):
            p.stepSimulation()

        grasp_x, grasp_y, grasp_z, grasp_angle, grasp_width = (0, 0, 0.01, 0, 0.08)

        # start grasping 
        t = 0
        while True:
            p.stepSimulation()
            t += 1
            if t % tt == 0:
                time.sleep(1./240.)
            
            if robot.step([grasp_x, grasp_y, grasp_z], grasp_angle, grasp_width/2):
                t = 0
                break

        # check all the objects, if it's within the target area then grasp succeed 
        if env.evalGraspAndRemove(z_thresh=0.2):
            if env.num_urdf == 0:
                p.disconnect()
                return
        
        robot.setArmPos([0.5, -0.6, 0.2])


if __name__ == "__main__":
    start_idx = 0      
    objs_num = 5       
    database_path = '/home/delta/Documents/Graduation_project_Jie/Simulation/objects_model/objs'
    run(database_path, start_idx, objs_num)