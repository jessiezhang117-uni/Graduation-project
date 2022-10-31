import time
import numpy as np
import math
import pybullet_data

useNullSpace = 1
ikSolver = 0

jointPositions=(0.02, 0.02)
rp = jointPositions

class GripperSim(object):
    def __init__(self, bullet_client, position, friction):
        self.p = bullet_client
        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.p.setPhysicsEngineParameter(solverResidualThreshold=0)
        orn = self.p.getQuaternionFromEuler([math.pi, 0., math.radians(0) + math.pi / 2])  
        self.gripperId = self.p.loadURDF("delta_drv90l_support/urdf/gripper.urdf", position, orn, useFixedBase=True, flags=flags)
        self.p.changeDynamics(self.gripperId, 2, lateralFriction=friction)
        self.p.changeDynamics(self.gripperId, 3, lateralFriction=friction)
        index = 0
        self.state = 0
        self.control_dt = 1./240.
        self.finger_target = 0
        self.gripper_height = 0.2
        #create a constraint to keep the fingers centered
        c = self.p.createConstraint(self.gripperId,
                          1,
                          self.gripperId,
                          2,
                          jointType=self.p.JOINT_GEAR,
                          jointAxis=[1, 0, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
        self.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
    
        for j in range(self.p.getNumJoints(self.gripperId))[1:]:
            self.p.changeDynamics(self.gripperId, j, linearDamping=0, angularDamping=0)
            info = self.p.getJointInfo(self.gripperId, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.p.JOINT_PRISMATIC):
                self.p.resetJointState(self.gripperId, j, jointPositions[index]) 
                index=index+1
        self.t = 0.
        self.setGripper(0.04)

    def setArm(self, dist, maxVelocity=10):
        self.p.setJointMotorControl2(self.gripperId, 0, self.p.POSITION_CONTROL, dist, force=50., maxVelocity=maxVelocity)
    
    def setGripper(self, finger_target):
        self.p.setJointMotorControl2(self.gripperId, 1, self.p.POSITION_CONTROL, finger_target, force=10)
        self.p.setJointMotorControl2(self.gripperId, 2, self.p.POSITION_CONTROL, finger_target, force=10)

    def resetGripperPose(self, pos, angle, gripper_w):
        orn = self.p.getQuaternionFromEuler([math.pi, 0., angle + math.pi / 2]) 
        self.p.resetBasePositionAndOrientation(self.gripperId, pos, orn)
        self.setGripper(gripper_w)


    def step(self, dist):
        # update state
        self.update_state()
        
        if self.state == -1:
            pass

        elif self.state == 0:
            self.setArm(dist, maxVelocity=1)
            
            return False

        elif self.state == 1:
            self.setGripper(0)
            self.setArm(dist, maxVelocity=1)
            return False

        elif self.state == 2:
            self.setArm(0, maxVelocity=0.5)
            return False
        
        elif self.state == 103:
            self.reset()    # reset state
            return True


    def reset(self):
        """
        reset state
        """
        self.state = 0
        self.state_t = 0
        self.cur_state = 0

class GripperSimAuto(GripperSim):
    def __init__(self, bullet_client, position, friction=1.0):
        """
        mode: 运行模式
            pick: 只抓取
            pick_place: 抓取物体后放置到另一个托盘里
        """
        GripperSim.__init__(self, bullet_client, position, friction)
        self.state_t = 0
        self.cur_state = 0
        self.states = [0, 1, 2, 103]
        self.state_durations = [0.6, 0.4, 1, 0.1]
    
    def update_state(self):
        self.state_t += self.control_dt
        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state >= len(self.states):
                self.cur_state = 0
            self.state_t = 0
            self.state = self.states[self.cur_state]
            # print("self.state =", self.state)
