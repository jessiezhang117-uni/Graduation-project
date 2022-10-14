import time
import numpy as np
import math

useNullSpace = 1
ikSolver = 0
drvEndEffectorIndex = 11 #8
drvNumDofs = 7

ll = [-7]*drvNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*drvNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*drvNumDofs
# restposes for null space
jointPositions=(0.8045609285966308, 0.525471701354679, -0.02519566900946519, -1.3925086098003587, 0.013443782914225877, 1.9178323512245277, -0.007207024243406651, 0.01999436579245478, 0.019977024051412193)
            # [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

class DrvSim(object):
    def __init__(self, bullet_client, offset):
        self.p = bullet_client
        self.p.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)
        
        flags = self.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        orn=[0, 0, 0, 1]
        self.robotId = self.p.loadURDF("delta_drv90l_support/urdf/drv90l.urdf", np.array([0,0,0])+self.offset, orn, useFixedBase=True, flags=flags)
        index = 0
        self.state = 0
        self.control_dt = 1./240.
        self.finger_target = 0
        self.gripper_height = 0.2
        #create a constraint to keep the fingers centered
        c = self.p.createConstraint(self.robotId,
                          9,
                          self.robotId,
                          10,
                          jointType=self.p.JOINT_GEAR,
                          jointAxis=[1, 0, 0],
                          parentFramePosition=[0, 0, 0],
                          childFramePosition=[0, 0, 0])
        self.p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
    
        for j in range(self.p.getNumJoints(self.robotId)):
            self.p.changeDynamics(self.robotId, j, linearDamping=0, angularDamping=0)
            info = self.p.getJointInfo(self.robotId, j)
            #print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.p.JOINT_PRISMATIC):
                self.p.resetJointState(self.robotId, j, jointPositions[index]) 
                index=index+1

            if (jointType == self.p.JOINT_REVOLUTE):
                self.p.resetJointState(self.robotId, j, jointPositions[index]) 
                index=index+1
        self.t = 0.

    def calcJointLocation(self, pos, orn):
        jointPoses = self.p.calculateInverseKinematics(self.robotId, drvEndEffectorIndex, pos, orn, ll, ul, jr, rp, maxNumIterations=20)
        return jointPoses

    def setArmPos(self, pos):
        orn = self.p.getQuaternionFromEuler([math.pi,0.,math.pi / 2])   
        jointPoses = self.calcJointLocation(pos, orn)
        self.setArm(jointPoses)

    def setArm(self, jointPoses, maxVelocity=10):
        for i in range(drvNumDofs):   # 7
            self.p.setJointMotorControl2(self.robotId, i, self.p.POSITION_CONTROL, jointPoses[i], force=5 * 240., maxVelocity=maxVelocity)
    
    def setGripper(self, finger_target):
        for i in [9,10]:
            self.p.setJointMotorControl2(self.robotId, i, self.p.POSITION_CONTROL, finger_target, force=20)


    def step(self, pos, angle, gripper_w):
        """
        pos: [x, y, z] end effector
        angle: radians
        gripper_w: gripper width 
        """
        # update state
        self.update_state()
        
        pos[2] += 0.048
        if self.state == -1:
            pass

        elif self.state == 0:
            # print('RESET')
            pos[2] = 0.2
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])  
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            self.setGripper(gripper_w)
            return False

        elif self.state == 1:
            # print('GO ABOVE OBJECT')
            pos[2] += 0.05
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])  
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses)
            return False

        elif self.state == 2:
            # print('GRASPING')
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses, maxVelocity=3)
            return False

        elif self.state == 3:
            # print('CLOSE GRIPPER')
            self.setGripper(0)
            return False
        
        elif self.state == 4:
            # print('PLACE OBJECT')
            pos[2] += 0.05
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])  
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses, maxVelocity=0.5)
            # self.setGripper(0)
            return False
        
        elif self.state == 5:
            # print('GO ABOVE OBJECT')
            pos[2] = 0.3
            orn = self.p.getQuaternionFromEuler([math.pi,0.,angle + math.pi / 2])   
            jointPoses = self.calcJointLocation(pos, orn)
            self.setArm(jointPoses, maxVelocity=0.5)
            # self.setGripper(0)
            return False

        elif self.state == 12:
            self.reset()    # reset state
            return True


    def reset(self):
        """
        reset state
        """
        self.state = 0
        self.state_t = 0
        self.cur_state = 0


class DrvSimAuto(DrvSim):
    def __init__(self, bullet_client, offset):
        DrvSim.__init__(self, bullet_client, offset)
        self.state_t = 0
        self.cur_state = 0
        self.states = [0, 1, 2, 3, 4, 5, 12]
        self.state_durations = [1.0, 0.5, 2.0, 0.5, 1.0, 1.0, 0.5]
    
    def update_state(self):
        self.state_t += self.control_dt
        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state >= len(self.states):
                self.cur_state = 0
            self.state_t = 0
            self.state = self.states[self.cur_state]