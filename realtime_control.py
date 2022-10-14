import pybullet as p
import time 
import pybullet_data
import math
from collections import namedtuple
from attrdict import AttrDict
from simEnv import SimEnv

physicsCLient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-9.81)

p.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])

flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("./delta_drv90l_support/urdf/drv90l.urdf",useFixedBase=True,flags=flags)



start_idx = 0      
objs_num = 5 
database_path = './objects_model/objs'
env = SimEnv(p, database_path,robotId)

env.loadObjsInURDF(start_idx,objs_num)


jointTypeList = ['REVOLUTE','PRISMATIC','SPHERICAL','PLANAR','FIXED']
numJoints = p.getNumJoints(robotId)
jointInfo = namedtuple('JointInfo',['id','name','type','lowrLimit','upperLimit','maxForce','maxVelocity'])
joints = AttrDict()

print(numJoints)
for i in range(numJoints):
    info = p.getJointInfo(robotId,i)
    jointID = info[0]
    jointName = info[1].decode('utf-8')
    jointType = jointTypeList[info[2]]
    jointLowerLimit = info[8]
    jointUpperLimit = info[9]
    jointMaxForce = info[10]
    jointMaxVelocity = info[11]
    singleInfo = jointInfo(jointID,jointName,jointType,jointLowerLimit,jointUpperLimit,jointMaxForce,jointMaxVelocity)
    joints[singleInfo.name] = singleInfo

print(joints)

position_control_group = []
position_control_group.append(p.addUserDebugParameter('joint_1',-2.96706,2.96706))
position_control_group.append(p.addUserDebugParameter('joint_2',-1.8326,2.35619))
position_control_group.append(p.addUserDebugParameter('joint_3',-3.57792,1.13446))
position_control_group.append(p.addUserDebugParameter('joint_4',-3.31613,3.31613))
position_control_group.append(p.addUserDebugParameter('joint_5',-2.0944,2.0944))
position_control_group.append(p.addUserDebugParameter('joint_6',-1.13446,1.13446))
position_control_group.append(p.addUserDebugParameter('finger_joint1',0,0.04))
position_control_group.append(p.addUserDebugParameter('finger_joint2',0,0.04))

position_control_joint_name = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6','finger_joint1','finger_joint2']
while True:
    time.sleep(0.01)
    parameter=[]
    for i in range(8):
        parameter.append(p.readUserDebugParameter(position_control_group[i]))
    num = 0
    #print("parameter: ",parameter)
    for jointName in joints:
        if jointName in position_control_joint_name:
            joint = joints[jointName]
            parameter_sim = parameter[num]
            p.setJointMotorControl2(robotId,joint.id,p.POSITION_CONTROL,targetPosition=parameter_sim)
            num = num +1



 
    p.stepSimulation()
