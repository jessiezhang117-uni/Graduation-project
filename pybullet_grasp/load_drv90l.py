
import pybullet as p
from time import sleep
import pybullet_data
from simEnv import SimEnv


physicsClient = p.connect(p.GUI)

# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# p.setGravity(0, 0, -10)
# planeId = p.loadURDF("plane.urdf")
flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
robotId = p.loadURDF("./delta_drv90l_support/urdf/drv90l.urdf",useFixedBase=True,flags=flags)



# useRealTimeSimulation = 0

# if (useRealTimeSimulation):
#   p.setRealTimeSimulation(1)

# while 1:
#   if (useRealTimeSimulation):
#     p.setGravity(0, 0, -10)
#     sleep(1000)  # Time in seconds.
#   else:
#     p.stepSimulation()
start_idx = 0      
objs_num = 5 
database_path = '/home/delta/Documents/Graduation_project_Jie/Simulation/objects_model/objs'
env = SimEnv(p, database_path,robotId)

env.loadObjsInURDF(start_idx,objs_num)


useRealTimeSimulation = 0

if (useRealTimeSimulation):
  p.setRealTimeSimulation(1)

while 1:
  if (useRealTimeSimulation):
    sleep(1000)  # Time in seconds.
  else:
    p.stepSimulation()