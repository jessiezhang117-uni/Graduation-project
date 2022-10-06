
import pybullet as p
from time import sleep
import pybullet_data


physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("./delta_drv90l_support/urdf/drv90l.urdf",useFixedBase=True)



useRealTimeSimulation = 0

if (useRealTimeSimulation):
  p.setRealTimeSimulation(1)

while 1:
  if (useRealTimeSimulation):
    p.setGravity(0, 0, -10)
    sleep(1000)  # Time in seconds.
  else:
    p.stepSimulation()