import os
import pybullet as p


path = './objects_model/objs/meshes'

p.connect(p.DIRECT)
files = os.listdir(path)
for file in files:
    print('processing ...', file)
    name_in = os.path.join(path, file)
    name_out = os.path.join(path, file.replace('.obj', '_col.obj'))
    name_log = "utils/collision_log.txt"
    p.vhacd(name_in, name_out, name_log)
