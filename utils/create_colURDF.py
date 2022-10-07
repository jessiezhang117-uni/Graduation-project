import os
import glob
import math
from random import choice
import sys
from mesh import Mesh


ss = '<?xml version="0.0" ?> \n' \
'<robot name="cube.urdf"> \n' \
  '<link name="legobrick"> \n' \
    '<contact> \n' \
      '<lateral_friction value="1.0"/>\n' \
      '<rolling_friction value="0.0"/>\n' \
      '<contact_cfm value="0.0"/>\n' \
      '<contact_erp value="1.0"/>\n' \
    '</contact> \n' \
    '<inertial> \n' \
      '<origin rpy="0 0 0" xyz="0.0 0.0 0.0"/> \n' \
       '<mass value="0.1"/> \n' \
       '<inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/> \n' \
    '</inertial> \n' \
    '<visual> \n' \
      '<origin rpy="1.570796 0 0" xyz="-0.03 0.03 -0.03"/> \n' \
      '<geometry> \n' \
        '<mesh filename="ball.obj" scale="0.00075 0.00075 0.00075"/> \n' \
      '</geometry> \n' \
       '<material name="yellow"> \n' \
        '<color rgba="0.6 0.6 0.6 1"/> \n' \
      '</material> \n' \
    '</visual> \n' \
    '<collision> \n' \
      '<origin rpy="1.570796 0 0" xyz="-0.03 0.03 -0.03"/> \n' \
      '<geometry> \n' \
	 		'<mesh filename="ball_col.obj" scale="0.00075 0.00075 0.00075"/> \n' \
      '</geometry> \n' \
    '</collision> \n' \
  '</link> \n' \
'</robot>'


def run():
    path = './objects_model/objs/meshes'

    objs_path = glob.glob(os.path.join(path, '*.obj'))
    for obj_path in objs_path:
      if obj_path.endswith('col.obj'):
          continue
      print('processing ... ', obj_path)
      # Determine the scale
      # scale = choice([1.0,  1.1, 1.2, 1.3])
      # scale = 0.002
      # scale = -1
      scale = 1.2

      # Read .obj files
      mesh = Mesh(obj_path, scale)
      pt = mesh.calcCenterPt()  # [x, y, z]
      scale = mesh.scale()

      sita = math.atan2(pt[2], pt[1])
      l = (pt[2] ** 2 + pt[1] ** 2) ** 0.5
      sita += math.pi / 2
      pt[2] = l * math.sin(sita)
      pt[1] = l * math.cos(sita)

      xyz = list(pt * -1)
      xyz = str(xyz[0]) + ' ' + str(xyz[1]) + ' ' + str(xyz[2])
      scale3 = str(scale) + ' ' + str(scale) + ' ' + str(scale)

      # ========================= Generate URDF files =========================
      # get obj name
      obj_name = os.path.basename(obj_path)
      # change URDF file content
      ss_ = ss.replace('ball.obj', obj_name)  # replace obj file name
      ss_ = ss_.replace('ball_col.obj', obj_name.replace('.obj', '_col.obj'))  
      ss_ = ss_.replace('-0.03 0.03 -0.03', xyz)
      ss_ = ss_.replace('0.00075 0.00075 0.00075', scale3)
      # URDF path
      urdf_path = obj_path.replace('.obj', '.urdf')
      f = open(urdf_path, 'w')
      f.write(ss_)
      f.close()
    
    print('Finished!')




if __name__ == "__main__":
    run()