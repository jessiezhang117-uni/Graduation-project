import os
import glob
import random

path = './objects_model/objs'

files = glob.glob(os.path.join(path, '*', '*.urdf'))
random.shuffle(files)

txt = open(path + '/object_list.txt', 'w+')
for f in files:
    fname = os.path.basename(f)
    pre_fname = os.path.basename(os.path.dirname(f))
    txt.write(pre_fname + '/' + fname[:-5] + '\n')
txt.close()
print('done')
