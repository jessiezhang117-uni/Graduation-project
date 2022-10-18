import cv2 as cv
import glob
import os
import copy
import time
import math
import numpy as np

path = './dataset/cornell'  # dataset path
savepath = './dataset/cornell_show'
label_win_name = 'labeling'
label_files = glob.glob(os.path.join(path,'pcd*Label.txt'))

for labelfile in label_files:
    pngfile = labelfile.replace('Label.txt','r.png')
    if not os.path.exists(pngfile):
        continue

    im = cv.imread(pngfile)

    f = open(labelfile)
    points = f.readlines()
    f.close()

    n = 0
    for point in points:
        point_data = point.split(' ')
        y = int(float(point_data[0]))
        x = int(float(point_data[1]))
        w = float(point_data[-2])/2
        n +=1
        if n % 10 ==0:
            if len(point_data) ==4:
                # circle
                cv.circle(im,(x,y),int(w),(0,255,0),1)

            elif len(point_data) == 5:
                # one side grasp
                angle = float(point_data[2])
                k = math.tan(angle)

                if k==0:
                    dx = w
                    dy = 0
                else:
                    dx = k/abs(k)*w/pow(k**2+1,0.5)
                    dy =k*dx
                if angle <math.pi:
                    cv.line(im,(x,y),(int(x+dx),int(y-dy)),(0,255,0),1)
                else:
                    cv.line(im,(x,y),(int(x-dx),int(y+dy)),(0,255,0),1)
            
            elif len(point_data) == 6:
                # two sides grasp
                angle1 = float(point_data[2])
                angle2 = float(point_data[3])
                k =math.tan(angle1)

                if k ==0:
                    dx=w
                    dy=0
                else:
                    dx = k/abs(k)*w/pow(k**2+1,0.5)
                    dy =k*dx
                
                if angle1<math.pi:
                    cv.line(im,(x,y),(int(x+dx),int(y-dy)),(0,255,0),1)
                else:
                    cv.line(im,(x,y),(int(x-dx),int(y+dy)),(0,255,0),1)
                
                if angle2<math.pi:
                    cv.line(im,(x,y),(int(x+dx),int(y-dy)),(0,255,0),1)
                else:
                    cv.line(im,(x,y),(int(x-dx),int(y+dy)),(0,255,0),1)
                
        cv.circle(im,(x,y),1,(255,0,0),-1)

    savefile = os.path.join(savepath,pngfile.split('pcd')[-1])
    cv.imwrite(savefile,im)
    print('generate {}'.format(savefile))

    # cv.imshow('im',im)
    # key = cv.waitKeyEx()
    # if key==27:
    #     cv.destroyAllWindows()
    #     break