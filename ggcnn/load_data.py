import os
import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

cornell_path = "./ggcnn/cornell"
graspf = glob.glob(os.path.join(cornell_path,'*','pcd*cpos.txt'))
graspf.sort()

rgbf = [filename.replace('cpos.txt','r.png') for filename in graspf]
depthf = [filename.replace('cpos.txt','d.tiff') for filename in graspf]


def str2num(point):
    '''
    :convert string to numeric , return tuple 
    :point: string coordinates
    '''
    x,y = point.split()
    x,y = int(round(float(x))),int(round(float(y)))
    
    return (x,y)

def get_rectangles(cornell_grasp_file):
    '''
    :get rectangle coordinates from grasp file
    
    '''
    grasp_rectangles = []
    with open(cornell_grasp_file,'r') as f:
        while True:
            grasp_rectangle = []
            point0 = f.readline().strip()
            if not point0:
                break
            point1,point2,point3 = f.readline().strip(),f.readline().strip(),f.readline().strip()
            grasp_rectangle = [str2num(point0),
                               str2num(point1),
                               str2num(point2),
                               str2num(point3)]
            grasp_rectangles.append(grasp_rectangle)
    
    return grasp_rectangles

def draw_rectangles(img_path,grasp_path):
    '''
    :draw rectangle on image
    
    '''
    img_path = img_path
    grasp_path = grasp_path
    
    img = cv2.imread(img_path)
    grs = get_rectangles(grasp_path)
    
    for gr in grs:
        #generate random color
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        #draw the rectangle 
        for i in range(3): #four lines for one rectangle
            img = cv2.line(img,gr[i],gr[i+1],color,3)
        img = cv2.line(img,gr[3],gr[0],color,2) #the last close line
    
    plt.figure(figsize = (10,10))
    plt.imshow(img)
    plt.show()
    
    return img

if __name__ == "__main__":
    img = draw_rectangles(rgbf[0],graspf[0])
