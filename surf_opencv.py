#! /usr/bin/python3

'''
Speeded Up Robust Features(SURF) OpenCV Sample script
'''

import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import os
import sys

def main():
    img = cv2.imread('sample.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(5000)
    kp, des = surf.detectAndCompute(gray,None)
    img2 = cv2.drawKeypoints(gray,kp,img,(255,0,0),4)

    # Save the Annotated Image to output directory
    path = os.getcwd() + '/output/'
    file_name = os.path.basename(sys.argv[0])

    cv2.imwrite(str(path)+'{0}.jpg'.format(file_name),img2)

    plt.imshow(img2), plt.show()

if __name__ == '__main__':
    main()