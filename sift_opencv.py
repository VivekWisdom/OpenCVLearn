#! /usr/bin/python3

'''
Scale Invariant Feature Transform (SIFT) OpenCV Sample script
'''

import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import os
import sys

def main():
    img = cv2.imread('sample.jpg')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    print(des)

    img=cv2.drawKeypoints(gray,kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Save the Annotated Image to output directory
    path = os.getcwd() + '/output/'
    file_name = os.path.basename(sys.argv[0])

    cv2.imwrite(str(path)+'{0}.jpg'.format(file_name),img)

if __name__ == '__main__':
    main()