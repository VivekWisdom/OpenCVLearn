#! /usr/bin/python3

'''
Sample Code for ORB (Oriented FAST and Rotated BRIEF) Feature Detector using OpenCV
'''

import cv2
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt

def main():
    img = cv2.imread('sample.jpg',0)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img,kp,None, color=(0,255,0), flags=0)
    plt.imshow(img2),plt.show()

     # Get Path and filename setup
    path = os.getcwd() + '/output/'
    file_name = os.path.basename(sys.argv[0])

    cv2.imwrite(str(path)+'{0}.jpg'.format(file_name),img2)


if __name__ == '__main__':
    main()