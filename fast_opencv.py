#! /usr/bin/python3

'''
Sample Code for FAST (Features from Accelerated Segment Test) Corner Detector using OpenCV
'''

import cv2
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt

def main():
    img = cv2.imread('sample.jpg', 0)

    # Initiate FAST Object with Default Values
    fast = cv2.FastFeatureDetector_create()

    # Find and Draw the Keypoints
    kp = fast.detect(img,None)
    img2 = cv2.drawKeypoints(img, kp, None, color = (0, 255, 0))

    # Print all default params
    print("Threshold: ", fast.getThreshold())
    print("nonmaxSuppression: ", fast.getNonmaxSuppression())
    print("neighborhood: ", fast.getType())
    print("Total Keypoints with nonmaxSuppression: ", len(kp))

    # Get Path and filename setup
    path = os.getcwd() + '/output/'
    file_name = os.path.basename(sys.argv[0])

    cv2.imwrite(str(path)+'{0}1.jpg'.format(file_name),img2)

    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img,None)

    print("Total Keypoints without nonmaxSuppression: ", len(kp))

    img3 = cv2.drawKeypoints(img, kp,None, color=(0,0,255))

    cv2.imwrite(str(path)+'{0}2.jpg'.format(file_name),img3)


if __name__ == '__main__':
    main()