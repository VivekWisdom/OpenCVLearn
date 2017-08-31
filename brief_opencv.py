#! /usr/bin/python3

'''
Sample Code for BRIEF (Binary Robust Independent Elementry Features) Feature Descriptor with 
STAR (CenSure) Feature Detector using OpenCV 3.0
'''

import cv2
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt

def main():
    # Read the input image
    img = cv2.imread('sample.jpg',0)

    # Initiate STAR detector
    star = cv2.xfeatures2d.StarDetector_create()

    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # find the keypoints with STAR
    kp = star.detect(img,None)

    # compute the descriptors with BRIEF
    kp, des = brief.compute(img, kp)

    #Draw the Keypoints on Image

    img = cv2.drawKeypoints(img, kp, None, color = (0, 255, 0))

    print(brief.descriptorSize())
    print(des.shape)

    # Get Path and filename setup
    path = os.getcwd() + '/output/'
    file_name = os.path.basename(sys.argv[0])

    cv2.imwrite(str(path)+'{0}.jpg'.format(file_name),img)

if __name__ == '__main__':
    main()