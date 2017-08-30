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
    img = cv2.imread('sample.jpg', 0)

    surf = cv2.xfeatures2d.SURF_create(400)
    kp = surf.detect(img,None)
    print(len(kp))
    img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

    # Save the Annotated Image to output directory
    path = 'C:/Users/VIVTRIPATHI/Desktop/Projects/OpenCVLearn/output/'
    file_name = os.path.basename(sys.argv[0])

    cv2.imwrite(str(path)+'{0}.jpg'.format(file_name),img)

    plt.imshow(img), plt.show()

if __name__ == '__main__':
    main()