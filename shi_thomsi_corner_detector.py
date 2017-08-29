#! /usr/bin/python3

'''
Shi Thomsi Corner Detector with good features predictor
'''
import os
import sys
import cv2
import numpy as np 
import matplotlib.pyplot as plt 


def main():
    img = cv2.imread('sample.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)

    path = 'C:/Users/VIVTRIPATHI/Desktop/Projects/OpenCVLearn/output/'
    file_name = os.path.basename(sys.argv[0])
    cv2.imwrite(str(path)+'{0}.jpg'.format(file_name),img)

    plt.imshow(img),plt.show()

if __name__ == '__main__':
    main()
