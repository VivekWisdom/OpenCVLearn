#! /usr/bin/python3

'''
Sample code for harris corner detection
'''
import cv2
import numpy as np
import sys    
import os

def main():
    filename = 'sample.jpg'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    path = 'C:/Users/VIVTRIPATHI/Desktop/Projects/OpenCVLearn/output/'
    file_name = os.path.basename(sys.argv[0])
    cv2.imwrite(str(path)+'{0}.jpg'.format(file_name),img)
    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()