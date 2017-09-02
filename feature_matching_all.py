#! /usr/bin/python3

'''
Sample Code for Feature Matching in OpenCV using different argument based Feature Descriptors like ORB, SIFT or SIFT with FLANN. (Features from Accelerated Segment Test) 

usage : python feature_matching_all.py [descriptor] [query_image] [train_image]
'''

import cv2
import numpy as np 
import os
import sys
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('descriptor_algo', help='Feature Descriptor Algo to be used any one of ORB, SIFT, FLANN')
    parser.add_argument('query_image', help='path of query image to be used')
    parser.add_argument('train_image', help='Path of train image to be used')
    args = parser.parse_args()

    descriptor = args.descriptor_algo 
    query_image = cv2.imread(args.query_image)
    train_image = cv2.imread(args.train_image)

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if descriptor == 'ORB':
        feature_matching_ORB(query_image, train_image)
    elif descriptor == 'SIFT':
        feature_matching_SIFT(query_image, train_image)
    elif descriptor == 'FLANN':
        feature_matching_FLANN(query_image, train_image)
    else:
        raise Exception('Entered Descriptor is not valid, please enter any one of (ORB, SIFT or FLANN)')

def feature_matching_ORB(query_image, train_image):

    # Create ORB object
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    query_kp, query_des = orb.detectAndCompute(query_image,None)
    train_kp, train_des = orb.detectAndCompute(train_image,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(query_des,train_des)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    result_img = cv2.drawMatches(query_image,query_kp,train_image,train_kp,matches[:100], flags=2, outImg = None)

    # Get Path and filename setup
    path = os.getcwd() + '/output/'
    file_name = os.path.basename(sys.argv[0])
    cv2.imwrite(str(path)+'{0}orb.jpg'.format(file_name),result_img)

    # Display the Image
    plt.imshow(result_img),plt.show()

def feature_matching_SIFT(query_image, train_image):
    #Create SIFT Object
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    query_kp, query_des = sift.detectAndCompute(query_image,None)
    train_kp, train_des = sift.detectAndCompute(train_image,None)

    # create BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(query_des,train_des, k = 2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    result_img = cv2.drawMatchesKnn(query_image,query_kp,train_image,train_kp,good,flags=2, outImg = None)

    # Get Path and filename setup
    path = os.getcwd() + '/output/'
    file_name = os.path.basename(sys.argv[0])
    cv2.imwrite(str(path)+'{0}sift.jpg'.format(file_name),result_img)

    # Display the Image
    plt.imshow(result_img),plt.show()

def feature_matching_FLANN(query_image, train_image):
    #Create SIFT Object
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    query_kp, query_des = sift.detectAndCompute(query_image,None)
    train_kp, train_des = sift.detectAndCompute(train_image,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(query_des,train_des,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)
    result_img = cv2.drawMatchesKnn(query_image,query_kp,train_image,train_kp,matches,None,**draw_params)

    # Get Path and filename setup
    path = os.getcwd() + '/output/'
    file_name = os.path.basename(sys.argv[0])
    cv2.imwrite(str(path)+'{0}flann.jpg'.format(file_name),result_img)

    plt.imshow(result_img)
    plt.show()

if __name__ == '__main__':
    main()