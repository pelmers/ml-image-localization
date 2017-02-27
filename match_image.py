#!/usr/bin/env python

import cv2
import sys
import os
from glob import glob
from pprint import pprint

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: match_image.py <test image path>"
    # Initiate SIFT detector
    orb = cv2.ORB()
    # keypoint and descriptor lists of training data
    names = []
    kp = []
    des = []
    t_path = sys.argv[1]
    for img_path in glob("sortedtraining/*.JPG"):
        if img_path == t_path:
            continue
        img = cv2.imread(img_path, 0)
        names.append(img_path)
        # find the keypoints and descriptors with SIFT
        k, d = orb.detectAndCompute(img, None)
        kp.append(k)
        des.append(d)

    # print best match
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # compute test picture keypoint, descriptor
    t_img = cv2.imread(t_path, 0)
    t_k, t_d = orb.detectAndCompute(t_img, None)
    m = float('inf')
    results = []
    for i, k, d in zip(names, kp, des):
        match = bf.match(d, t_d)
        results.append((i, sum(sorted([y.distance for y in match])[:10])))
    top10 = sorted(results, key=lambda x: x[1])[:10]
    pprint(top10)
    try:
        os.system("open %s" % top10[0][0])
        os.system("open %s" % t_path)
    except:
        pass
