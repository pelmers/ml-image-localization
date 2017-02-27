#!/usr/bin/env python

import cv2
import sys
import os
from glob import glob
from os.path import basename, splitext
from pprint import pprint

# Initiate SIFT detector
orb = cv2.ORB()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def match_test_image(t_path, names, des):
    '''Return top 10 matches with test image from training set.
    '''
    # compute test picture keypoint, descriptor
    t_img = cv2.imread(t_path, 0)
    t_k, t_d = orb.detectAndCompute(t_img, None)
    m = float('inf')
    results = []
    for i, d in zip(names, des):
        match = bf.match(d, t_d)
        results.append((i, sum(sorted([y.distance for y in match])[:10])))
    return sorted(results, key=lambda x: x[1])[:10]

def loc_from_filename(path):
    """Return x,y,o from filename.
    """
    f = basename(path)
    root, _ = splitext(f)
    x, y, o = root.split('_')
    return float(x), float(y), o

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Usage: match_image.py <test image paths>"
    # keypoint and descriptor lists of training data
    names = []
    kp = []
    des = []
    t_paths = sys.argv[1:]
    for img_path in glob("sortedtraining/*.JPG"):
        if img_path in t_paths:
            continue
        img = cv2.imread(img_path, 0)
        names.append(img_path)
        # find the keypoints and descriptors with SIFT
        k, d = orb.detectAndCompute(img, None)
        kp.append(k)
        des.append(d)

    success = 0
    for t_path in t_paths:
        t_x, t_y, t_o = loc_from_filename(t_path)
        top10 = match_test_image(t_path, names, des)
        #print t_path, top10
        m_x, m_y, m_o = loc_from_filename(top10[0][0])
        dist = ((t_x - m_x)**2 + (t_y - m_y)**2)**0.5
        if dist > 9:
            print t_path, " matched too far from ", top10[0][0]
            pprint(top10)
        else:
            success += 1
        if len(t_paths) == 1:
            try:
                os.system("open %s" % top10[0][0])
                os.system("open %s" % t_paths[0])
            except:
                pass
    print "Success rate", 100*success / float(len(t_paths))
