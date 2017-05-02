import cv2
import numpy as np
from matcher import ImageMatcher
from collections import defaultdict

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
match_threshold = 35

class CountMatcher(ImageMatcher):

    def __init__(self):
        self.kp = {}
        self.des = {}
        self.feature_frequency = defaultdict(int)
        self.feature_strs = {}
        # Big matrix of all the features in self.feature_strs, each feature as a row.
        self.all_features = None

    def train(self, train_paths):
        for img_path in train_paths:
            img = cv2.imread(img_path, 0)
            # find the keypoints and descriptors with SIFT
            k, d = orb.detectAndCompute(img, None)
            if self.all_features is None:
                self.all_features = d
            else:
                self.all_features = np.append(self.all_features, d, axis=0)
            self.kp[img_path] = k
            self.des[img_path] = d

    def match_test_image(self, q_path, _):
        t_img = cv2.imread(q_path, 0)
        t_k, t_d = orb.detectAndCompute(t_img, None)
        results = []
        for i, d in self.des.iteritems():
            match = bf.match(t_d, d)
            weighted_matches = []
            for y in match:
                if y.distance < match_threshold:
                    weighted_matches.append(1)
            results.append((i, sum(weighted_matches)))
        return sorted(results, key=lambda x: -x[1])[:10]


    def debug_display(self, q_path, matches, threshold):
        # Draw matches between q_path and the top matched image.
        return
