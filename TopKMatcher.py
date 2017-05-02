import cv2
from matcher import ImageMatcher

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

class TopKMatcher(ImageMatcher):

    def __init__(self):
        self.names = []
        self.kp = []
        self.des = []

    def train(self, train_paths):
        for img_path in train_paths:
            img = cv2.imread(img_path, 0)
            self.names.append(img_path)
            # find the keypoints and descriptors with SIFT
            k, d = orb.detectAndCompute(img, None)
            self.kp.append(k)
            self.des.append(d)

    def match_test_image(self, q_path, _):
        t_img = cv2.imread(q_path, 0)
        t_k, t_d = orb.detectAndCompute(t_img, None)
        results = []
        for i, d in zip(self.names, self.des):
            match = bf.match(d, t_d)
            results.append((i, sum(sorted([y.distance for y in match])[:10])))
        return sorted(results, key=lambda x: x[1])[:10]

    def debug_display(self, q_path, matches, threshold):
        # Draw matches between q_path and the top matched image.
        return
