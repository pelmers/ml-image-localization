import cv2
import numpy as np
from matcher import ImageMatcher, test_loose_visibility_constraint

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

class BFORBMatcher(ImageMatcher):

    def __init__(self):
        self.kp = {}    # (file_name)-> array of keypoints {location}
        self.des = {}   # file_name) -> array of descriptors {numeric vectors}
        self.threshold = 30

    def train(self, train_paths):
        for img_path in train_paths:
            img = cv2.imread(img_path, 0)
            # find the keypoints and descriptors with SIFT
            k, d = orb.detectAndCompute(img, None)
            self.kp[img_path] = k
            self.des[img_path] = d
        self.clean_images()

    def filter_indices(self, original, to_remove):
        return np.array([original[i] for i in range(len(original)) if i not in to_remove])

    # Returns Map[image_path] -> List[image_path] of non-matchable images
    def spatial_constraint_map(self):
        # Return map of location: [locations with which it could not share keypoints, not including self]
        # Assumes camera field of view is no more than 135 degrees.
        names = self.kp.keys()
        return {k1: filter(lambda k2: test_loose_visibility_constraint(k1, k2), names) for k1 in names}

    def clean_images(self):
        scm = self.spatial_constraint_map()
        count = 0
        for name in self.kp:
            for unmatchable in scm[name]:
                match = bf.match(self.des[name], self.des[unmatchable])
                bad_name_features = []
                bad_unmatchable_features = []
                for y in match:
                    if y.distance < self.threshold:
                        bad_name_features.append(y.queryIdx)
                        bad_unmatchable_features.append(y.trainIdx)
                        count += 1
                self.des[name] = self.filter_indices(self.des[name], bad_name_features)
                self.kp[name] = self.filter_indices(self.kp[name], bad_name_features)
                self.des[unmatchable] = self.filter_indices(self.des[unmatchable], bad_unmatchable_features)
                self.kp[unmatchable] = self.filter_indices(self.kp[unmatchable], bad_unmatchable_features)
        print "Removed " + str(count) + " features"

    def match_test_image(self, q_path, threshold=-1):
        t_img = cv2.imread(q_path, 0)
        t_k, t_d = orb.detectAndCompute(t_img, None)
        results = []
        for i, d in self.des.iteritems():
            match = bf.match(d, t_d)
            if (threshold < 0):
                results.append((i, sum(sorted([y.distance for y in match])[:abs(threshold)])))
            else:
                results.append((i, sum([128 - y.distance for y in match if y.distance < threshold])))
        const = -threshold / abs(threshold)
        return sorted(results, key=lambda x: const * x[1])[:10]

    def debug_display(self, q_path, matches, threshold):
        # Draw matches between q_path and the top matched image.
        img1 = cv2.imread(q_path)
        m_path = matches[0][0]
        img2 = cv2.imread(m_path)
        q_kp, q_descriptors = orb.detectAndCompute(img1, None)
        matches = bf.match(self.des[m_path], q_descriptors)
        cv2.drawMatches(img1, q_kp, img2, self.kp[m_path],
                        [y for y in matches if y.distance < threshold], flags=2)

