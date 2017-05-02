import cv2
from matcher import ImageMatcher
import numpy as np
from random import shuffle

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
surf = cv2.xfeatures2d.SURF_create()
bf = cv2.FlannBasedMatcher(index_params, search_params)


class BOWMatcher(ImageMatcher):
    def __init__(self):
        self.names = []
        self.vocab_size = 5000
        self.des = np.empty((0, self.vocab_size), float)
        self.BOW = cv2.BOWKMeansTrainer(self.vocab_size)
        self.bowDiction = cv2.BOWImgDescriptorExtractor(surf, bf)
        self.idf = np.zeros((1, self.vocab_size))

    def train(self, train_paths):
        count = 0
        shuffle(train_paths)
        for img_path in train_paths:
            img = cv2.imread(img_path, 0)
            self.names.append(img_path)
            # find the keypoints and descriptors with SIFT
            k, d = surf.detectAndCompute(img, None)
            self.BOW.add(d)
            count += 1
            print count
            # Randomly sample the first 50 images to train the BoW features
            if count > 50:
                break
        print "Training BOW"
        dictionary = self.BOW.cluster()
        print "Finished training BOW"
        self.bowDiction.setVocabulary(dictionary)
        for img_path in train_paths:
            img = cv2.imread(img_path, 0)
            d = self.bowDiction.compute(img, surf.detect(img))
            self.des = np.append(self.des, d, axis=0)
            self.idf += d
            count += 1

    def match_test_image(self, q_path, _):
        t_img = cv2.imread(q_path, 0)
        t_d = self.bowDiction.compute(t_img, surf.detect(t_img))
        results = []
        for i, d in zip(self.names, self.des):
            weighted_td = np.multiply(t_d, self.idf)
            similarity = np.dot(weighted_td, d.reshape(self.vocab_size, 1))
            results.append((i, similarity))
        return sorted(results, key=lambda x: -x[1])[:1]

    def debug_display(self, q_path, matches, threshold):
        # Draw matches between q_path and the top matched image.
        return
