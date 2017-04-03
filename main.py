from contextlib import contextmanager
from glob import glob
from os.path import basename, join, splitext
from pprint import pprint
from time import time

import numpy as np
import matplotlib.pyplot as plt
import cv2


@contextmanager
def timer(message):
    start = time()
    yield
    end = time()
    print message % (end - start)


def loc_from_filename(path):
    """Return x,y,o from filename.
    """
    f = basename(path)
    root, _ = splitext(f)
    x, y, o = root.split('_')
    return float(x), float(y), o


def test_visibility_constraint(l1, l2):
    """Return whether l1 and l2 can NOT share keypoints.
    """
    x1, y1, o1 = l1
    x2, y2, o2 = l2
    ret = True
    if o1 in {'N', 'NE', 'NW'}:
        if y2 > y1:
            ret = False
        elif o2 in {'N', 'NE', 'NW'}:
            ret = False
    if o1 in {'E', 'NE', 'SE'}:
        if y2 > y1:
            ret = False
        elif o2 in {'N', 'NE', 'NW'}:
            ret = False
    if o1 in {'S', 'SE', 'SW'}:
        if y2 > y1:
            ret = False
        elif o2 in {'N', 'NE', 'NW'}:
            ret = False
    if o1 in {'W', 'SW', 'NW'}:
        if y2 > y1:
            ret = False
        elif o2 in {'W', 'SW', 'NW'}:
            ret = False
    return ret


def spatial_constraint_map(locations):
    # Return map of location: [locations with which it could not share keypoints, not including self]
    # Assumes camera field of view is no more than 135 degrees.
    return {k1: filter(lambda k2: test_visibility_constraint(k1, k2), locations) for k1 in locations}


def expected_results(test_paths, train_paths):
    # The expected answer is the closest point with same orientation.
    exp = {}
    train_locs = zip(train_paths, map(loc_from_filename, train_paths))
    for q in test_paths:
        q_x, q_y, q_o = loc_from_filename(q)
        exp[q] = min([(p, p_x, p_y) for (p, (p_x, p_y, p_o)) in train_locs if p_o == q_o],
                     key=lambda (_, p_x, p_y): (q_x - p_x)**2 + (q_y - p_y)**2)[0]
    return exp


def deviation_over_expected(expect, actual):
    """Compute the difference between expected and actual squared error against
    the test set, per image.
    """
    err = {}
    for q in expect:
        q_x, q_y, _ = loc_from_filename(q)
        e_x, e_y, _ = loc_from_filename(expect[q])
        a_x, a_y, _ = loc_from_filename(actual[q])
        err[q] = (q_x - a_x)**2 + (q_y - a_y)**2 - (q_x - e_x)**2 - (q_y - e_y)**2
    return err


def mse(expect, actual):
    """Return the mean across the deviation over expected for provided set of
    results.
    """
    return float(sum(deviation_over_expected(expect, actual).values())) / len(expect)


def barchart_dict(d, title="", to_sort=False, key_labels=False):
    """Show a bar chart using given dictionary key, values as x-y axis. If
    to_sort, then sort keys by ascending value. If key_labels, then label the
    x-axis using key strings.
    """
    x = d.keys()
    if to_sort:
        x = sorted(x, key=lambda k: d[k])
    y = [d[k] for k in x]
    x_pos = np.arange(len(x))
    plt.bar(x_pos, y, align='center', color='#66c2a5', alpha=0.6)
    if key_labels:
        plt.xticks(x_pos, x)
    plt.title(title)
    plt.show()


def barchart_class_dict(d, title=""):
    """Show a bar chart using given dictionary as keys as above but display key
    labels with their type name instead.
    """
    barchart_dict({type(k).__name__: v for k, v in d.iteritems()}, title,
            key_labels=True)


if __name__ == '__main__':
    train_folder = "sortedtraining"
    train_paths = glob(join(train_folder, "*.JPG")) + glob(join(train_folder, "*.jpg"))
    train_locs = map(loc_from_filename, train_paths)
    pprint(spatial_constraint_map(train_locs))
    # Find the features for every image.
    # If some feature matches well against the constraint set then I would want to lower it (?).
    brisk = cv2.BRISK()
    img_brisks = {path : brisk.detectAndCompute(cv2.imread(path), None) for path in train_paths}
    pprint(img_brisks[train_paths[1]])
    print train_paths[1]

