from argparse import ArgumentParser
from contextlib import contextmanager
from glob import glob
import os
from os.path import basename, join, splitext
from pprint import pprint
from time import time

import numpy as np
import matplotlib.pyplot as plt

from BFORBmatcher import BFORBMatcher
from BFORBmatcher_fail import BFORBMatcher_fail
from TFMatcher import TFMatcher
from matcher import loc_from_filename

parser = ArgumentParser(description='Run some models.')
parser.add_argument('--query', help='path to query image, default will benchmark on all sortedtesting images', default=None)
parser.add_argument('--train_folder', help='folder holding all training images. default: sortedtraining', default='sortedtraining/')
parser.add_argument('--matchers', help='select matchers to use, comma-separated (default: use all models)', default='')
parser.add_argument('--detail', action='store_true', help='output additional detailed scoring breakdowns')
parser.add_argument('--debug', action='store_true', help='have matcher output debug info after each match')
parser.add_argument('--charts', action='store_true', help='should I show bar charts')
parser.add_argument('--threshold', default=-1)


all_matchers = [TFMatcher]


@contextmanager
def timer(message):
    start = time()
    yield
    end = time()
    print message % (end - start)

def expected_results(test_paths, train_paths, threshold=2):
    """Return map of {test_path: [equally closest train paths]}
    """
    # The expected answer is the closest points with same orientation within
    # threshold radius of the minimum.
    # Squared difference between locs q, p
    dist = lambda q, p: (q[0] - p[0])**2 + (q[1] - p[1])**2
    exp = {}
    train_locs = zip(train_paths, map(loc_from_filename, train_paths))
    for q_path in test_paths:
        q_loc = loc_from_filename(q_path)
        exp[q_path] = set([])
        train_distances = {}
        for t_path, t_loc in train_locs:
            train_distances[t_path] = dist(t_loc, q_loc)
        min_dist = min(train_distances.values())
        for t_path, distance in train_distances.iteritems():
            if distance < min_dist + threshold:
                exp[q_path].add(t_path)
    return exp


def deviation_over_expected(expect, actual):
    """Compute the difference between expected and actual squared error against
    the test set, per image.
    """
    err = {}
    for q, q_set in expect.iteritems():
        a_x, a_y, _ = loc_from_filename(actual[q])
        err[q] = float('inf')
        for e_path in q_set:
            e_x, e_y, _ = loc_from_filename(e_path)
            err[q] = min(err[q], ((a_x - e_x)**2 + (a_y - e_y)**2)**0.5)
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


def barchart_class_dict(d, title=""):
    """Show a bar chart using given dictionary as keys as above but display key
    labels with their type name instead.
    """
    barchart_dict({type(k).__name__: v for k, v in d.iteritems()}, title,
            key_labels=True)


if __name__ == '__main__':
    args = parser.parse_args()
    def the_thing():
        # 1. Train model (s)
        # 2a. Bench? Then try every test image.
        # 2b. Otherwise use provided image to test
        matchers_selected = args.matchers.split(',')
        use_matchers = [m for m in all_matchers
                        if type(m).__name__.lower().rstrip('matcher') in matchers_selected]
        if not len(use_matchers):
            use_matchers = all_matchers[:]

        trained_matchers = []
        train_paths = glob(join(args.train_folder, "*.JPG")) + glob(join(args.train_folder, "*.jpg"))
        for Matcher in use_matchers:
            m = Matcher()
            m.train(train_paths)
            trained_matchers.append(m)
        queries = [args.query] if args.query else glob("sortedtesting/*.JPG")
        all_results = {}
        exp = expected_results(queries, train_paths)
        threshold = int(args.threshold)
        for m in trained_matchers:
            all_results[m] = {}
            for q_path in queries:
                # matches is list of tuples (path, score)
                matches = m.match_test_image(q_path, threshold)
                top = matches[0] if len(matches) else None
                all_results[m][q_path] = top[0]
                if args.detail:
                    print q_path, matches
                if args.debug:
                    print "{} best matches {} with score {}".format(q_path, top[0], top[1])
                    print "{} expected match to any of {}".format(q_path, exp[q_path])
                    m.debug_display(q_path, matches, threshold)
                    if args.charts:
                        loc_dict = { loc_from_filename(t[0]): t[1] for t in matches }
                        loc_dict = { "{}, {}:{}".format(*k): v for k,v in loc_dict.iteritems() }
                        barchart_dict(loc_dict, title="Ranking matches to {}".format(q_path), to_sort=True, key_labels=True)

        # Now grade the results and show plots.
        accuracy = {m: sum(1.0/len(exp) for q in exp if all_results[m][q] in exp[q])
                    for m in all_results}
        mses = {m: mse(exp, all_results[m]) for m in all_results}
        if args.charts:
            barchart_class_dict(accuracy, "Accuracy")
            plt.figure()
            barchart_class_dict(mses, "Mean squared error")
        if args.detail:
            print "Accuracy", accuracy
            for m in all_results:
                errs = deviation_over_expected(exp, all_results[m])
                if args.charts:
                    plt.figure()
                    barchart_dict(errs, title="{} per file result".format(type(m).__name__))
                print "{} per-file squared errors:".format(type(m).__name__)
                pprint(sorted(errs.items(), key=lambda i: errs[i[0]]))
            print "Mean squared error", mses
            for m in all_results:
                print m, "Median", sorted(errs.values())[len(all_results[m]) / 2]
        if args.charts:
            plt.show()
        return mses[m]

    min_mse = float('inf')
    best_params = None
    for radius in range(0, 40, 5):
        for threshold in range(0, 40, 5):
            os.putenv("RADIUS", str(radius))
            os.putenv("THRESH", str(threshold))
            m = the_thing()
            print radius, threshold, m
            if m < min_mse:
                min_mse = mse
                best_params = (radius, threshold)
    print "I am the best", min_mse
    print "I am the best radius and athrrehsold", best_params
