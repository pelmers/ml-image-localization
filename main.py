from argparse import ArgumentParser
from contextlib import contextmanager
from glob import glob
from os.path import basename, join, splitext
from pprint import pprint
from time import time

import numpy as np
import matplotlib.pyplot as plt

from BFORBmatcher import BFORBMatcher

parser = ArgumentParser(description='Run some models.')
parser.add_argument('--query', help='path to query image, default will benchmark on all sortedtesting images', default=None)
parser.add_argument('--train_folder', help='folder holding all training images. default: sortedtraining', default='sortedtraining/')
parser.add_argument('--matchers', help='select matchers to use, comma-separated (default: use all models)', default='')
parser.add_argument('--detail', action='store_true', help='output additional detailed scoring breakdowns')
parser.add_argument('--debug', action='store_true', help='have matcher output debug info after each match')
parser.add_argument('--charts', action='store_true', help='should I show bar charts')

all_matchers = [BFORBMatcher]


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
    args = parser.parse_args()
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
        with timer("Training {} took %.3f seconds.".format(type(m).__name__)):
            m.train(train_paths)
        trained_matchers.append(m)
    queries = [args.query] if args.query else glob("sortedtesting/*.JPG")
    all_results = {}
    exp = expected_results(queries, train_paths)
    for m in trained_matchers:
        all_results[m] = {}
        with timer("{} queries with {} took %.3f seconds.".format(len(queries), type(m).__name__)):
            for q_path in queries:
                # matches is list of tuples (path, score)
                matches = m.match_test_image(q_path)
                top = matches[0] if len(matches) else None
                all_results[m][q_path] = top[0]
                if args.detail:
                    print q_path, matches
                elif top:
                    print "{} best matches {} with score {}".format(q_path, top[0], top[1])
                else:
                    print "{} no match found :(".format(q_path)
                print "{} expected match to {}".format(q_path, exp[q_path])
                if args.debug:
                    m.debug_display(q_path, matches)
                    if args.charts:
                        loc_dict = { loc_from_filename(t[0]): t[1] for t in matches }
                        loc_dict = { "{}, {}:{}".format(*k): v for k,v in loc_dict.iteritems() }
                        barchart_dict(loc_dict, title="Ranking matches to {}".format(q_path), to_sort=True, key_labels=True)

    # Now grade the results and show plots.
    accuracy = {m: sum(1.0/len(exp) for q in exp if exp[q] == all_results[m][q])
                for m in all_results}
    mses = {m: mse(exp, all_results[m])}
    if args.charts:
        barchart_class_dict(accuracy, "Accuracy")
        barchart_class_dict(mses, "Mean squared error")
    print "Accuracy", accuracy
    print "Mean squared error", mses
    for m in all_results:
        errs = deviation_over_expected(exp, all_results[m])
        if args.charts:
            barchart_dict(errs, title="{} per file result".format(type(m).__name__))
        print "{} per-file squared errors:".format(type(m).__name__)
        pprint(sorted(errs.items(), key=lambda i: errs[i[0]]))

