from os.path import basename, splitext

def loc_from_filename(path):
    """Return x,y,o from filename.
    """
    f = basename(path)
    root, _ = splitext(f)
    x, y, o = root.split('_')
    return float(x), float(y), o

def test_loose_visibility_constraint(l1, l2):
    """Return whether l1 and l2 can NOT share keypoints.
    """
    x1, y1, o1 = loc_from_filename(l1)
    x2, y2, o2 = loc_from_filename(l2)
    ret = True
    if o1 in {'N', 'NE', 'NW'}:
        if y2 > y1:
            ret = False
        elif o2 in {'N', 'NE', 'NW'}:
            ret = False
    if o1 in {'E', 'NE', 'SE'}:
        if x2 > x1:
            ret = False
        elif o2 in {'E', 'NE', 'SE'}:
            ret = False
    if o1 in {'S', 'SE', 'SW'}:
        if y2 < y1:
            ret = False
        elif o2 in {'S', 'SE', 'SW'}:
            ret = False
    if o1 in {'W', 'SW', 'NW'}:
        if x2 < x1:
            ret = False
        elif o2 in {'W', 'SW', 'NW'}:
            ret = False
    return ret

class ImageMatcher(object):
    """Interface for image matching algorithms.
    """

    def train(self, train_paths):
        """Train the matcher object by from given training data.
        """
        pass

    def match_test_image(self, q_path):
        """Match query image at q_path against the model.
        Return list of results,
        where each result is pair (training image path, score)
        """
        pass

    def debug_display(self, q_path, matches):
        """Optional method. Display any helpful debugging information for the
        given query image that returned the provided matches. matches is the
        output of some previous call to match_test_image.
        """
        pass

    def __eq__(self, other):
        return self is other
