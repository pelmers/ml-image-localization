from os.path import basename, splitext

def loc_from_filename(path):
    """Return x,y,o from filename.
    """
    f = basename(path)
    root, _ = splitext(f)
    x, y, o = root.split('_')
    return float(x), float(y), o

class ImageMatcher(object):

    def train(self, train_paths):
        pass

    def match_test_image(self, q_path):
        pass

    def debug_display(self, q_path, matches):
        pass

    def __eq__(self, other):
        return self is other
