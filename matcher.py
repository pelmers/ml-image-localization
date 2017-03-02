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
