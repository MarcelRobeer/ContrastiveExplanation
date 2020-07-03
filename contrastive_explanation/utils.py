"""Utilities used by other modules.

Contains the following utilities:
- `Decorators`
- `Helper functions`
- `Custom prints`
- `Encoders`
    - `One Hot Encoder`
    - `Label Encoder`
"""

import functools
import inspect
import numpy as np
import pandas as pd
import os
import urllib.request

from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.tree import _tree
from scipy.sparse import coo_matrix


###############################
# Decorators
###############################


def cache(f):
    """Add cache to function."""
    cache = f.cache = {}

    @functools.wraps(f)
    def cached_f(*a, **kw):
        key = str(a) + str(kw)
        if key not in cache:
            cache[key] = f(*a, **kw)
        return cache[key]
    return cached_f


def checkargs(argchecks, checkfunc):
    """Check arguments of function.

    Args:
        argchecks: Arguments to check
        checkfunc: Function that evaluated to True if arguments
            are correct, else evaluates to False

    Returns:
        Executes function if arguments pass check

    Raises:
        ValueError: Argument is not in the correct range of values
    """
    def dec(f):
        def argchecked_f(*a, **kw):
            args = inspect.getargspec(f)[0]
            vals = inspect.getcallargs(f, *a, **kw)
            for (arg, val) in argchecks:
                if arg in args:
                    checkfunc(arg, vals[arg], val)
            return f(*a, **kw)
        return argchecked_f
    return dec


def check_stringvar(*argchecks):
    """Check a string variable."""
    def checkfunc(a, v_actual, v_target):
        if v_actual not in v_target:
            raise ValueError(f'Unknown {a} "{v_actual}", ' +
                             'should be one of "' +
                             '", "'.join(str(v) for v in v_target) +
                             '"')
    return checkargs(argchecks, checkfunc)


def check_relvar(*argchecks):
    """Check a relative variable."""
    def checkfunc(a, v_actual, v_target):
        if not eval(str(v_actual) + str(v_target)):
            raise ValueError(f'{a} should be {v_target} but is "{v_actual}"')
    return checkargs(argchecks, checkfunc)


###############################
# Helper functions
###############################


def show_image(img):
    """Show image using matplotlib."""
    if img is None:
        return
    if np.max(img) > 1:
        img = img.astype(int)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def softmax(a):
    """Apply softmax."""
    sums = a.sum(axis=0, keepdims=1)
    sums[sums == 0] = 1
    return a / sums


def rbf(d, sigma=0.1):
    """Apply radial basis function (RBF)."""
    return np.exp(-d ** 2 / (2 * sigma ** 2))


###############################
# Custom prints
###############################


def print_binary_tree(t, sample):
    """Print a binary tree and sample to a string."""
    fact_leaf = t.apply(sample)[0]
    print(sample[0])

    t_ = t.tree_
    print("def tree():")

    def recurse(node, depth):
        indent = "  " * depth
        if t_.feature[node] != _tree.TREE_UNDEFINED:
            name = t_.feature[node]
            threshold = t_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(t_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(t_.children_right[node], depth + 1)
        else:
            print("{}return {}{}".format(indent, np.argmax(t_.value[node]),
                                         ' (fact)' if node == fact_leaf else ''))

    recurse(0, 1)

###############################
# Data set downloader
###############################


def download_data(url, folder='data'):
    """Download external data from URL into folder.

    Args:
        url (str): URL to download from
        folder (str): Folder to save file to

    Returns:
        Location of saved file
    """
    url = url.strip()
    name = url.rsplit('/', 1)[-1]
    fname = os.path.join(folder, name)

    # Create folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Download file if it does not exist
    if not os.path.isfile(fname):
        try:
            urllib.request.urlretrieve(url, fname)
        except Exception as e:
            print(e)
            print(f'Error downloading {url}, please download manually')
            return None

    return fname


###############################
# One Hot Encoder
###############################

class Encoder:
    """Encoder class."""

    def __init__(self):
        """Encoder/decoder for a single categorical feature."""
        self.name2idx = None
        self.idx2name = None

    def fit(self, levels):
        """Get mapping for encoding and decoding."""
        self.name2idx = {x: i for i, x in enumerate(levels)}
        self.idx2name = {i: x for i, x in enumerate(levels)}

    def transform(self, column_data, inverse=False):
        """Apply encoding / decoding mapping to column.

        Args:
            column_data: Data to encode/decode
            inverse: If True decode, if False encode.

        Returns:
            Encoded/decoded column_data.
        """
        if inverse:
            if column_data.ndim == 1:
                return self.idx2name[np.argmax(column_data)]
            return np.vectorize(self.idx2name.get)(np.argmax(column_data, axis=1))
        else:
            row_cols = [(i, self.name2idx[x])
                        for i, x in enumerate(column_data) if x in self.name2idx]
            data = np.ones(len(row_cols)).astype(int)
            return coo_matrix((data, zip(*row_cols)),
                               shape=(column_data.shape[0], len(self)))

    def __eq__(self, other):
        """Check whether two encodings are the same."""
        return self.name2idx == other.name2idx

    def __len__(self):
        """Get length of sparse encoding."""
        if self.name2idx is None:
            return 0
        return len(self.name2idx.items())


###############################
# Label Encoder
###############################

class CustomLabelEncoder(LabelEncoder):
    """Custom LabelEncoder."""

    def __init__(self, to_encode):
        """Define a custom LabelEncoder for sklearn pipeline."""
        self.to_encode = to_encode
        self.encode_indices = []
        self.label_encoders = dict([(feature, LabelEncoder())
                                    for feature in to_encode])

    def fit(self, y):
        """Fit CustomLabelEncoder on data."""
        self.encode_indices = [y.columns.get_loc(c)
                               for c in self.to_encode]
        for feature in self.to_encode:
            self.label_encoders[feature].fit(y[feature])
        return self

    def fit_transform(self, y, *args, **kwargs):
        """Fit and transform CustomLabelEncoder."""
        return self.transform(y, *args, **kwargs)

    def transform(self, y, *args, **kwargs):
        """Transform data using CustomLabelEncoder."""
        y_ = y.copy()
        if type(y_) is pd.core.frame.DataFrame:
            for feature in self.to_encode:
                y_.loc[:, feature] = self.label_encoders[feature] \
                                         .transform(y_[feature])
        else:
            for i, feature in enumerate(self.encode_indices):
                y_[:, feature] = self.label_encoders[self.to_encode[i]] \
                                                     .transform(y_[:, feature])
        return y_
