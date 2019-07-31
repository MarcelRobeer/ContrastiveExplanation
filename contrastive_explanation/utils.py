import functools
import inspect
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import _tree
from scipy.sparse import coo_matrix, hstack
import inspect


###############################
# Decorators
###############################


def cache(f):
    cache = f.cache = {}

    @functools.wraps(f)
    def cached_f(*a, **kw):
        key = str(a) + str(kw)
        if key not in cache:
            cache[key] = f(*a, **kw)
        return cache[key]
    return cached_f


def checkargs(argchecks, checkfunc):
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
    def checkfunc(a, v_actual, v_target):
        if v_actual not in v_target:
            raise ValueError(f'Unknown {a} "{v_actual}", ' +
                             f'should be one of "' +
                             f'", "'.join(str(v) for v in v_target) +
                             f'"')
    return checkargs(argchecks, checkfunc)


def check_relvar(*argchecks):
    def checkfunc(a, v_actual, v_target):
        if not eval(str(v_actual) + str(v_target)):
            raise ValueError(f'{a} should be {v_target} but is "{v_actual}"')
    return checkargs(argchecks, checkfunc)


###############################
# Helper functions
###############################


def show_image(img):
    if img is None:
        return
    if np.max(img) > 1:
        img = img.astype(int)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def softmax(a):
    sums = a.sum(axis=0, keepdims=1)
    sums[sums == 0] = 1
    return a / sums


def rbf(d, sigma=0.1):
    return np.exp(-d ** 2 / (2 * sigma ** 2))


###############################
# Custom prints
###############################


def print_binary_tree(t, sample):
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
# One Hot Encoder
###############################

class Encoder():
    def __init__(self):
        self.name2idx = None
        self.idx2name = None
    
    def fit(self, levels):
        self.name2idx = {x: i for i, x in enumerate(levels)}
        self.idx2name = {i: x for i, x in enumerate(levels)}
    
    def transform(self, column_data, inverse=False):
        if inverse:
            if column_data.ndim == 1:
                return self.idx2name[np.argmax(column_data)]
            return np.vectorize(self.idx2name.get)(np.argmax(column_data, axis=1))
        else:
            row_cols = [(i, self.name2idx[x])
                        for i,x in enumerate(column_data) if x in self.name2idx]
            data = np.ones(len(row_cols)).astype(int)
            return(coo_matrix((data, zip(*row_cols)),
                            shape=(column_data.shape[0], len(self))))
    
    def __eq__(self, other):
        return self.name2idx == other.name2idx
    
    def __len__(self):
        if self.name2idx is None:
            return 0
        return len(self.name2idx.items())