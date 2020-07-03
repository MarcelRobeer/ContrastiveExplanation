"""DomainMappers map from a domain (e.g. tabular or images) into a generic tabular
format for the Explainer to use, and mapping the explanation back into
this domain.
"""

import pandas as pd
import numpy as np
import sklearn
import warnings
import itertools

from sklearn.utils import check_random_state
from scipy.sparse import coo_matrix, hstack

from .rules import Literal, Operator
from .utils import cache, check_stringvar, show_image, rbf, Encoder

# Suppress FutureWarning of sklearn
warnings.simplefilter(action='ignore', category=FutureWarning)


class DomainMapper:
    """General DomainMapper class."""

    def __init__(self,
                 train_data,
                 contrast_names=None,
                 kernel_width=.25,
                 seed=1):
        """Init.

        Args:
            train_data: Original data, to obtain input distributions
                for generating neighborhood data
            contrast_names (list/dict): Names of contrasts (including fact
                and foil)
            seed (int): Seed for random functions
        """
        self.train_data = train_data
        self.kernel_fn = lambda d: rbf(d, sigma=kernel_width)

        if type(contrast_names) is list:
            contrast_names = dict(enumerate(contrast_names))
        self.contrast_class = contrast_names
        self.contrast_map = None
        if type(self.contrast_class) is dict:
            self.contrast_map = [*self.contrast_class]

        self.seed_ = seed
        self.seed = check_random_state(seed)

    def map_contrast_names(self,
                           contrast,
                           inverse=False):
        """Map a descriptive name to a contrast if present.

        Args:
            contrast (int): Identifier of contrast
            inverse (bool): Whether to return contrast name (False)
                or contrast identifier (True)
        """
        if inverse:
            if self.contrast_class is not None:
                if np.any(np.in1d(self.contrast_class, contrast)):
                    return np.unravel_index((self.contrast_class == contrast).argmax(),
                                             self.contrast_class.shape)[0]
                else:
                    warnings.warn(f'Unknown foil {contrast}, ',
                                   'using default foil_method')
                    return None
        else:
            if self.contrast_class is not None:
                if self.contrast_map is not None:
                    return self.contrast_class[self.contrast_map[contrast]]
                return self.contrast_class[contrast]
        return contrast

    def _weights(self,
                 data,
                 distance_metric,
                 sample=None):
        """Calculate sample weights based on distance metric."""
        if sample is None:
            sample = data[0].reshape(1, -1)
        distances = sklearn.metrics.pairwise_distances(
            data,
            sample,
            metric=distance_metric
        ).ravel()
        return self.kernel_fn(distances)

    def _data(self,
              data,
              scaled_data,
              predict_data,
              distance_metric,
              predict_fn):
        # Calculate weights
        if scaled_data is None:
            scaled_data = data
        weights = self._weights(scaled_data, distance_metric)

        # Predict; distinguish between .predict and .predict_proba
        preds = predict_fn(predict_data)
        if preds.ndim > 1:
            preds = np.argmax(preds, axis=1)

        return data, weights, preds

    def unweighted_sample_training(self,
                                   predict_fn,
                                   n_samples,
                                   seed=1,
                                   **kwargs):
        """Randomly sample from training data, without weights."""
        if self.train_data is None:
            raise Exception('Can only sample from training data '
                            'when it is made available')

        # Predict
        ys_p = predict_fn(self.train_data)
        ys = ys_p.argmax(axis=1) if ys_p.ndim > 1 else ys_p

        # TODO: make prob based on class distrib
        self.seed = check_random_state(seed)
        to_select = self.seed.choice(range(len(ys)),
                                     size=n_samples)

        return self.train_data[to_select], ys[to_select], ys_p[to_select]

    @cache
    def sample_training_data(self,
                             sample,
                             predict_fn,
                             distance_metric='euclidean',
                             n_samples=500,
                             seed=1,
                             **kwargs):
        """Sample neighborhood from training data."""
        xs, ys, _ = self.unweighted_sample_training(predict_fn,
                                                    n_samples=n_samples,
                                                    seed=seed,
                                                    **kwargs)
        return xs, self._weights(xs, distance_metric), ys, sample

    def generate_neighborhood_data(self,
                                   sample,
                                   predict_fn,
                                   distance_metric='euclidean',
                                   n_samples=500,
                                   **kwargs):
        """Generate neighborhood data. Should be implemented in subclass."""
        raise NotImplementedError('Implemented in subclasses')

    def map_feature_names(self, descriptive_path, remove_last=False):
        """Map generic rule to rule with feature names.
        Should be implemented in subclass.
        """
        raise NotImplementedError('Implemented in subclasses')

    def explain(self, fact, foil, counterfactuals, confidence, **kwargs):
        """Form an explanation. Should be implemented in subclass."""
        raise NotImplementedError('Implemented in subclasses')


class DomainMapperTabular(DomainMapper):
    """Domain mapper for tabular data (columns and rows,
    with feature names for columns).
    """

    def __init__(self,
                 train_data,
                 feature_names,
                 contrast_names=None,
                 categorical_features=None,
                 kernel_width=None,
                 assume_independence=False,
                 seed=1):
        """Init.

        Args:
            feature_names (list): Feature names (should be same length
                as # columns)
            contrast_names (list-like): Names of contrasts
            categorical_features (list): Indices of categorical features
            kernel_width (float): Width of kernel for neighborhood weights
            assume_independence (bool): Whether to assume feature
                independence when generating data
            seed (int): Seed
        """
        if kernel_width is None:
            kernel_width = np.sqrt(train_data.shape[1]) * .75

        super().__init__(train_data=train_data,
                         contrast_names=contrast_names,
                         kernel_width=kernel_width,
                         seed=seed)

        if type(train_data) is pd.core.frame.DataFrame:
            raise Exception('Use the subclass DomainMapperPandas',
                            'to work with Pandas DataFrames')
        self.train_data = np.array(train_data)
        if feature_names is None:
            feature_names = [i for i in range(train_data.shape[1])]
        self.features = feature_names
        self.categorical_features = categorical_features

        self.assume_independence = assume_independence

        self.unique_vals = None
        self.encoders = None
        self.feature_map = categorical_features
        self.feature_map_inv = dict()
        self.feature_map_inv_verbose = dict()

        self.original_train_data = self.train_data
        self.train_data = self._one_hot_encode(self.train_data)
        self.feature_counts = self._get_counts()

        self.norm_covariances = None
        self.norm_means = None
        self.scaler = self._init_scaler()

    def _one_hot_encode(self, data):
        """One hot encoding of data, so that it can be
        used by the decision tree.

        Args:
            data: Data to encode

        Returns:
            One-hot Encoded data
        """
        if self.categorical_features is None:
            return data

        # Get unique value for all levels in a categorical feature
        self.unique_vals = dict()
        for column in self.categorical_features:
            self.unique_vals[column] = set(data[:, column])

        # Create encoders
        self.encoders = {feature: Encoder()
                         for feature in self.categorical_features}
        for feature in self.categorical_features:
            self.encoders[feature].fit(self.unique_vals[feature])

        # Create new mapping of indices
        features = np.arange(self.original_train_data.shape[1])
        self.feature_map = dict()
        self.feature_map_inv = dict()
        current_idx = 0

        for feature in features:
            if feature in self.categorical_features:
                new_idx = current_idx + len(self.encoders[feature])
                r = range(current_idx, new_idx)
                self.feature_map[feature] = r
                self.feature_map_inv[r[0]] = feature
                current_idx = new_idx
            else:
                self.feature_map[feature] = current_idx
                self.feature_map_inv[current_idx] = feature
                current_idx += 1

        cfi = itertools.chain.from_iterable
        self.feature_map_inv_verbose = dict(cfi([(v, k)] if type(v) is int else [(v_, k)
                                            for v_ in v]
                                            for (k, v) in self.feature_map.items()))
        return self.apply_encode(data)

    def _get_counts(self):
        """Get counts of categorical features."""
        count_dict = dict()
        if self.categorical_features is not None:
            for f in self.categorical_features:
                count_dict[f] = [self.train_data[:, i].sum() for i in self.feature_map[f]]
        return count_dict

    def _init_scaler(self):
        """Fit a standard scaler to get means and standard devs per feature."""
        scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        scaler.fit(self.train_data)

        # Set mean and standard deviation for one-hot encoded categoricals
        if self.categorical_features is not None:
            for feature in self.categorical_features:
                for sub_feature in self.feature_map[feature]:
                    scaler.mean_[sub_feature] = 0
                    scaler.scale_[sub_feature] = 1

        # Calculate covariances
        normalized_data = sklearn.preprocessing.normalize(self.train_data)
        self.norm_covariances = np.cov(normalized_data, rowvar=False)
        self.norm_means = normalized_data.mean(0)

        return scaler

    def apply_encode(self, data):
        """Encode an instance or data set."""
        if self.categorical_features is None:
            return data
        x = []
        if data.ndim == 1:  # Single instance
            data = data.reshape(1, -1)
            for i, value in enumerate(data.T):
                if i in self.categorical_features:
                    x.extend(self.encoders[i].transform(value).toarray())
                else:
                    x.append(value.astype(int))
            return hstack(x).toarray()[0]
        else:
            for i, column in enumerate(data.T):
                if i in self.categorical_features:
                    x.append(self.encoders[i].transform(column))
                else:
                    x.append(coo_matrix(column.reshape(-1, 1).astype(int)))
            return hstack(x).toarray()

    def _apply_decode(self, data):
        """Decode an encoded instance or data set."""
        if self.categorical_features is None:
            return data
        x = []
        if data.ndim == 1:  # Single instance
            for k, _ in enumerate(self.feature_map_inv):
                to_map = self.feature_map[k]
                if k in self.categorical_features:
                    x.append(self.encoders[k].transform(data[to_map], inverse=True))
                else:
                    x.append(data[to_map])
            return np.array(x)
        else:
            for k, _ in enumerate(self.feature_map_inv):
                to_map = self.feature_map[k]
                if k in self.categorical_features:
                    x.append(self.encoders[k].transform(data[:, to_map],
                                                        inverse=True))
                else:
                    x.append(data[:, to_map])
            return np.column_stack(x)

    @cache
    def generate_neighborhood_data(self,
                                   sample,
                                   predict_fn,
                                   distance_metric='euclidean',
                                   n_samples=500,
                                   seed=1,
                                   **kwargs):
        """Generate neighborhood data for a given point.

        Args:
            sample: Observed sample
            predict_fn: Black box predictor to predict all points
            distance_metric: Distance metric used for weights
            n_samples: Number of samples to generate

        Returns:
            neighor_data (xs around sample),
            weights (weights of instances in xs),
            neighor_data_labels (ys around sample, corresponding to xs)
        """
        neighor_data = self.__generate(sample, n_samples)
        scaled_data = (neighor_data - self.scaler.mean_) / self.scaler.scale_
        predict_data = self._apply_decode(neighor_data)
        return (*self._data(neighor_data, scaled_data,
                            predict_data, distance_metric,
                            predict_fn),
                sample)

    def __generate(self, sample, n_samples,
                   sample_around_instance=False):
        columns = sample.shape[0]

        # Continuous features
        data = self.seed.normal(0, 1, n_samples * columns).reshape(n_samples, columns)
        if not self.assume_independence:  # ensure the same covariances as original data
            try:
                data = np.dot(np.linalg.cholesky(self.norm_covariances), data.T).T
            except np.linalg.LinAlgError:
                data = self.seed.multivariate_normal(self.norm_means,
                                                     self.norm_covariances,
                                                     size=n_samples)
        if sample_around_instance:
            data = data * self.scaler.scale_ + sample
        else:
            data = data * self.scaler.scale_ + self.scaler.mean_

        # Categorical features
        if self.categorical_features is not None:
            for column in self.categorical_features:
                values = [i for i in self.feature_map[column]]
                freqs = self.feature_counts[column] / sum(self.feature_counts[column])
                data[:, values] = 0
                picked_indices = self.seed.choice(values, size=n_samples,
                                                  replace=True, p=freqs)
                data[np.arange(len(data)), picked_indices] = 1

        # Set first data point to original instance
        data[0] = sample
        return data

    def map_feature_names(self, explanation, remove_last=False):
        """Replace feature ids with feature names in a descriptive path.

        Args:
            explanation: Explanation obtained with
                get_explanation() or descriptive_path() function
            remove_last: Remove last tuple from explanation

        Returns:
            Explanation with feature names mapped
        """
        def get_feature(x):
            ret = x
            if x[0] >= 0 and x[0] < len(self.features):
                ret = list(x)
                ret[0] = self.features[ret[0]]
                if type(x) is Literal:
                    ret = Literal(*ret)
                else:
                    ret = type(x)(ret)

            # For categorical features
            if (type(ret) is Literal and self.categorical_features is not None):
                feature = self.feature_map_inv_verbose[x[0]]
                if feature in self.categorical_features:
                    offset = x[0] - self.feature_map[feature][0]
                    ret.feature = self.features[feature]
                    ret.value = self.encoders[feature].idx2name[offset]
                    ret.operator = Operator.EQ if ret.operator is Operator.GEQ \
                                    else Operator.NOTEQ
                    ret.categorical = True
            return ret

        if self.features is not None:
            ex = [get_feature(e) for e in explanation]

            if remove_last:
                ex = ex[:-1]
        return ex

    def rule_to_str(self,
                    rule,
                    remove_last=False):
        """Convert a rule to string."""
        rule = rule or []
        rules = [str(c) for c in self.map_feature_names(rule, remove_last)]
        return ' and '.join(rules)

    def explain(self,
                fact,
                foil,
                counterfactuals,
                factuals,
                confidence,
                fidelity,
                time,
                **kwargs):
        """Explain an instance using the results of
        ContrastiveExplanation.explain_instance()

        Args:
            fact: ID of fact
            foil: ID of foil
            counterfactuals: List of Literals that form
                explanation as disjoint set of foil and
                not-fact rules
            factuals: List of Literals that form explanation
                as set of fact rules
            confidence [0, 1]: Confidence of explanation
                on neighborhood data
            fidelity: ...
            time: Time taken to explain (s)
        """
        fact = self.map_contrast_names(fact)
        foil = self.map_contrast_names(foil)

        e = f"The model predicted '{fact}' instead of '{foil}' " \
            f"because '{self.rule_to_str(counterfactuals)}'"

        if factuals is None:
            return e
        else:
            return (e, f"The model predicted '{fact}' because "
                       f"'{self.rule_to_str(factuals, remove_last=True)}'")


class DomainMapperPandas(DomainMapperTabular):
    """Domain mapper for Pandas dataframes."""

    def __init__(self,
                 train_data,
                 contrast_names=None,
                 kernel_width=None,
                 seed=1):
        """Init.

        Args:
            train_data (pd.core.frame.DataFrame): Training data
            contrast_names (list-like): Names of contrasts
            kernel_width (float): Width of kernel for neighborhood weights
            seed (int): Seed
        """
        if type(train_data) is not pd.core.frame.DataFrame:
            raise Exception('Use DomainMapperTabular to work with other',
                            'types of tabular data')

        feature_names = train_data.columns.values
        obj_cols = train_data.select_dtypes('object').columns
        categorical_features = train_data.columns.get_indexer(obj_cols)
        if categorical_features == []:
            categorical_features = None
        train_data = train_data.to_numpy(copy=True)
        super().__init__(train_data,
                         feature_names,
                         contrast_names=contrast_names,
                         categorical_features=categorical_features,
                         kernel_width=kernel_width,
                         seed=seed)


class DomainMapperImage(DomainMapper):
    """Domain mapper for image data (CNN-only) using feature_fn."""

    def __init__(self,
                 train_data,
                 feature_fn,
                 contrast_names=None,
                 kernel_width=.25,
                 seed=1):
        """Init.

        Args:
            train_data: Training data
            feature_fn: `???`
            contrast_names (list-like): Names of contrasts
            kernel_width (float): Width of kernel for neighborhood weights
            seed (int): Seed
        """
        super().__init__(train_data, contrast_names=contrast_names,
                         kernel_width=kernel_width, seed=seed)
        self.feature_fn = feature_fn

    @cache
    def sample_training_data(self,
                             sample,
                             predict_fn,
                             distance_metric='euclidean',
                             n_samples=100,
                             seed=1,
                             **kwargs):
        """Sample neighborhood from training data."""
        xs, ys, _ = super().unweighted_sample_training(predict_fn,
                                                       n_samples=n_samples,
                                                       seed=seed,
                                                       **kwargs)
        xs = self.feature_fn(xs)
        return (xs, self._weights(xs, distance_metric),
                ys, self.feature_fn(sample))

    @cache
    def generate_neighborhood_data(self,
                                   sample,
                                   predict_fn,
                                   distance_metric='euclidean',
                                   n_samples=100,
                                   seed=1,
                                   **kwargs):
        """Generate neighborhood data."""
        raise NotImplementedError()

    def map_feature_names(self, descriptive_path, remove_last=False):
        """Map generic rule to rule with feature names."""
        raise NotImplementedError()

    def explain(self, fact, foil, counterfactuals, factuals, confidence, fidelity, time):
        """Get an explanation for an image."""
        fact = self.map_contrast_names(fact)
        foil = self.map_contrast_names(foil)

        return (f"The model predicted '{fact}' instead of '{foil}'",
                counterfactuals)


class DomainMapperImageSegments(DomainMapper):
    """Domain mapper for image data (model-agnostic) using segments."""

    def __init__(self,
                 *args,
                 **kwargs):
        """Init.

        For arguments see `DomainMapperImage`.
        """
        super().__init__(*args, **kwargs)
        self.image = None
        self.segments = None
        self.alt_image = None

    def n_features(self):
        """Determine number of features."""
        if self.segments is None:
            return 0
        return np.unique(self.segments).shape[0]

    def apply_encode(self, data):
        """Apply encoding to data."""
        return data

    def _data_labels(self,
                     predict_fn,
                     batch_size=30):
        from copy import deepcopy
        from itertools import product
        data = np.array([list(i) for i in product([0, 1],
                         repeat=self.n_features())])
        labels = []
        imgs = []
        for row in data:
            temp = deepcopy(self.image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(self.segments.shape).astype(bool)
            for z in zeros:
                mask[self.segments == z] = True
            temp[mask] = self.alt_image[mask]
            imgs.append(temp)
            if len(imgs) == batch_size:
                preds = predict_fn(np.array(imgs))
                labels.extend(preds)
                imgs = []
        if len(imgs) > 0:
            preds = predict_fn(np.array(imgs))
            labels.extend(preds)
        return data, np.argmax(np.array(labels), axis=1)

    @check_stringvar(('segmentation_fn', ['quickshift', 'felzenszwalb', 'slic']))
    def __high_level_features(self,
                              sample,
                              predict_fn,
                              distance_metric='euclidean',
                              segmentation_fn='quickshift',
                              alt_image=None,
                              hide_color=None,
                              n_samples=30,
                              batch_size=30,
                              seed=1):
        from skimage.segmentation import quickshift, felzenszwalb, slic

        self.image = sample

        # Segment
        if segmentation_fn == 'felzenschwalb':
            segments = felzenszwalb(sample)
        elif segmentation_fn == 'slic':
            segments = slic(sample, n_segments=10)
        else:
            segments = quickshift(sample, kernel_size=2,
                                  max_dist=200, ratio=0.2,
                                  random_seed=seed)
        self.segments = segments

        # Get fudged image, unless given
        if alt_image is None:
            alt_image = sample.copy()
            if hide_color is None:
                for x in np.unique(segments):
                    alt_image[segments == x] = (
                        np.mean(sample[segments == x][:, 0]),
                        np.mean(sample[segments == x][:, 1]),
                        np.mean(sample[segments == x][:, 2]))
            else:
                alt_image[:] = hide_color
        self.alt_image = alt_image

        data, preds = self._data_labels(predict_fn, batch_size=batch_size)

        return (data, self._weights(data, distance_metric),
                preds, np.zeros(self.n_features()))

    @cache
    def sample_training_data(self,
                             sample,
                             predict_fn,
                             distance_metric='euclidean',
                             n_samples=30,
                             seed=1,
                             foil_encode_fn=None,
                             **kwargs):
        """Sample neighborhood instances from the training data."""
        fn = super().unweighted_sample_training
        if foil_encode_fn is None:
            self.alt_image = fn(predict_fn, n_samples=1, seed=seed,
                                **kwargs)[0][0]
        else:  # Foil-sensitive
            data, preds, preds_probs = fn(predict_fn, n_samples=n_samples,
                                          seed=seed, **kwargs)

            # Highest confidence foil
            foils = np.argwhere(foil_encode_fn(preds) == 1).ravel()
            foils_p = [preds_probs[f][preds[f]] for f in foils]
            self.alt_image = data[foils[np.argmax(foils_p)]]

        return self.__high_level_features(sample,
                                          predict_fn,
                                          distance_metric=distance_metric,
                                          alt_image=self.alt_image,
                                          n_samples=n_samples,
                                          seed=seed)

    @cache
    def generate_neighborhood_data(self,
                                   sample,
                                   predict_fn,
                                   distance_metric='euclidean',
                                   n_samples=30,
                                   batch_size=10,
                                   seed=1,
                                   hide_color=None,
                                   segmentation_fn='quickshift',
                                   **kwargs):
        """Generate perturbed instances in the neighborhood of the
        instance to explain.
        """
        while sample.ndim > 3:
            sample = sample[0]

        return self.__high_level_features(sample,
                                          predict_fn,
                                          distance_metric=distance_metric,
                                          segmentation_fn=segmentation_fn,
                                          hide_color=hide_color,
                                          n_samples=n_samples,
                                          batch_size=batch_size,
                                          seed=seed)

    def map_feature_names(self, explanation):
        """Map abstract features to features in the DomainMapper."""
        if (self.image is None or self.alt_image is None or
                self.segments is None):
            return

        for e in explanation:
            if type(e) is Literal:
                temp = np.zeros(self.image.shape)
                temp[self.segments == e.feature] = 1
                if e.operator is Operator.SEQ:
                    temp = self.image * temp
                elif e.operator is Operator.GT:
                    temp = self.alt_image * temp
                show_image(temp)
        return explanation

    def explain(self, fact, foil, counterfactuals, factuals,
                confidence, fidelity, time, **kwargs):
        """Get an explanation for an image."""
        fact = self.map_contrast_names(fact)
        foil = self.map_contrast_names(foil)

        print(fact)
        show_image(self.image)
        print(foil)
        show_image(self.alt_image)

        self.map_feature_names(counterfactuals)

        return (f"The model predicted '{fact}' instead of '{foil}'",
                counterfactuals)
