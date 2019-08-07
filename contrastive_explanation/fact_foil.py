"""FactFoils determine the fact (actual outcome) and
foil (outcome of interest), and encode the target variable
(outcomes) into a binary classification problem where
`fact = 0` and `foil = 1`.
"""

import numpy as np
import warnings


class FactFoil:
    """Base class for fact/foil."""

    default_method = None

    def __init__(self,
                 verbose=False):
        """Init.

        Args:
            verbose (bool): Print intermediary steps of algorithm
        """
        self.verbose = verbose
        self.fact = None
        self.foil = None

    @staticmethod
    def _pred(model_predict, sample, pred_has_max=True):
        try:
            pred = model_predict(sample.reshape(1, -1))[0]
        except ValueError:
            pred = model_predict(np.array([sample]))[0]
        if pred_has_max:
            _pred = pred
            if len(_pred) == 1:
                _pred = np.array([pred[0], 1 - pred[0]])
            return pred, np.argmax(_pred)
        else:
            return pred

    def __check_validity(self):
        if self.fact == self.foil:
            raise Exception('Fact and foil cannot be equal')

    def fact_foil_encode(self, model, sample, ys,
                         foil_method=default_method):
        """Determine fact and foil, and encode target variable into
        fact and foil.
        """
        fact, foil = self.get_fact_foil(model, sample, foil_method=foil_method)
        ys = self.encode(ys)
        return fact, foil, ys

    def get_fact(self, model_predict, sample, foil):
        """Determine the fact for a given sample, and use a
        manually provided foil.

        Args:
            model_predict: Predictor to predict sample x
            sample: Instance
            foil: Chosen foil

        Returns:
            Tuple (fact, foil)
        """
        _, self.fact = self._pred(model_predict, sample)
        self.foil = foil
        self.__check_validity()
        return self.fact, self.foil

    def get_fact_foil(self, model, sample, foil_method=default_method):
        """Determine the fact and foil for a given sample.

        Args:
            model: Black box predictor m() to predict sample x
            sample: Input to determine fact and foil
            foil_method: Method to pick foil

        Returns:
            Tuple (fact, foil)
        """
        if foil_method is None or '':
            foil_method = self.default_method

        self.fact, self.foil = self._get_fact_foil_impl(model, sample,
                                                        foil_method)

        if self.verbose:
            print(f'[F] Picked foil "{self.foil}" using foil selection '
                  f'strategy "{foil_method}"')

        self.__check_validity()
        return self.fact, self.foil

    def _get_fact_foil_impl(self, model_predict, sample, foil_method):
        raise NotImplementedError('Implemented in subclasses')

    def encode(self, ys):
        """Encode outcomes (ys) into fact (0) and foil (1)."""
        raise NotImplementedError('Implemented in subclasses')


class FactFoilClassification(FactFoil):
    """Fact/foil implementation for classification and unsupervised tasks."""

    default_method = 'second'

    def get_foil(self, class_probs, method=default_method):
        """Get foil for a probability distribution of outputs.

        Args:
            class_probs: Class probabilities (output of .predict_proba)
            method: Method to pick foil from class probabilities in
                ('second', 'random')

        Returns:
            Class index of foil
        """
        if method == 'second':
            return np.argsort(-class_probs)[1]
        elif method == 'random':
            return np.random.choice(class_probs[1:])
        return np.argmin(class_probs)

    def _get_fact_foil_impl(self, model_predict, sample,
                            foil_method=default_method):
        pred, fact = self._pred(model_predict, sample)
        foil = self.get_foil(pred, foil_method)

        return fact, foil

    def encode(self, ys):
        """Encode target variable into fact (0) and foil (1)."""
        return (ys == self.foil) * 1


class FactFoilRegression(FactFoil):
    """Fact/foil implementation for regression."""

    default_method = 'greater'

    def __init__(self,
                 verbose=False,
                 epsilon=0.1):
        """Init.

        Args:
            epsilon: Small offset for smaller/greater to
                better distinguish when converting from regression
                into a binary (fact-vs-foil) classification problem.
        """
        if epsilon < 0:
            warnings.warn(f'Epsilon should be >0 but is {epsilon}')
        super().__init__(verbose=verbose)
        self.epsilon = epsilon

    def get_foil(self, fact, method=default_method):
        """Get foil for regression outcomes.

        Args:
            fact: Value of fact
            method: Method to pick foil given the outcomes in
                ('greater', 'smaller')

        Returns:
            Foil corresponding to fact

        """
        if method == 'greater':
            return f'more than {fact}'
        elif method == 'smaller':
            return f'less than {fact}'
        raise NotImplementedError('TODO: Regression')

    def _get_fact_foil_impl(self, model_predict, sample,
                            foil_method=default_method):
        fact = self._pred(model_predict, sample, pred_has_max=False)
        foil = self.get_foil(fact, method=foil_method)
        return fact, foil

    def encode(self, ys):
        """Encode target variable into fact (0) and foil (1)."""
        if str(self.foil).startswith('less'):
            return (ys < (self.fact - self.epsilon)) * 1
        elif str(self.foil).startswith('more'):
            return (ys > (self.fact + self.epsilon)) * 1
        raise NotImplementedError('TODO: Regression')
