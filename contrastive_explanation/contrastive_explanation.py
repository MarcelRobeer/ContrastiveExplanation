import numpy as np
import operator
import warnings
import sklearn
import time

from itertools import groupby
from sklearn.utils import check_random_state

from .rules import Operator, Literal
from .domain_mappers import DomainMapper
from .explanators import Explanator, TreeExplanator
from .fact_foil import FactFoilClassification, FactFoilRegression

"""Contrastive Explanation [WIP]: pedagogical (model-agnostic) method
for persuasive explanations based on the user's outcome of interest.

Marcel Robeer (c) 2018
TNO, Utrecht University

TODO:
    * Add more domain_mappers (image, text)
    * Add more methods to obtain a foil
    * Define new strategies for DecisionTreeExplanator
    * Extend support for regression
    * Adjust DomainMapper.generate_neighborhood_data() to
        generate foil samples
"""


class ContrastiveExplanation:
    '''General class for creating a Contrastive Explanation.'''

    def __init__(self,
                 domain_mapper,
                 explanator=None,
                 regression=False,
                 verbose=False,
                 seed=1):
        '''Init.

        Args:
            explanator: explanator.Explanator() to create an explanation
            domain_mapper: domain_mapper.DomainMapper() for generating
                neighborhood data and giving descriptive names to features
                and contrasts
            regression (bool): regression (True) or other type of ML
            verbose (bool): Print intermediary steps of algorithm
            seed (int): Seed for random functions
        '''
        self.seed = check_random_state(seed)

        if not explanator:
            explanator = TreeExplanator(seed=self.seed)

        if not isinstance(domain_mapper, DomainMapper):
            raise Exception('domain_mapper should be a DomainMapper')
        if not isinstance(explanator, Explanator):
            raise Exception('explanator should be an Explanator')

        self.explanator = explanator
        self.domain_mapper = domain_mapper
        self.regression = regression
        self.verbose = verbose
        self.fact_foil = None

    def _combine_features(self, decision_path):
        '''Combine tuples with the same feature in a decision path.

        Args:
            decision_path: Decision path

        Returns:
            Decision path with duplicate features merged.
        '''
        if self.verbose:
            print(f'[C] Combining full rules {decision_path}...')

        def combine(rules):
            seq = []
            geq = []
            eq = []
            for rule in rules:
                if rule.operator == Operator.SEQ:
                    seq.append(rule[2])
                elif rule.operator == Operator.GT:
                    geq.append(rule[2])
                elif rule.operator == Operator.EQ:
                    eq.append(rule[2])

            feature = rules[0][0]
            if not seq and not geq and len(eq) <= 1:
                return rules
            elif len(eq) > 1:
                return [Literal(feature, Operator.EQ, eq)]
            elif not seq:
                return [Literal(feature, Operator.GT, max(geq))]
            elif not geq:
                return [Literal(feature, Operator.SEQ, min(seq))]
            else:
                return [Literal(feature, Operator.SEQ, min(seq)),
                        Literal(feature, Operator.GT, max(geq))]

        combined = [combine(list(subiter))
                    for _, subiter in groupby(decision_path, lambda t: t[0])]

        return [c for sc in combined for c in sc]

    def form_explanation(self, decision, contrastive=True):
        '''Form an explanation of Literals, combine Literals
        when they describe the same feature.
        '''
        if decision is None:
            return None

        if self.verbose:
            print(f'[C] Decision obtained: {decision}')

        # Get explanation
        exp = self.explanator.get_explanation(decision,
                                              contrastive=contrastive)
        exp = list(filter(None, exp))

        # Combine explanation
        return self._combine_features(exp)

    def explain_instance(self,
                         model_predict,
                         fact_sample,
                         foil=None,
                         foil_pick_method=None,
                         foil_strategy='informativeness',
                         generate_data=True,
                         n_samples=500,
                         include_factual=False,
                         epsilon=0.1,
                         **kwargs):
        '''Contrastively explain an instance (counterfactual).

        Args:
            model_predict: Black-box model predictor (proba for class)
            fact_sample: Input sample of fact
            foil: Manually enter a foil (if None, uses foil_pick_method)
            foil_pick_method: Method to decide on foil, choose
                class: ('second' = second most probable decision,
                 'random' = randomly pick from not-foil)
                reg: ('greater' = greater than fact,
                 'smaller' = smaller than fact)
            foil_strategy: How to determine the contrastive
                decision region for the foil, choose from:
                ('closest' = closest to fact,
                 'size' = based on number of instances in node,
                 'impurity' = minimize the impurity difference,
                 'informativeness' = weighted function of size and impurity,
                 'random' = random pick)
            generate_data (bool): Generate neighborhood data (True) or pick
                from training data (False)
            n_samples (int): Number of samples to pick from data
            include_factual (bool): Also return a factual explanation tree,
                trained on generated/sampled data.
            epsilon: Small offset for regression, increase when no explanation
                is found.

        Returns:
            Tuple (fact, foil, counterfactual), feed into the explain()
            function in the domain_mapper
        '''
        st = time.time()

        # Get fact and foil
        if self.regression:
            self.fact_foil = FactFoilRegression(verbose=self.verbose,
                                                epsilon=epsilon)
        else:
            self.fact_foil = FactFoilClassification(verbose=self.verbose)
    
        if foil is not None:
            foil = self.domain_mapper.map_contrast_names(foil, inverse=True)
            fact, foil = self.fact_foil.get_fact(model_predict, fact_sample, foil)
        if foil is None:
            fact, foil = self.fact_foil.get_fact_foil(model_predict, fact_sample,
                                                      foil_method=foil_pick_method)

        # Generate neighborhood data
        if self.verbose:
            print('[D] Obtaining neighborhood data')

        if generate_data:
            data_fn = self.domain_mapper.generate_neighborhood_data
        else:
            data_fn = self.domain_mapper.sample_training_data
        xs, weights, ys, fact_sample = data_fn(fact_sample,
                                               model_predict,
                                               n_samples=n_samples,
                                               foil_encode_fn=self.fact_foil.encode,
                                               **kwargs)

        # Encode foil such that foil = 1 / else = 0
        ys_foil = self.fact_foil.encode(ys)

        if 1 not in ys_foil:
            warnings.warn('Neighborhood data does not contain any foils')
            return fact, foil, None, None, 0, 0, time.time() - st

        # Train model and get rules
        rule, confidence, local_fidelity = self.explanator.get_rule(fact_sample,
                                                                    fact, foil,
                                                                    xs, ys_foil,
                                                                    weights,
                                                                    foil_strategy=foil_strategy)

        # Explain difference between fact and closest decision
        counterfactual = self.form_explanation(rule)

        # Also explain using factual if required
        factual = None
        if include_factual:
            if type(self.explanator) is TreeExplanator:
                e = self.explanator
            else:
                e = TreeExplanator()

            if not self.regression:
                t = sklearn.tree.DecisionTreeClassifier(random_state=self.seed,
                                                        class_weight='balanced')
            else:
                t = sklearn.tree.DecisionTreeRegressor(random_state=self.seed)

            t.fit(xs, ys, sample_weight=weights)

            if t.tree_.node_count > 1:
                fact_rule = e.decision_path(t, fact_sample)
                factual = self.form_explanation(fact_rule, contrastive=False)[:-1]
            else:
                factual = None

        # Warnings
        if not counterfactual:
            # First, try to overfit more to get explanation
            if (type(self.explanator) is TreeExplanator and
                    self.explanator.generalize < 2):
                self.explanator.generalize = 2
                return self.explain_instance(model_predict,
                                             fact_sample,
                                             foil_pick_method=foil_pick_method,
                                             foil_strategy=foil_strategy,
                                             generate_data=generate_data,
                                             n_samples=n_samples,
                                             include_factual=include_factual,
                                             epsilon=epsilon,
                                             **kwargs)

            n = self.domain_mapper.map_contrast_names
            warnings.warn(f'Could not find a difference between fact '
                          f'"{n(fact)}" and foil "{n(foil)}"')           
            if self.regression:
                warnings.warn('Consider increasing epsilon')

        return (fact, foil,
                counterfactual, factual,
                confidence, local_fidelity,
                time.time() - st)

    def explain_instance_domain(self,
                                *args,
                                **kwargs):
        '''Explain instance and map to domain. For arguments see
        ContrastiveExplanation.explain_instance().'''
        return self.domain_mapper.explain(*self.explain_instance(*args,
                                                                 **kwargs))
