"""Uses generic tabular data to explain a single instance with
a contrastive/counterfactual explanation.

Attributes:
    DEBUG (bool): Debug mode enabled
"""

import numpy as np
import networkx as nx
import warnings

from sklearn import tree, ensemble, metrics
from sklearn.tree import _tree
from sklearn.utils import check_random_state

from .rules import Operator, Literal
from .utils import cache, check_stringvar, check_relvar, print_binary_tree


DEBUG = False


class Explanator:
    """General class for Explanators (method to acquire explanation)."""

    def __init__(self,
                 verbose=False,
                 seed=1):
        """Init.

        Args:
            verbose (bool): Print intermediary steps of algorithm
            seed (int): Seed for random functions
        """
        self.verbose = verbose
        self.seed = check_random_state(seed)

    def get_rule(self,
                 fact_sample,
                 fact,
                 foil,
                 xs,
                 ys,
                 weights,
                 **kwargs):
        """Get rules for 'fact' and 'foil' using an explanator.

        Args:
            fact_sample: Sample x of fact
            fact: Outcome y = m(x) of fact
            foil: Outcome y for foil
            xs: Training data
            ys: Training data labels, has to contain
                observations with the foil
            weights: Weights of training data, based on
                distance to fact_sample
            foil_strategy: Strategy for finding the
                foil decision region ('closest', 'random')

        Returns:
            foil_path (descriptive_path for foil), confidence
        """
        raise NotImplementedError('Implemented in subclasses')

    def get_explanation(self, rules):
        """Get explanation given a set of rules."""
        raise NotImplementedError('Implemented in subclasses')


class RuleExplanator(Explanator):
    """General class for rule-based Explanators."""

    def get_explanation(self, rules, contrastive=True):
        """Get an explanation given a rule, of why the fact
        is outside of the foil decision boundary (contrastive) or
        why the fact is inside the fact decision boundary.
        """
        for feature, threshold, _, foil_greater, fact_greater in rules:
            if (contrastive and fact_greater and not foil_greater or
                    not contrastive and foil_greater):
                yield Literal(feature, Operator.GT, threshold)
            elif (contrastive and not fact_greater and foil_greater or
                    not contrastive and not foil_greater):
                yield Literal(feature, Operator.SEQ, threshold)
            else:
                yield None


class TreeExplanator(RuleExplanator):
    """Explain using a decision tree."""

    def __init__(self,
                 generalize=2,
                 verbose=False,
                 seed=1):
        """Init.

        Args:
            Generalize [0, 1]: Lower = overfit more, higher = generalize more
        """
        super().__init__(verbose=verbose, seed=seed)
        self.generalize = generalize
        self.tree = None
        self.graph = None

    @cache
    def _foil_tree(self, xs, ys, weights, seed, **dtargs):
        """Classifies foil-vs-rest using a DecisionTreeClassifier.

        Args:
            xs: Input data
            ys: Input labels (1 = foil, 0 = else)
            weights: Input sample weights
            **dtargs: Pass on additional arguments to
                    DecisionTreeClassifier

        Returns:
            Trained model on input data for binary
            classification (output vs rest)
        """
        model = tree.DecisionTreeClassifier(random_state=check_random_state(seed),
                                            class_weight='balanced',
                                            **dtargs)
        model.fit(xs, ys, sample_weight=weights)

        # If we only have a root node there is no explanation, so try acquiring
        # and explanation by training a forest of trees and picking the highest
        # performance estimator
        if model.tree_.max_depth < 2:
            seed_ = check_random_state(seed)
            forest = ensemble.RandomForestClassifier(random_state=seed_,
                                                     class_weight='balanced')
            forest.fit(xs, ys, sample_weight=weights)

            estimators = [(e.score(xs, ys), e) for e in forest.estimators_
                          if e.tree_.max_depth > 1]

            if estimators is not None and estimators:
                model = sorted(estimators, key=lambda x: x[0], reverse=True)[0][1]

        local_fidelity = metrics.accuracy_score(ys, model.predict(xs))

        if self.verbose:
            print('[E] Fidelity of tree on neighborhood data =', local_fidelity)

        if DEBUG:
            print_binary_tree(model, xs[0].reshape(1, -1))

        return model, local_fidelity

    def descriptive_path(self, decision_path, sample, tree):
        """Create a descriptive path for a decision_path of node ids.

        Args:
            decision_path (list, np.array): Node ids to describe
            sample: Sample to describe
            tree: sklearn tree used to create decision_path

        Returns:
            Tuples (feature, threshold, sample value, greater,
                    decision_path > threshold,
                    sample value > threshold)
            for all node ids in the decision_path
        """
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        return [(feature[node],
                 threshold[node],
                 sample[feature[node]],
                 greater,
                 float(sample[feature[node]]) > threshold[node])
                for node, greater in decision_path]

    def decision_path(self, tree, sample):
        """Get a descriptive decision path of a sample.

        Args:
            tree: sklearn tree
            sample: Sample to decide decision path of

        Returns:
            Descriptive decision path for sample
        """
        dp = list(np.nonzero(tree.decision_path(sample.reshape(1, -1)))[1])
        if len(dp) == 0:
            return []
        turned_right = [dp[i] in tree.tree_.children_right
                        for i, node in enumerate(dp[:-1])] + [False]

        return self.descriptive_path(list(zip(dp, turned_right)), sample, tree)

    def __to_graph(self, t, node=0):
        """Recursively obtain graph of a sklearn tree.

        Args:
            t: sklearn tree.tree_
            node: Node ID

        Returns: Graph of tuples (parent_id, child_id, right_path_taken)
        """
        left = t.children_left[node]
        right = t.children_right[node]

        if left != _tree.TREE_LEAF:
            left_path = [(node, left, False)] + self.__to_graph(t, left)
            right_path = [(node, right, True)] + self.__to_graph(t, right)
            return left_path + right_path
        return []

    def __get_nodes(self, graph):
        nodes = []
        for g in graph:
            nodes.extend(g)
        return [n for n in list(set(nodes)) if n not in [True, False]]

    @cache
    def _fact_foil_graph(self, tree, start_node=0):
        """Convert a tree into a graph from the fact_leaf to
        all other leaves.

        Args:
            tree: sklearn tree.tree_
            start_node: Node ID to start constructing graph from

        Returns:
            Graph, list of foil nodes
        """
        # Convert tree to graph
        graph = self.__to_graph(tree, node=start_node)

        # Acquire the foil leafs
        foil_nodes = [node for node in self.__get_nodes(graph)
                      if (tree.feature[node] == _tree.TREE_UNDEFINED and
                          np.argmax(tree.value[node]) == 1)]

        return graph, foil_nodes

    def __construct_tuples(self, graph, tree_data, strategy='informativeness'):
        for v1, v2, greater in graph:
            if strategy == 'closest':
                yield v1, v2, greater, 1.0
            elif strategy == 'size':
                yield v1, v2, greater, 1 - (tree_data.n_node_samples[v2] /
                                            sum(tree_data.n_node_samples))
            elif strategy == 'impurity':
                yield v1, v2, greater, 1 - abs(tree_data.impurity[v1] -
                                               tree_data.impurity[v2])
            elif strategy == 'informativeness':
                yield v1, v2, greater, (1 / abs(tree_data.impurity[v1] -
                                                tree_data.impurity[v2]) +
                                        1 / tree_data.n_node_samples[v2])
            elif strategy == 'random':
                yield v1, v2, greater, np.random.random_sample()
            else:
                yield v1, v2, greater, 0.0

    def __shortest_path(self, g, start, end):
        """Determine shortest path from 'start' to
        'end' in undirected graph 'g'.

        Args:
            g: Graph represented using list of tuples
                (vertex1, vertex2, _, vertex_weight)
            start: Start vertex
            end: End vertex

        Returns:
            Shortest path (list of vertices)
        """
        G = nx.Graph()
        for v1, v2, _, w in g:
            G.add_edge(v1, v2, weight=w)
        return nx.shortest_path(G, start, end, weight='weight')

    @check_stringvar(('strategy', ['closest', 'size', 'impurity',
                                   'informativeness', 'random']))
    def _get_path(self,
                  graph,
                  fact_node,
                  foil_nodes,
                  tree_data,
                  strategy='informativeness'):
        """Get shortest path in graph based on strategy.

        Args:
            graph: Unweighted graph with tuples (v1, v2, _)
                reconstructed from decision tree.
            fact_node: Leaf node 'fact_sample' ended up in
            foil_nodes: List of nodes with decision foil
            tree_data: sklearn.tree.tree_
            strategy: Weight strategy (see 'get_rules()')

        Returns:
            List of foil decisions, represented as descriptive_path
        """
        # Add weights to vertices
        weighted_graph = list(self.__construct_tuples(graph, tree_data,
                                                      strategy))

        # Add final point '-1' to find shortest path to, add 0 weight edge
        foil_sink = -1
        final_graph = np.array(weighted_graph + [(f, foil_sink, False, 0.0)
                                                 for f in foil_nodes],
                               dtype=np.dtype([('v1', 'int'),
                                               ('v2', 'int'),
                                               ('greater', 'bool'),
                                               ('w', 'float')]))

        # Get shortest path
        shortest_path = self.__shortest_path(final_graph,
                                             fact_node,
                                             foil_sink)[:-1]

        # Get confidence (accuracy of foil leaf)
        foil_leaf_classes = tree_data.value[shortest_path[-1]]
        confidence = foil_leaf_classes[0, 1] / np.sum(foil_leaf_classes)

        if self.verbose:
            print(f'[E] Found shortest path {shortest_path} using '
                  f'strategy "{strategy}"')

        # Decisions taken for path
        foil_decisions = []
        for a, b in zip(shortest_path[:-1], shortest_path[1:]):
            for edge in final_graph:
                if a == edge[0] and b == edge[1]:
                    foil_decisions.append((edge[0], edge[2]))

        return foil_decisions, confidence

    @check_relvar(('beta', '>= 1'))
    def closest_decision(self, tree, sample,
                         strategy='informativeness',
                         beta=5):
        """Find the closest decision that is of a class other than the
        target class.

        Args:
            tree: sklearn tree
            sample: Entry to explain
            beta: Hyperparameter >= 1 to determine when to only
                search part of tree (higher = search smaller area)

        Returns:
            Ordered descriptive decision path difference,
            confidence of leaf decision
        """
        # Only search part of tree depending on tree size
        decision_path = tree.decision_path(sample.reshape(1, -1)).indices
        if len(decision_path) < 2:
            warnings.warn('Stub tree')
            return None, 0.0
        start_depth = int(round(len(decision_path) / beta))
        start_node = decision_path[start_depth]

        # Get decision for sample
        fact_leaf = tree.apply(sample.reshape(1, -1)).item(0)

        # TODO: Retrain tree if wrong prediction
        if np.argmax(tree.tree_.value[fact_leaf]) != 0:
            warnings.warn('Tree did not predict as fact')

        # Find closest leaf that does not predict output x, based on a strategy
        graph, foil_nodes = self._fact_foil_graph(tree.tree_,
                                                  start_node=start_node)

        if self.verbose:
            print(f'[E] Found {len(foil_nodes)} contrastive decision regions, '
                  f'starting from node {start_node}')

        if len(foil_nodes) == 0:
            return None, 0

        # Contrastive decision region
        foil_path, confidence = self._get_path(graph,
                                               fact_leaf,
                                               foil_nodes,
                                               tree.tree_,
                                               strategy)

        return self.descriptive_path(foil_path, sample, tree), confidence

    def get_rule(self,
                 fact_sample,
                 fact,
                 foil,
                 xs,
                 ys,
                 weights,
                 foil_strategy='informativeness'):
        """Get rules for 'fact' and 'foil' using a
        decision tree explanator. For arguments see
        Explanator.get_rule().
        """
        if self.verbose:
            print("[E] Explaining with a decision tree...")

        # Train a one-vs-rest tree on the foil data
        self.tree, fidelity = self._foil_tree(xs, ys, weights,
                                              self.seed,
                                              min_samples_split=self.generalize)

        # Get decision path
        path, confidence = self.closest_decision(self.tree,
                                                 fact_sample,
                                                 strategy=foil_strategy)

        return path, confidence, fidelity


class PointExplanator(Explanator):
    """Explain by selecting and comparing to a prototype point."""

    @check_stringvar(('strategy', ['closest', 'medoid', 'random']))
    def contrastive_prototype(self,
                              xs,
                              ys,
                              weights,
                              strategy='closest'):
        """Get a contrastive sample based on strategy."""
        # Get foil xs
        ys_slice = [idx for idx, y in enumerate(ys) if y == 1]
        xs_foil = xs[ys_slice]

        if xs_foil is None:
            return None

        if strategy == 'closest':
            return xs_foil[np.argmax(weights[1:]) + 1]
        elif strategy == 'medoid':
            print(xs_foil)
            return xs_foil[0][0]
        elif strategy == 'random':
            return xs_foil[np.random.randint(xs_foil.shape[0], size=1), :][0]

    def path_difference(self,
                        fact_sample,
                        foil_sample,
                        normalize=False):
        """Calculate difference between two equal length samples.

        Args:
            fact_sample: Sample for fact
            foil_sample: Sample for foil
            normalize (bool): TODO

        Returns:
            Difference between fact_sample and foil_sample ordered
            by magnitude of difference
        """
        if len(fact_sample) != len(foil_sample):
            raise Exception('Number of features of fact sample and '
                            'prototype point should be equal')

        difference = fact_sample - foil_sample
        difference_path = [(i, abs(d), fact_sample[i], d < 0)
                           for i, d in enumerate(difference)]

        # Sort by magnitude of difference
        return sorted(difference_path, key=lambda d: d[1], reverse=True)

    def get_rule(self,
                 fact_sample,
                 fact,
                 foil,
                 xs,
                 ys,
                 weights,
                 foil_strategy='closest',
                 **kwargs):
        """Get rules for 'fact' and 'foil' using a
        point explanator. For arguments see Explanator.get_rule().
        """
        if self.verbose:
            print("[E] Explaining with a prototype point...")

        # Acquire prototype for foil
        foil_sample = self.contrastive_prototype(xs, ys, weights,
                                                 strategy=foil_strategy)
        if foil_sample is None:
            return None, 0

        if self.verbose:
            print(f'[E] Found prototype point {foil_sample} using '
                  f'strategy "{foil_strategy}"')

        # Explain difference as path
        return self.path_difference(fact_sample, foil_sample), 0, 0

    def get_explanation(self, rules, contrastive=True):
        """Get an explanation given a rule, of why the fact
        is not a foil (contrastive) or why it is a fact.
        """
        for feature, difference, _, is_negative in rules:
            if (contrastive and is_negative or
                    not contrastive and not is_negative):
                yield Literal(feature, Operator.MINUSEQ, difference)
            else:
                yield Literal(feature, Operator.PLUSEQ, difference)
