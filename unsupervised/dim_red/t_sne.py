
# coding:utf-8
import logging

import numpy as np


np.random.seed(999)

"""
References:
https://lvdmaaten.github.io/tsne/
Based on:
https://lvdmaaten.github.io/tsne/code/tsne_python.zip
"""
class BaseEstimator:
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None):
        """Ensure inputs to an estimator are in the expected format.

        Ensures X and y are stored as numpy ndarrays by converting from an
        array-like object if necessary. Enables estimators to define whether
        they require a set of y target values or not with y_required, e.g.
        kmeans clustering requires no target labels and is fit against only X.

        Parameters
        ----------
        X : array-like
            Feature dataset.
        y : array-like
            Target values. By default is required, but if y_required = false
            then may be omitted.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.size == 0:
            raise ValueError("Got an empty matrix.")

        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])

        self.X = X

        if self.y_required:
            if y is None:
                raise ValueError("Missed required argument y")

            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.size == 0:
                raise ValueError("The targets array must be no-empty.")

        self.y = y

    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("You must call `fit` before `predict`")

    def _predict(self, X=None):
        raise NotImplementedError()

class TSNE(BaseEstimator):
    y_required = False

    def __init__(self, n_components=2, perplexity=30.0, max_iter=200, learning_rate=500):
        """A t-Distributed Stochastic Neighbor Embedding implementation.

        Parameters
        ----------
        max_iter : int, default 200
        perplexity : float, default 30.0
        n_components : int, default 2
        """
        self.max_iter = max_iter
        self.perplexity = perplexity
        self.n_components = n_components
        self.initial_momentum = 0.5
        self.final_momentum = 0.8
        self.min_gain = 0.01
        self.lr = learning_rate
        self.tol = 1e-5
        self.perplexity_tries = 50


    def l2_distance(self, X):
        sum_X = np.sum(X * X, axis=1)
        return (-2 * np.dot(X, X.T) + sum_X).T + sum_X


    def fit_transform(self, X, y=None):
        self._setup_input(X, y)

        Y = np.random.randn(self.n_samples, self.n_components)
        velocity = np.zeros_like(Y)
        gains = np.ones_like(Y)

        P = self._get_pairwise_affinities(X)

        iter_num = 0
        while iter_num < self.max_iter:
            iter_num += 1

            D = self.l2_distance(Y)
            Q = self._q_distribution(D)

            # Normalizer q distribution
            Q_n = Q / np.sum(Q)

            # Early exaggeration & momentum
            pmul = 4.0 if iter_num < 100 else 1.0
            momentum = 0.5 if iter_num < 20 else 0.8

            # Perform gradient step
            grads = np.zeros(Y.shape)
            for i in range(self.n_samples):
                grad = 4 * np.dot((pmul * P[i] - Q_n[i]) * Q[i], Y[i] - Y)
                grads[i] = grad

            gains = (gains + 0.2) * ((grads > 0) != (velocity > 0)) + (gains * 0.8) * ((grads > 0) == (velocity > 0))
            gains = gains.clip(min=self.min_gain)

            velocity = momentum * velocity - self.lr * (gains * grads)
            Y += velocity
            Y = Y - np.mean(Y, 0)

            error = np.sum(P * np.log(P / Q_n))
            logging.info("Iteration %s, error %s" % (iter_num, error))
        return Y

    def _get_pairwise_affinities(self, X):
        """Computes pairwise affinities."""
        affines = np.zeros((self.n_samples, self.n_samples), dtype=np.float32)
        target_entropy = np.log(self.perplexity)
        distances = self.l2_distance(X)

        for i in range(self.n_samples):
            affines[i, :] = self._binary_search(distances[i], target_entropy)

        # Fill diagonal with near zero value
        np.fill_diagonal(affines, 1.0e-12)

        affines = affines.clip(min=1e-100)
        affines = (affines + affines.T) / (2 * self.n_samples)
        return affines

    def _binary_search(self, dist, target_entropy):
        """Performs binary search to find suitable precision."""
        precision_min = 0
        precision_max = 1.0e15
        precision = 1.0e5

        for _ in range(self.perplexity_tries):
            denom = np.sum(np.exp(-dist[dist > 0.0] / precision))
            beta = np.exp(-dist / precision) / denom

            # Exclude zeros
            g_beta = beta[beta > 0.0]
            entropy = -np.sum(g_beta * np.log2(g_beta))

            error = entropy - target_entropy

            if error > 0:
                # Decrease precision
                precision_max = precision
                precision = (precision + precision_min) / 2.0
            else:
                # Increase precision
                precision_min = precision
                precision = (precision + precision_max) / 2.0

            if np.abs(error) < self.tol:
                break

        return beta

    def _q_distribution(self, D):
        """Computes Student t-distribution."""
        Q = 1.0 / (1.0 + D)
        np.fill_diagonal(Q, 0.0)
        Q = Q.clip(min=1e-100)
        return Q



