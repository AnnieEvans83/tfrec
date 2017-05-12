from __future__ import division, absolute_import, print_function
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


# TODO: Standarize docstrings
# TODO: Write docs and publish on readthedocs
# TODO: Package and Publish on PyPI


class Recommender(BaseEstimator):
    """A fast recommender engine built on TensorFlow; created at Galvanize.

    This engine publishes a sklearn-style interfaces.

    See Also
    --------
    - http://www.galvanize.com/
    - https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
    - http://scikit-learn.org/stable/developers/
    - http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
    """

    _estimator_type = "regressor"

    def __init__(self, k=8, dtype='float32',
                       lambda_factors=0.1, lambda_biases=1e-4,
                       n_iter=1000, learning_rate=0.00001):
        self.k = k
        self.dtype = dtype
        self.lambda_factors = lambda_factors
        self.lambda_biases = lambda_biases
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """
        Fit on user-item ratings data.

        Parameters
        ----------
        X : array-like, with shape (N, 2)
            The user-item data (`N` samples), where the columns contain:
            [`user_id`, `item_id`]
        y : array-like, with shape (N,)
            The ratings (`N` samples) corresponding to the user-item data
            held in `X`.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)

        self.value_to_index_map_ = {}
        self.index_to_value_map_ = {}

        self.real_dtype = tf.float32 if (self.dtype == 'float32') else tf.float64

        return self

    def predict(self, X):
        check_is_fitted(self, ['value_to_index_map_', 'index_to_value_map_'])
        X = check_array(X)
        return np.zeros(shape=(len(X),))

    @staticmethod
    def _convert_to_indices(values, value_to_index_map=None, index_to_value_map=None):
        """Static helper method to convert opaque user- and item- values
        into 0-based-indices.
        """
        if value_to_index_map is None:
            value_to_index_map = {}
        if index_to_value_map is None:
            index_to_value_map = {}
        for value in values:
            if value not in value_to_index_map:
                next_index = len(value_to_index_map)
                value_to_index_map[value] = next_index
                index_to_value_map[next_index] = value
        indices = np.array([value_to_index_map[value] for value in values])
        return indices, value_to_index_map, index_to_value_map


if __name__ == '__main__':
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(Recommender)

