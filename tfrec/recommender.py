from __future__ import division, absolute_import, print_function
from . import logger
from uuid import uuid4 as gen_uuid
import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


LOCAL_LOG = logger.easy_setup(__name__, console_output=True)


class Recommender(BaseEstimator):
    """A fast recommender engine built on TensorFlow; created at Galvanize.

    This engine publishes a sklearn-style interfaces.

    Attributes
    ----------
    user_to_index_map_ : dict
        maps original `user_id`s to the indices used in the matrices

    index_to_user_map_ : dict
        maps the indices used in the matrices to the original `user_id`s

    item_to_index_map_ : dict
        maps original `item_id`s to the indices used in the matrices

    index_to_item_map_ : dict
        maps the indices used in the matrices to the original `item_id`s

    mu_ : float-type
        the global average of the training data

    See Also
    --------
    - http://www.galvanize.com/
    """

    _estimator_type = "regressor"

    def __init__(self, k=8, dtype='float32',
                       lambda_factors=0.1, lambda_biases=1e-4,
                       init_factor_mean=0.0, init_factor_stddev=0.01,
                       n_iter=10, learning_rate=0.00001, batch_size=-1):
        self.k = k
        self.dtype = dtype
        self.lambda_factors = lambda_factors
        self.lambda_biases = lambda_biases
        self.init_factor_mean = init_factor_mean
        self.init_factor_stddev = init_factor_stddev
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def fit(self, X, y, **kwargs):
        """Fit on user-item ratings data.

        Parameters
        ----------
        X : array-like, with shape (N, 2)
            The user-item data (`N` samples), where the columns contain:
            [`user_id`, `item_id`]

        y : array-like, with shape (N,)
            The ratings (`N` samples) corresponding to the user-item data
            held in `X`.

        n_iter : int, optional (default=None)
            If not None, then override the `n_iter` given to `__init__()`.

        tune : bool, optional (default=False)
            If `fit()` was previously called, then pick up where it left off by training
            for more iterations with the given `X` and `y`.

        batch_size : int, optional (default=None)
            If not None, then override the `batch_size` given to `__init__()`.

        verbose : bool, optional (default=False)
            If True, log verbose output.

        log : logging.Logger, optional (default=LOCAL_LOG)
            If given, use this `log` when printing verbose output; otherwise
            use the `LOCAL_LOG`.
            If the special string 'unique' is given, then create a unique
            logger object for this recommender; such an operation is useful
            if you are doing a parallel gridsearch so that each recommender
            will build their own independent log output.

        Returns
        -------
        self
        """

        # Have sklearn check and convert the inputs.
        X, y = check_X_y(X, y, dtype=None, y_numeric=True, estimator=self)

        # In our specific case (a recommender engine), there should be exactly two features.
        if X.ndim     != 2: raise ValueError("X must be a 2d ndarray")
        if X.shape[1] != 2: raise ValueError("X must have exactly 2 features")

        # Pull out the columns into better variable names.
        user_array   = X[:,0]
        item_array   = X[:,1]
        rating_array = y

        # Prepare the data by creating 0-based indices for the users and items,
        # and by counting number of unique users and items.
        user_indices, item_indices, num_users, num_items = \
                self._prep_data_for_train(user_array, item_array, rating_array)

        # Build the TensorFlow computation graph!
        dtype = tf.float32 if (self.dtype == 'float32') else tf.float64
        self._build_computation_graph(dtype, num_users, num_items)

        # Start the TensorFlow session.
        self._start_session()

        # Tell TensorFlow to run gradient descent for us! (...doing several epochs,
        # and optionally doing SGD rather than full-batch)
        log = kwargs.get('log', None)
        if log == 'unique':
            uuid = gen_uuid().hex[:12]
            log = logger.easy_setup(uuid, console_output=True, filename="log_{}.txt".format(uuid))
        if log is None:
            log = LOCAL_LOG
        self._run_gradient_descent(user_indices, item_indices, rating_array,
                                   kwargs.get('n_iter', self.n_iter),
                                   kwargs.get('batch_size', self.batch_size),
                                   kwargs.get('verbose', False),
                                   log)

        return self

    def predict(self, X):
        """Predict the ratings of new user-item pairs.

        Parameters
        ----------
        X : array-like, with shape (N, 2)
            The user-item data (`N` samples), where the columns contain:
            [`user_id`, `item_id`]

        Returns
        -------
        ndarray of predicted ratings of shape (N,)
        """

        # Have sklearn check the that fit has been called previously, and
        # have sklearn check and convert the inputs.
        check_is_fitted(self, ['index_to_user_map_', 'index_to_item_map_', 'train_step_op'])
        X = check_array(X, dtype=None, estimator=self)

        # In our specific case (a recommender engine), there should be exactly two features.
        if X.ndim     != 2: raise ValueError("X must be a 2d ndarray")
        if X.shape[1] != 2: raise ValueError("X must have exactly 2 features")

        # Pull out the columns into better variable names.
        user_array = X[:,0]
        item_array = X[:,1]

        # Prep the data by converting the users and items to the same 0-based
        # indices used by the `fit()` method.
        user_indices, item_indices = \
                self._prep_data_for_predict(user_array, item_array)

        # Make the predictions.
        return self._predict(user_indices, item_indices)

    def _prep_data_for_train(self, user_array, item_array, rating_array):
        """Private helper method to prep the training set."""

        # Compute the global average rating.
        self.mu_ = rating_array.mean()

        # The `user_id`s can be anything (strings, large integers, whatever),
        # so we want to convert them to be 0-based indices. We'll also keep maps
        # to go back-and-forth to convert from 0-based index to original value
        # and back again.
        user_indices, self.user_to_index_map_, self.index_to_user_map_ = \
                Recommender._convert_to_indices(user_array, allow_new_entries=True)

        # Same for the `item_id`s.
        item_indices, self.item_to_index_map_, self.index_to_item_map_ = \
                Recommender._convert_to_indices(item_array, allow_new_entries=True)

        # Note the number of unique users and items.
        num_users = len(self.user_to_index_map_)
        num_items = len(self.item_to_index_map_)

        return user_indices, item_indices, num_users, num_items

    def _prep_data_for_predict(self, user_array, item_array):
        """Private helper method to prep the out-of-sample dataset."""

        # The `user_id`s can be anything (strings, large integers, whatever),
        # so we want to convert them to be 0-based indices. We'll also keep maps
        # to go back-and-forth to convert from 0-based index to original value
        # and back again.
        user_indices, _, _ = \
                Recommender._convert_to_indices(user_array, self.user_to_index_map_, self.index_to_user_map_)

        # Same for the `item_id`s.
        item_indices, _, _ = \
                Recommender._convert_to_indices(item_array, self.item_to_index_map_, self.index_to_item_map_)

        return user_indices, item_indices

    @staticmethod
    def _convert_to_indices(values, value_to_index_map=None, index_to_value_map=None, allow_new_entries=False):
        """Static helper method to convert opaque user- and item- values
        into 0-based-indices.
        """
        indices = []
        if value_to_index_map is None:
            value_to_index_map = {'__unknown__': 0}
        if index_to_value_map is None:
            index_to_value_map = {0: '__unknown__'}
        if allow_new_entries:
            for value in values:
                if value not in value_to_index_map:
                    next_index = len(value_to_index_map)
                    value_to_index_map[value] = next_index
                    index_to_value_map[next_index] = value
                indices.append(value_to_index_map[value])
        else:
            for value in values:
                if value not in value_to_index_map:
                    indices.append(0)
                else:
                    indices.append(value_to_index_map[value])
        indices = np.array(indices)
        return indices, value_to_index_map, index_to_value_map

    def _build_computation_graph(self, dtype, num_users, num_items):

        # Create the user, item, and rating tf placeholders.
        self.user_indices_placeholder = tf.placeholder(dtype=tf.int32, name="user_indices")
        self.item_indices_placeholder = tf.placeholder(dtype=tf.int32, name="item_indices")
        self.rating_array_placeholder = tf.placeholder(dtype=dtype, name="rating_array")

        # This tf placeholder will hold the learning rate, alpha.
        self.alpha_placeholder = tf.placeholder(dtype=dtype, name="alpha")

        # Create the regularization (lambda) tf placeholders.
        # These are placeholders _not_ because you'll want to change them
        # while training (probably...), but intead they're placeholders so
        # that we don't have to re-create this whole tf computation graph
        # when doing a grid-search.
        self.lambda_factors_placeholder = tf.placeholder(dtype=dtype, name="lambda_factors")
        self.lambda_biases_placeholder  = tf.placeholder(dtype=dtype, name="lambda_biases")

        # Placeholder for mu, the average rating in the dataset. This is fixed and is not learned.
        # This is a placeholder _not_ because you'll want to change it
        # while training (probably...), but intead it is a placeholder so
        # that we can create the tf computation graph prior to knowing this value.
        self.mu_placeholder = tf.placeholder(dtype=dtype, name="mu")

        # The random normal parameters to initialize the user- and item-factors (as placeholders).
        self.init_factor_mean_placeholder = tf.placeholder(dtype=dtype, shape=(),
                                                           name="init_factor_mean")
        self.init_factor_stddev_placeholder = tf.placeholder(dtype=dtype, shape=(),
                                                             name="init_factor_stddev")

        # U will represent user-factors, and V will represent item-factors.
        self.U_var = tf.Variable(tf.truncated_normal([num_users-1, self.k],
                                                     mean=self.init_factor_mean_placeholder,
                                                     stddev=self.init_factor_stddev_placeholder,
                                                     dtype=dtype),
                                 name="user_factors_no_unknown_row")
        self.U_var = tf.concat([tf.zeros([1, self.k]),
                                self.U_var],
                               0,
                               name="user_factors")
        self.V_var = tf.Variable(tf.truncated_normal([self.k, num_items-1],
                                                      mean=self.init_factor_mean_placeholder,
                                                      stddev=self.init_factor_stddev_placeholder,
                                                      dtype=dtype),
                                 name="item_factors_no_unknown_col")
        self.V_var = tf.concat([tf.zeros([self.k, 1]),
                                self.V_var],
                               1,
                               name="item_factors")

        # Build the user- and item-bias vectors.
        self.user_biases_var = tf.Variable(tf.zeros([num_users, 1], dtype=dtype),
                                           name="user_biases")
        self.item_biases_var = tf.Variable(tf.zeros([1, num_items], dtype=dtype),
                                           name="item_biases")

        # For conveniance, let's concat the biases onto the end of the factor vectors.
        self.U_concat_bias_var = tf.concat([self.U_var,
                                            self.user_biases_var,
                                            tf.ones([num_users, 1], dtype=dtype)],
                                           1,
                                           name="user_factors_concat_user_biases")
        self.V_concat_bias_var = tf.concat([self.V_var,
                                            tf.ones([1, num_items], dtype=dtype),
                                            self.item_biases_var],
                                           0,
                                           name="item_factors_concat_item_biases")

        # The model:
        self.centered_reconstruction_op = tf.matmul(self.U_concat_bias_var, self.V_concat_bias_var,
                                                    name="centered_reconstruction")

        # For training, we don't need the whole reconstruction matrix above. We
        # only need the indices of the user/item pairs for which we have _known_
        # ratings. The numpy-equivalent of the tf code below would be:
        #   reconstruction_gather_ratings = reconstruction[user_indices, item_indices]
        # See:
        #   https://github.com/tensorflow/tensorflow/issues/206
        #   https://github.com/tensorflow/tensorflow/issues/418
        self.centered_reconstruction_gather_ratings_op = tf.gather(
                    tf.reshape(self.centered_reconstruction_op, [-1]),
                    self.user_indices_placeholder * tf.shape(self.centered_reconstruction_op)[1]
                                                              + self.item_indices_placeholder,
                    name="centered_reconstruction_gather_ratings")

        # Add the average rating to the centered reconstruciton, to make it a non-centered reconstruction.
        self.reconstruction_gather_ratings_op = tf.add(self.centered_reconstruction_gather_ratings_op,
                                                       self.mu_placeholder,
                                                       name="reconstruction_gather_ratings")

        # Calculate the reconstruction residuals.
        self.residual_op = tf.subtract(self.reconstruction_gather_ratings_op,
                                       self.rating_array_placeholder,
                                       name="residuals")

        # Calculate the RSS (residual sum of squares), MSE (mean squared error), and the RMSE (root mean squared error).
        self.rss_op = tf.reduce_sum(tf.square(self.residual_op), name="rss")
        self.mse_op = tf.divide(self.rss_op, tf.cast(tf.shape(self.rating_array_placeholder)[0], dtype=dtype), name="mse")
        self.rmse_op = tf.sqrt(self.mse_op, name="rmse")

        # Declare the factor regularizer!!!
        self.U_square_op = tf.square(self.U_var)
        self.U_sum_rows_op = tf.reduce_sum(self.U_square_op, 1)
        self.U_sum_penalty_op = tf.reduce_sum(tf.gather(self.U_sum_rows_op, self.user_indices_placeholder))
        self.V_square_op = tf.square(self.V_var)
        self.V_sum_cols_op = tf.reduce_sum(self.V_square_op, 0)
        self.V_sum_penalty_op = tf.reduce_sum(tf.gather(self.V_sum_cols_op, self.item_indices_placeholder))
        self.factor_regularizer_op = tf.multiply(tf.add(self.U_sum_penalty_op, self.V_sum_penalty_op), self.lambda_factors_placeholder,
                                                 name="factor_regularizer")

        # Declare the biases regularizer!!!
        self.user_biases_square_op = tf.square(self.user_biases_var)
        self.user_biases_sum_op = tf.reduce_sum(tf.gather(tf.reshape(self.user_biases_square_op, [-1]), self.user_indices_placeholder))
        self.item_biases_square_op = tf.square(self.item_biases_var)
        self.item_biases_sum_op = tf.reduce_sum(tf.gather(tf.reshape(self.item_biases_square_op, [-1]), self.item_indices_placeholder))
        self.bias_regularizer_op = tf.multiply(tf.add(self.user_biases_sum_op, self.item_biases_sum_op), self.lambda_biases_placeholder,
                                               name="bias_regularizer")

        # Declare the cost function!!!
        self.cost_op = tf.add(self.rss_op, tf.add(self.factor_regularizer_op, self.bias_regularizer_op), name="cost")

        # TensorFlow's magic graident descent optimization:
        self.optimizer = tf.train.GradientDescentOptimizer(self.alpha_placeholder)
        self.train_step_op = self.optimizer.minimize(self.cost_op)

    def _start_session(self):
        """Private helper to start a TensorFlow session and init the TensorFlow globals."""

        init_params_dict = {
            self.init_factor_mean_placeholder: self.init_factor_mean,
            self.init_factor_stddev_placeholder: self.init_factor_stddev
        }

        # The global variable initalization operation + a new TensorFlow Session!
        self.sess = None
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op, feed_dict=init_params_dict)

    def _run_gradient_descent(self, user_indices, item_indices, rating_array,
                              num_steps, batch_size, verbose, log):
        """
        Trains for (S)GD iterations, and returns the RMSE of the entire training set.
        """

        # Here's what to feed the session if you want to deal with the whole training set.
        full_train_batch_feed = {
            self.user_indices_placeholder: user_indices,
            self.item_indices_placeholder: item_indices,
            self.rating_array_placeholder: rating_array,
            self.mu_placeholder: self.mu_
        }

        if verbose:
            log.info("Starting {}Gradient Descent".format('Stochastic ' if batch_size>0 else ''))
            begin_rmse = self.rmse_op.eval(session=self.sess, feed_dict=full_train_batch_feed)
            log.info("training set RMSE = {}".format(begin_rmse))

        hyperparam_dict = {
            self.alpha_placeholder: self.learning_rate,
            self.lambda_factors_placeholder: self.lambda_factors,
            self.lambda_biases_placeholder: self.lambda_biases
        }

        for i in range(num_steps):
            if batch_size <= 0:
                feed_dict = dict(full_train_batch_feed)
                prefix = ""
            else:
                rand_indices = np.random.choice(len(rating_array), size=batch_size, replace=False)
                feed_dict = {
                    self.user_indices_placeholder: user_indices[rand_indices],
                    self.item_indices_placeholder: item_indices[rand_indices],
                    self.rating_array_placeholder: rating_array[rand_indices],
                    self.mu_placeholder: self.mu_
                }
                prefix = "approx. "
            feed_dict.update(hyperparam_dict)
            self.train_step_op.run(session=self.sess, feed_dict=feed_dict)
            if verbose:
                log.info("Finished iteration #{}".format(i))
                curr_rmse = self.rmse_op.eval(session=self.sess, feed_dict=feed_dict)
                log.info("{}training set RMSE = {}".format(prefix, curr_rmse))

        return self.rmse_op.eval(session=self.sess, feed_dict=full_train_batch_feed)

    def _predict(self, user_indices, item_indices):

        feed_dict = {
            self.user_indices_placeholder: user_indices,
            self.item_indices_placeholder: item_indices,
            self.mu_placeholder: self.mu_
        }
        return self.reconstruction_gather_ratings_op.eval(session=self.sess, feed_dict=feed_dict)


