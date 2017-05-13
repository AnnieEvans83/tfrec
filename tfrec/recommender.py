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
        """Build a Recommender object.

        Parameters
        ----------
        k : int
            The number of latent factors to use in the matrix factorization.

        dtype : string in {'float32', 'float64'}
            The floating point precision to use for all matrix operations.

        lambda_factors : float
            The regularization hyperparameter applied to the factors (i.e. to the
            matrices `U` and `V` of the factorization.

        lambda_biases : float
            The regularization hyperparameter applied to the user-bases and item-
            biases.

        init_factor_mean : float
            The mean of the Gaussian distribution used to randomize the `U` and
            `V` matrices.

        init_factor_stddev : float
            The std. deviation of the Gaussian distribution used to randomize the `U` and
            `V` matrices.

        n_iter : int
            The number of iterations (aka, 'epochs') of gradient descent to run.

        learning_rate : float
            The learning rate (aka, `alpha` or `step size`) used by gradient descent.

        batch_size : int
            The batch size to use for gradient descent. If -1, the full-batch gradient
            descent is used. Otherwise if >0, Stochastic Gradient Descent will be used.
        """
        self.k = k
        self.dtype = dtype
        self.lambda_factors = lambda_factors
        self.lambda_biases = lambda_biases
        self.init_factor_mean = init_factor_mean
        self.init_factor_stddev = init_factor_stddev
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sess = None

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

        lambda_factors : float, optional (default=None)
            If not None, then override the `lambda_factors` given to `__init__()`.

        lambda_biases : float, optional (default=None)
            If not None, then override the `lambda_biases` given to `__init__()`.

        n_iter : int, optional (default=None)
            If not None, then override the `n_iter` given to `__init__()`.

        learning_rate : float, optional (default=None)
            If not None, then override the `learning_rate` given to `__init__()`.

        batch_size : int, optional (default=None)
            If not None, then override the `batch_size` given to `__init__()`.

        tune : bool, optional (default=False)
            If True and if `fit()` was previously called, then pick up where the
            the last `fit()` left off by training for more iterations with the
            given `X` and `y`. Note: You can pass _new_ `X` and `y` (and pass
            `tune=True`) in order train the model on _additional_ ratings without
            forgetting what was learned by previous calls to `fit()`. If the _new_
            `X` has `user_id`s and/or `item_id`s which were previously unseen,
            then the `U` and `V` matrices will be extended to accommodate these
            new users/items. Of course, probably a more common use-case will be
            to call `fit()` again using the same `X` and `y` as was used previously,
            allowing you to simply run more (S)GD iterations.
            (Final note: self.mu_ is not recalculated when the model is tuned in this
            way.)

        verbose : bool, optional (default=False)
            If True, log verbose output.

        log : logging.Logger, optional (default=LOCAL_LOG)
            If given, use this `log` when printing verbose output; otherwise
            use the `LOCAL_LOG`.
            If the special string 'unique' is given, then create a unique
            logger object for this recommender; such an operation is useful
            if you are doing a parallel gridsearch so that each recommender
            will build its own independent log output, making it easier for
            you to dig into the results of each cell of the gridsearch.

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

        # Do the tuning logic:
        tune = kwargs.get('tune', False)
        if tune and getattr(self, 'U_concat_bias_var', None) is not None and getattr(self, 'V_concat_bias_var', None) is not None:
            # We'll tune the matrices we already have. As such, we will not include the "unknown records"
            # because they have already been included in the first call the `fit()`. Also, we'll make
            # note of the current `U` and `V` matrices so we can extend them if needed.
            include_unknown_records = False
            old_U_concat_bias_var = self.U_concat_bias_var
            old_V_concat_bias_var = self.V_concat_bias_var
        else:
            # In this case, we should start everything fresh, treating the current data as the
            # only data ever seen.
            include_unknown_records = True
            old_U_concat_bias_var = None
            old_V_concat_bias_var = None
            self.user_to_index_map_ = None
            self.index_to_user_map_ = None
            self.item_to_index_map_ = None
            self.index_to_item_map_ = None
            self.num_users = 0
            self.num_items = 0
            self.completed_iters = 0
            self.mu_ = rating_array.mean()

        # Prepare the data by creating 0-based indices for the users and items,
        # and by counting number of unique users and items.
        user_indices, item_indices, num_users, num_items = \
                self._prep_data_for_train(user_array, item_array)

        # Keep track of how many new users and new itesm we have here.
        new_num_users = num_users - self.num_users
        new_num_items = num_items - self.num_items
        self.num_users = num_users
        self.num_items = num_items

        # Build the TensorFlow computation graph!
        self.needs_init = set()
        dtype = tf.float32 if (self.dtype == 'float32') else tf.float64
        if new_num_users > 0 or new_num_items > 0:
            self._build_computation_graph(dtype,
                                          new_num_users, new_num_items,
                                          include_unknown_records,
                                          old_U_concat_bias_var, old_V_concat_bias_var)

        # Start the TensorFlow session and initialize the variables.
        if self.sess is None:
            self.sess = tf.Session()
        self._init_variables()

        # Tell TensorFlow to run gradient descent for us! (...doing several epochs,
        # and optionally doing SGD rather than full-batch)
        log = kwargs.get('log', None)
        if log == 'unique':
            uuid = gen_uuid().hex[:12]
            log = logger.easy_setup(uuid, console_output=True, filename="log_{}.txt".format(uuid))
        if log is None:
            log = LOCAL_LOG
        self._run_gradient_descent(user_indices, item_indices, rating_array,
                                   kwargs.get('lambda_factors', self.lambda_factors),
                                   kwargs.get('lambda_biases', self.lambda_biases),
                                   kwargs.get('learning_rate', self.learning_rate),
                                   kwargs.get('n_iter', self.n_iter),
                                   kwargs.get('batch_size', self.batch_size),
                                   kwargs.get('verbose', False),
                                   log)

        return self

    def predict(self, X):
        """Predict the ratings of new user-item pairs.

        If users exist in `X` which were not seen by `fit()`, then each of
        these new user's predicted rating will be exactly the bias of the
        corresponding item. As such, the behaviour will roughly be an item-
        popularity recommender.

        If items exist in `X` which were not seen by `fit()`, then each of
        these new item's predicted ratings will be exactly the bias of the
        corresponding user.

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
        check_is_fitted(self, ['train_step_op'])
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

    def _prep_data_for_train(self, user_array, item_array):
        """Private helper method to prep the training set."""

        # The `user_id`s can be anything (strings, large integers, whatever),
        # so we want to convert them to be 0-based indices. We'll also keep maps
        # to go back-and-forth to convert from 0-based index to original value
        # and back again.
        user_indices, self.user_to_index_map_, self.index_to_user_map_ = \
                Recommender._convert_to_indices(user_array, self.user_to_index_map_, self.index_to_user_map_,
                                                allow_new_entries=True)

        # Same for the `item_id`s.
        item_indices, self.item_to_index_map_, self.index_to_item_map_ = \
                Recommender._convert_to_indices(item_array, self.item_to_index_map_, self.index_to_item_map_,
                                                allow_new_entries=True)

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
    def _convert_to_indices(values, value_to_index_map, index_to_value_map, allow_new_entries=False):
        """Private static helper method to convert opaque user- and item- values
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

    def _build_computation_graph(self, dtype,
                                 new_num_users, new_num_items,
                                 include_unknown_records,
                                 old_U_concat_bias_var, old_V_concat_bias_var):
        """Private helper method to build the TensorFlow computation graph."""

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
        if include_unknown_records:
            num_U_rows = new_num_users-1
            num_V_cols = new_num_items-1
        else:
            num_U_rows = new_num_users
            num_V_cols = new_num_items
        self.U_var = tf.Variable(tf.truncated_normal([num_U_rows, self.k],
                                                     mean=self.init_factor_mean_placeholder,
                                                     stddev=self.init_factor_stddev_placeholder,
                                                     dtype=dtype),
                                 name="user_factors_1")
        self.V_var = tf.Variable(tf.truncated_normal([self.k, num_V_cols],
                                                      mean=self.init_factor_mean_placeholder,
                                                      stddev=self.init_factor_stddev_placeholder,
                                                      dtype=dtype),
                                 name="item_factors_1")
        self.needs_init.add(self.U_var)
        self.needs_init.add(self.V_var)
        if include_unknown_records:
            self.U_var = tf.concat([tf.zeros([1, self.k]),
                                    self.U_var],
                                   0,
                                   name="user_factors_2")
            self.V_var = tf.concat([tf.zeros([self.k, 1]),
                                    self.V_var],
                                   1,
                                   name="item_factors_2")

        # Build the user- and item-bias vectors.
        self.user_biases_var = tf.Variable(tf.zeros([new_num_users, 1], dtype=dtype),
                                           name="user_biases")
        self.item_biases_var = tf.Variable(tf.zeros([1, new_num_items], dtype=dtype),
                                           name="item_biases")
        self.needs_init.add(self.user_biases_var)
        self.needs_init.add(self.item_biases_var)

        # For conveniance, let's concat the biases onto the end of the factor vectors.
        self.U_concat_bias_var = tf.concat([self.U_var,
                                            self.user_biases_var,
                                            tf.ones([new_num_users, 1], dtype=dtype)],
                                           1,
                                           name="user_factors_concat_user_biases")
        self.V_concat_bias_var = tf.concat([self.V_var,
                                            tf.ones([1, new_num_items], dtype=dtype),
                                            self.item_biases_var],
                                           0,
                                           name="item_factors_concat_item_biases")

        # If we are doing extended training (i.e. we are picking up where the last
        # training left off), then we will extend the user-factors and item-factors.
        if old_U_concat_bias_var is not None:
            self.U_concat_bias_var = tf.concat([old_U_concat_bias_var,
                                                self.U_concat_bias_var],
                                               0,
                                               name="user_factors_concat_user_biases")
        if old_V_concat_bias_var is not None:
            self.V_concat_bias_var = tf.concat([old_V_concat_bias_var,
                                                self.V_concat_bias_var],
                                               1,
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

    def _init_variables(self):
        """Private helper to init the TensorFlow globals."""

        init_params_dict = {
            self.init_factor_mean_placeholder: self.init_factor_mean,
            self.init_factor_stddev_placeholder: self.init_factor_stddev
        }

        init_op = tf.variables_initializer(self.needs_init)
        self.sess.run(init_op, feed_dict=init_params_dict)

    def _run_gradient_descent(self, user_indices, item_indices, rating_array,
                              lambda_factors, lambda_biases, learning_rate,
                              num_steps, batch_size, verbose, log):
        """Private helper that trains (S)GD iterations."""

        # Here's what to feed the session if you want to deal with the whole training set.
        full_train_batch_feed = {
            self.user_indices_placeholder: user_indices,
            self.item_indices_placeholder: item_indices,
            self.rating_array_placeholder: rating_array,
            self.mu_placeholder: self.mu_
        }

        # Verbosity is helpful...
        if verbose:
            log.info("Starting {}Gradient Descent for {} iterations".format('Stochastic ' if batch_size>0 else '', num_steps))
            begin_rmse = self.rmse_op.eval(session=self.sess, feed_dict=full_train_batch_feed)
            log.info("training set RMSE = {}".format(begin_rmse))

        # These are the hyperparameters needed for training.
        hyperparam_dict = {
            self.alpha_placeholder: learning_rate,
            self.lambda_factors_placeholder: lambda_factors,
            self.lambda_biases_placeholder: lambda_biases
        }

        # Train for a while...
        for _ in range(num_steps):
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
            self.completed_iters += 1
            if verbose:
                log.info("Finished iteration #{}".format(self.completed_iters))
                curr_rmse = self.rmse_op.eval(session=self.sess, feed_dict=feed_dict)
                log.info("{}training set RMSE = {}".format(prefix, curr_rmse))

        # Yay more verbosity.
        if verbose:
            log.info("Ending {}Gradient Descent".format('Stochastic ' if batch_size>0 else ''))
            end_rmse = self.rmse_op.eval(session=self.sess, feed_dict=full_train_batch_feed)
            log.info("training set RMSE = {}".format(end_rmse))

    def _predict(self, user_indices, item_indices):
        """Private helper to make predictions."""

        feed_dict = {
            self.user_indices_placeholder: user_indices,
            self.item_indices_placeholder: item_indices,
            self.mu_placeholder: self.mu_
        }

        # The `.eval` on a tf operation returns an ndarray.
        return self.reconstruction_gather_ratings_op.eval(session=self.sess, feed_dict=feed_dict)


