from __future__ import division, absolute_import, print_function

from tfrec import Recommender, logger

from jokes_case_study import scoring

import numpy as np
import pandas as pd


log = logger.easy_setup(__name__, console_output=True)


training_set = pd.read_csv('jokes_case_study/jester_train.csv')
test_set = pd.read_csv('jokes_case_study/jester_test.csv')

X = training_set[['user_id', 'joke_id']].values
y = training_set['rating'].values

model = Recommender()

for _ in range(5):

    model.fit(X, y, verbose=True, tune=True, n_iter=100)
    y_pred = model.predict(X)

    manual_rmse = np.sqrt(np.mean((y_pred - y) ** 2))
    log.info('Manual Training RMSE: {}'.format(manual_rmse))

    predictions = test_set.copy()
    predictions['rating'] = model.predict(test_set[['user_id', 'joke_id']].values)
    log.info('Test Set RMSE: {}'.format(scoring.score_rmse(predictions)))
    log.info('Test Set top-5-percent score: {}'.format(scoring.score_top_5_percent(predictions)))

