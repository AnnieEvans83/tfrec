from __future__ import division, absolute_import, print_function
from tfrec import Recommender
import numpy as np

import pandas as pd
training_set = pd.read_csv('jokes_case_study/jester_train.csv')
test_set = pd.read_csv('jokes_case_study/jester_test.csv')

from jokes_case_study import scoring

X = training_set[['user_id', 'joke_id']].values
y = training_set['rating'].values

model = Recommender()
model.fit(X, y, verbose=True)
y_pred = model.predict(X)

manual_rmse = np.sqrt(np.mean((y_pred - y) ** 2))
print('Manual Training RMSE:', manual_rmse)

predictions = test_set.copy()
predictions['rating'] = model.predict(test_set[['user_id', 'joke_id']].values)
print('Test Set RMSE:', scoring.score_rmse(predictions))
print('Test Set top-5-percent score:', scoring.score_top_5_percent(predictions))

