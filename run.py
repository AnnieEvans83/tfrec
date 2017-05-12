from tfrec import Recommender

import pandas as pd
training_set = pd.read_csv('jokes_case_study/jester_train.csv')
test_set = pd.read_csv('jokes_case_study/jester_test.csv')

from jokes_case_study import scoring

X = training_set[['user_id', 'joke_id']].values
y = training_set['rating'].values

model = Recommender()
model.fit(X, y, verbose=True)
print model.predict(X)

