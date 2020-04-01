#!/usr/bin/env python3
import os
from time import time
from pprint import pprint

import pandas as pd
# import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


df = pd.read_csv(os.path.join('data', 'cleaned_reviews.csv'))

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(df.clean_text,
                                                    df.label,
                                                    random_state=42)

# Build a pipeline (vectorizer => transformer => classifier)
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier()),
])

parameters = {
    # 'clf__n_estimators': [10, 50, 110],
    # 'clf__max_depth': [4, 6, 8, 10, None],
    # 'clf__min_samples_leaf': [1, 4, 6, 7],
    # 'clf__min_samples_split': [2, 4, 5],
    # 'clf__max_features': [2, 3, 4],

        'clf__n_estimators': [5, 10, 30],
        'clf__max_depth': [3, 4],
        'clf__min_samples_leaf': [1, 2, 3],
        'clf__min_samples_split': [2, 3],
        'clf__max_features': [1, 2],
}

if __name__ == '__main__':
    gs_clf = GridSearchCV(text_clf,
                          parameters,
                          cv=5,
                          verbose=1,
                          n_jobs=-1
                          )
    print("Performing grid search...")
    print("parameters:")
    pprint(parameters)
    t0 = time()
    # Search on a subset to speed up
    gs_clf = gs_clf.fit(X_train[:4000], y_train[:4000])
    print(f'Done in {time() - t0} seconds')
    print('Best parameters:')
    print(gs_clf.best_params_)


# Performing grid search...
# parameters:
# {'clf__max_depth': [4, 6, 8, 10, None],
#  'clf__max_features': [2, 3, 4],
#  'clf__min_samples_leaf': [1, 4, 6, 7],
#  'clf__min_samples_split': [2, 4, 5],
#  'clf__n_estimators': [10, 50, 110]}
# Fitting 5 folds for each of 540 candidates, totalling 2700 fits
# [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  2.0min
# [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  7.3min
# [Parallel(n_jobs=-1)]: Done 442 tasks      | elapsed: 16.5min
# [Parallel(n_jobs=-1)]: Done 792 tasks      | elapsed: 33.5min
# [Parallel(n_jobs=-1)]: Done 1242 tasks      | elapsed: 52.1min
# [Parallel(n_jobs=-1)]: Done 1792 tasks      | elapsed: 73.8min
# [Parallel(n_jobs=-1)]: Done 2442 tasks      | elapsed: 101.1min
# [Parallel(n_jobs=-1)]: Done 2700 out of 2700 | elapsed: 112.3min finished
# Done in 6776.927423000336 seconds
# Best parameters:
# {'clf__max_depth': +4, 'clf__max_features': 2, 'clf__min_samples_leaf': 1, 'clf__min_samples_split': 2, 'clf__n_estimators': 10}

#
# Performing grid search...
# parameters:
# {'clf__max_depth': [3, 4],
#  'clf__max_features': [1, 2],
#  'clf__min_samples_leaf': [1, 2, 3],
#  'clf__min_samples_split': [2, 3],
#  'clf__n_estimators': [5, 10, 30]}
# Fitting 5 folds for each of 72 candidates, totalling 360 fits
# [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  1.7min
# [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  6.6min
# [Parallel(n_jobs=-1)]: Done 360 out of 360 | elapsed: 12.3min finished
# Done in 765.8352587223053 seconds
# Best parameters:
# {'clf__max_depth': 3, 'clf__max_features': 1, 'clf__min_samples_leaf': 1, 'clf__min_samples_split': 2, 'clf__n_estimators': 5}
