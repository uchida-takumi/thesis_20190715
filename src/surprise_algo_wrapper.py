#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wrapper function for suprise.alogo to be able to use
.fit(user_ids, item_ids, rating) and .predict(user_ids, item_ids)
"""

import pandas as pd

from surprise import Dataset
from surprise import Reader

from copy import deepcopy

class surprise_algo_wrapper:
    def __init__(self, algo, X_columns={'user':0, 'item':1, 'timestamp':2}):
        self.algo = algo
        self.init_algo = deepcopy(algo)
        self.X_columns = X_columns

    def fit(self, X, y):
        trainset = convert_to_Surprise_dataset(X, y, self.X_columns)
        self.algo.fit(trainset)

    def predict(self, X):
        predicted_result = []
        for u,i in X[:, [self.X_columns['user'], self.X_columns['item']]]:
            predicted_result.append(self.algo.predict(u,i).est)
        return predicted_result
    
    def reset(self):
        self.algo = self.init_algo


def convert_to_Surprise_dataset(X, y, X_columns={'user':0, 'item':1}):
    ratings_dict = {
            'user': X[:,X_columns['user']],
            'item': X[:,X_columns['item']],
            'rating': y,
            }
    df = pd.DataFrame(ratings_dict)

    # The columns must correspond to user id, item id and ratings (in that order).
    reader = Reader(rating_scale=(min(y), max(y)))
    dataset = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)

    return dataset.build_full_trainset()

if __name__ == 'how to use':
    from surprise import SVD
    # INPUTs
    import pandas as pd
    csv_fp = 'data/ml-latest-small/ratings.csv'
    data = pd.read_csv(csv_fp)
    column_names = ['userId', 'movieId', 'timestamp']
    label_name = 'rating'
    
    X, y = data[column_names].values, data[label_name].values

    algo = SVD()
    algo_w = surprise_algo_wrapper(algo)

    algo_w.fit(X, y)
    algo_w.predict(X)
    algo_w.reset()
    
