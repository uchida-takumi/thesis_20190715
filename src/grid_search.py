#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid search の手順を定義します。
"""

from src.evaluation import cv

def grid_search(X, y, models, metric='hit_top_50_precision', k_hold=3, seed=999):
    scores, each_scores = [], []
    for model in models:
        cv_result = cv(model, X, y, k_hold=k_hold, need_hit=True, seed=seed)
        score = cv_result['total_mean'][metric]
        scores.append(score)
        each_score = cv_result['each_k_hold'][metric]
        each_scores.append(each_score)
    best_indice = scores.index(max(scores))
    return best_indice, scores, each_scores
    
if __name__ == "how to use":
    import pandas as pd
    csv_fp = 'data/ml-latest-small/ratings.csv'
    data = pd.read_csv(csv_fp)
    column_names = ['userId', 'movieId', 'timestamp']
    label_name = 'rating'    
    X, y = data[column_names].values, data[label_name].values
    X, y = X[:1000], y[:1000] 

    from surprise import SVD
    from src.surprise_algo_wrapper import surprise_algo_wrapper
    key_args = [
            {'n_factors':10,  'reg_all':0.02},
            {'n_factors':50,  'reg_all':0.02},
            {'n_factors':100, 'reg_all':0.02},
            {'n_factors':10,  'reg_all':0.00},
            {'n_factors':10,  'reg_all':0.01},
            {'n_factors':10,  'reg_all':0.10},
            ]
    models = [surprise_algo_wrapper(SVD(**key_arg)) for key_arg in key_args]

    best_indice, scores, each_scores = grid_search(X, y, models)    
    print(scores)
    print(key_args[best_indice])
    