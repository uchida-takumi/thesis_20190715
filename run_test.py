#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 15:50:29 2019

@author: takumi_uchida
"""

import pandas as pd
csv_fp = 'data/ml-latest-small/ratings.csv'
data = pd.read_csv(csv_fp)
column_names = ['userId', 'movieId', 'timestamp']
label_name = 'rating'

X, y = data[column_names].values, data[label_name].values
X, y = X[:10000], y[:10000]


# 自作のgrid_search　モジュールを読み込みます。
from src.grid_search import grid_search



from surprise import SVD  
from src.surprise_algo_wrapper import surprise_algo_wrapper # I/F統一用のラッパー

model_module = SVD
wrapper = surprise_algo_wrapper

params = [
        {'n_factors':4,   'reg_all':0.010},
        {'n_factors':8,   'reg_all':0.010}, # the BEST
        {'n_factors':16,   'reg_all':0.010},
        {'n_factors':8,   'reg_all':0.001}, 
        {'n_factors':8,   'reg_all':0.100}, 
        ]
models = [wrapper(model_module(**param)) for param in params]
best_indice, scores, each_scores = grid_search(X, y, models)    
print('scores={}'.format(scores))
print('best_param={}'.format(params[best_indice]))
print('each_scores={}'.format(each_scores))


# モデルを読み込み
from src.control_model import random_model, popular_model
from surprise import SVD  # matrix factorization
from pyfm.pylibfm import FM # factorization machine
from src.DNN_recommender import RFNN # Re-FNN
from src.DNN_recommender import R_Wide_and_Deep # Re-FNN

# I/F統一用のラッパー
from src.surprise_algo_wrapper import surprise_algo_wrapper 
from src.pyfm_model_wrapper import pyfm_model_wrapper
from src.keras_model_wrapper import keras_model_wrapper 

max_user, max_item = X[:,0].max()+1, X[:,1].max()+1

# 以下がグリッドサーチによってチューニングされた変数とする。
best_models = {
    "random" : random_model() ,
    "popular" : popular_model(),
    "svd" : surprise_algo_wrapper(SVD(n_factors=8,  reg_all=0.01)),
    "fm"  : pyfm_model_wrapper(FM(num_factors=8, num_iter=5, reg_0=0.0, reg_w=0.01, reg_v=0.01, validation_size=0.01, task='regression')),
    "rfnn" : keras_model_wrapper(RFNN, dict(max_user=max_user, max_item=max_item, embedding_size=8, dnn_hidden_units=(64), l2_reg=0.01)),
    "rmd" : keras_model_wrapper(R_Wide_and_Deep, dict(max_user=max_user, max_item=max_item, embedding_size=8, dnn_hidden_units=(64), l2_reg=0.01)),
    }


from src.evaluation import cv
k_hold = 5
results = {'total_mean':[], 'metrics_by_labeled_user':[], 'metrics_by_labeled_item':[], 'metrics_by_labeled_user_item':[]}

for name, model in best_models.items():
    print(f"==== {name} ===")
    _result = cv(model, X, y, k_hold=k_hold, need_hit=True, seed=999)
    
    # result of total_mean
    _dict = {k:v for k,v in _result['total_mean'].items() if k not in results}
    _df = pd.DataFrame(_dict, index=[1])
    _df['model'] = name
    results['total_mean'].append(_df)
    
    # result of metrics_by_labeled_*
    for key in [key for key in results if key not in ('total_mean')]:
        _df = _result['total_mean'][key]
        _df['model'] = name
        results[key].append(_df)
        

# cross validation の結果をresults_dfに整理して、pickleとして保存する。
results_df = {}
for name, df_list in results.items():
    results_df[name] = pd.concat(df_list, axis=0)

# pickle として保存しておく。
import pickle
with open('output/results_df.pickle', 'wb') as f:
    pickle.dump(results_df, f)

print('以下の結果が、results_dfには格納されている。')    
print(results_df.keys())

# 例えば、itemのsparse度合いごとの精度は以下の通りである。
results_df['metrics_by_labeled_item']

print(results_df)
