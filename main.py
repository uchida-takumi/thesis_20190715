#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is main program.
 1. load data from src/modules/inputs.py
 2. define same interface of recommendater models
 3. evaluate models with src/modules/evaluatinos/*
"""
#######################
# 1. load data from src/modules/inputs.py
import os
import pandas as pd
csv_fp = 'data/ml-latest-small/ratings.csv'
data = pd.read_csv(csv_fp)
column_names = ['userId', 'movieId', 'timestamp']
label_name = 'rating'

X, y = data[column_names].values, data[label_name].values

#### for test ####
#X, y = X[:10000], y[:10000]
##################
max_user = int(X[:,0].max()) + 1; max_item = int(X[:,1].max()) + 1


########################
# 2. define same interface of recommendater models

from src.surprise_algo_wrapper import surprise_algo_wrapper
from src.keras_model_wrapper import keras_model_wrapper

from surprise import SVD
svd = SVD()
svd = surprise_algo_wrapper(svd)

from src.DNN_recommender import RFNN
rfnn = RFNN(max_user, max_item, embedding_size=8, dnn_hidden_units=(128, 128), l2_reg=1e-5)
rfnn = keras_model_wrapper(rfnn, epochs=5,batch_size=128) # メモ：epochs=5,batch_size=128が最高性能が出たことを手動で検証。

from src.DNN_recommender import R_Wide_and_Deep
rwd = R_Wide_and_Deep(max_user, max_item, embedding_size=8, dnn_hidden_units=(128, 128), l2_reg=1e-5)
rwd = keras_model_wrapper(rwd, epochs=5,batch_size=128)


########################
# 3. evaluate models with src/modules/evaluatinos/*
from src.evaluation import cv

k_hold = 10
need_hit = True

svd_result  = cv(svd, X, y, k_hold=k_hold, need_hit=need_hit, seed=123)
rfnn_result = cv(rfnn, X, y, k_hold=k_hold, need_hit=need_hit, seed=123)
rwd_result  = cv(rwd, X, y, k_hold=k_hold, need_hit=need_hit, seed=123)

for model_name, _result in [('svd',svd_result), ('rfnn',rfnn_result), ('rwd',rwd_result)]:
    for key in ['metrics_by_labeled_user', 'metrics_by_labeled_item', 'metrics_by_labeled_user_item']:
        fp = os.path.join('output', '.'.join([model_name, key, 'csv']))
        _result['total_mean'][key].to_csv(fp)
    


if __name__ == 'grid search':

    ############################
    # manuarl grid search
    k_hold = 3
    hit_test_size = 100
    args = [ # 以下でgred-searchした結果、どれでも良いとなった。
            {'embedding_size':2, 'dnn_hidden_units':(128,128), 'l2_reg':1e-5},
            {'embedding_size':4, 'dnn_hidden_units':(128,128), 'l2_reg':1e-5},
            {'embedding_size':8, 'dnn_hidden_units':(128,128), 'l2_reg':1e-5},
            {'embedding_size':16, 'dnn_hidden_units':(128,128), 'l2_reg':1e-5},
            {'embedding_size':32, 'dnn_hidden_units':(128,128), 'l2_reg':1e-5}, #embedding_size=32が良いかも
            {'embedding_size':64, 'dnn_hidden_units':(128,128), 'l2_reg':1e-5},
            {'embedding_size':8, 'dnn_hidden_units':(32,32), 'l2_reg':1e-5},
            {'embedding_size':8, 'dnn_hidden_units':(64,64), 'l2_reg':1e-5},
            {'embedding_size':8, 'dnn_hidden_units':(256,256), 'l2_reg':1e-5},#dnn_hidden_unitsは正直どれでも良い
            {'embedding_size':8, 'dnn_hidden_units':(128,128), 'l2_reg':1e-5, 'drop_out_rate':0.1},
            {'embedding_size':8, 'dnn_hidden_units':(128,128), 'l2_reg':1e-5, 'drop_out_rate':0.3},
            {'embedding_size':8, 'dnn_hidden_units':(128,128), 'l2_reg':1e-5, 'drop_out_rate':0.5},
            ]
    
    
    
    
    results = []
    for arg in args:
        rfnn = RFNN(max_user, max_item, **arg)
        rfnn = keras_model_wrapper(rfnn, epochs=5, batch_size=128)
        result = cv(rfnn, X, y, k_hold=k_hold, need_hit=False)
        results.append(result)
    
    import numpy as np
    for r in results:
        print(r['random_split:mae'])
        print(np.mean(r['random_split:mae']))
