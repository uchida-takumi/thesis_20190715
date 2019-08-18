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


########################
# 2. define same interface of recommendater models
max_user = int(X[:,0].max()) + 1; max_item = int(X[:,1].max()) + 1

from src.surprise_algo_wrapper import surprise_algo_wrapper
from src.keras_model_wrapper import keras_model_wrapper

from surprise import SVD
from src.DNN_recommender import RFNN
from src.DNN_recommender import R_Wide_and_Deep
from src.control_model import random_model, popular_model

random = random_model()
pop = popular_model()

model = SVD()
svd = surprise_algo_wrapper(model)

model = RFNN(max_user, max_item, l2_reg=1e-5)
rfnn = keras_model_wrapper(model)

model = R_Wide_and_Deep(max_user, max_item, l2_reg=1e-5)
rwd = keras_model_wrapper(model)

model = R_Wide_and_Deep(max_user, max_item, l2_reg=1e-5, fix_global_bias=3.5)
rwd_fix = keras_model_wrapper(model)

model = R_Wide_and_Deep(max_user, max_item, l2_reg=2e-3)
rwd_reg = keras_model_wrapper(model)

model = R_Wide_and_Deep(max_user, max_item, l2_reg=2e-3, fix_global_bias=3.5)
rwd_fix_reg = keras_model_wrapper(model)

########################
# 3. evaluate models with src/modules/evaluatinos/*
from src.evaluation import cv

k_hold = 1
need_hit = True

random_result = cv(random, X, y, k_hold=k_hold, need_hit=need_hit, seed=123)
pop_result = cv(pop, X, y, k_hold=k_hold, need_hit=need_hit, seed=123)
svd_result  = cv(svd, X, y, k_hold=k_hold, need_hit=need_hit, seed=123)
rfnn_result = cv(rfnn, X, y, k_hold=k_hold, need_hit=need_hit, seed=123)
rwd_result  = cv(rwd, X, y, k_hold=k_hold, need_hit=need_hit, seed=123)
rwd_fix_result = cv(rwd_fix, X, y, k_hold=k_hold, need_hit=need_hit, seed=123)
rwd_reg_result = cv(rwd_reg, X, y, k_hold=k_hold, need_hit=need_hit, seed=123)
rwd_fix_reg_result  = cv(rwd_fix_reg, X, y, k_hold=k_hold, need_hit=need_hit, seed=123)

#　結果をcsvで保存しておきます。
for model_name, _result in [('random',random_result), ('pop',pop_result), ('svd',svd_result), ('rfnn',rfnn_result), ('rwd',rwd_result), ('rwd_fix',rwd_fix_result), ('rwd_reg',rwd_reg_result), ('rwd_fix_reg',rwd_fix_reg_result)]:
    for key in ['metrics_by_labeled_user', 'metrics_by_labeled_item', 'metrics_by_labeled_user_item']:
        fp = os.path.join('output', '.'.join([model_name, key, 'csv']))
        _result['total_mean'][key].to_csv(fp)


''' 結果分析用
# svd に比べ、表現力の広いDeepLearningはhitの性能が劣る。
# しかし、これを悪いとは決定づけることはできない。未知の最適な組み合わせを上位にあげている可能性を否定できない。


import numpy as np
import pandas as pd
X_train, y_train = X[:80000], y[:80000]
X_test, y_test = X[80000:], y[80000:]

svd = SVD()
svd = surprise_algo_wrapper(svd)

#model = RFNN(max_user, max_item, embedding_size=8, fix_global_bias=None, dnn_hidden_units=(128, 128), l2_reg=1e-5)
#rfnn = keras_model_wrapper(model, epochs=5,batch_size=128)

#model = RFNN(max_user, max_item, embedding_size=8, fix_global_bias=y_train.mean(), dnn_hidden_units=(128, 128), l2_reg=1e-5)
#rfnn_fix = keras_model_wrapper(model, epochs=5,batch_size=128)

model = R_Wide_and_Deep(max_user, max_item, embedding_size=8, fix_global_bias=None, dnn_hidden_units=(128, 128), l2_reg=1e-5)
rwd = keras_model_wrapper(model, epochs=5,batch_size=128)

#model = R_Wide_and_Deep(max_user, max_item, embedding_size=8, fix_global_bias=y_train.mean(), dnn_hidden_units=(128, 128), l2_reg=1e-5)
#rwd_fix = keras_model_wrapper(model, epochs=5,batch_size=128)

#model = R_Wide_and_Deep(max_user, max_item, embedding_size=4, fix_global_bias=y_train.mean(), dnn_hidden_units=(32, 32), l2_reg=1e-5)
#rwd_fix_simple = keras_model_wrapper(model, epochs=5,batch_size=128)

model = R_Wide_and_Deep(max_user, max_item, embedding_size=8, fix_global_bias=y_train.mean(), dnn_hidden_units=(128, 128), l2_reg=0.002)
rwd_fix_reg = keras_model_wrapper(model, epochs=5,batch_size=128)

#model = R_Wide_and_Deep(max_user, max_item, embedding_size=8, fix_global_bias=y_train.mean(), dnn_hidden_units=(128, 128), l2_reg=1e-5)
#rwd_fix_epoch = keras_model_wrapper(model, epochs=20,batch_size=128)



svd.fit(X_train, y_train)
#rfnn.fit(X_train, y_train)
#rfnn_fix.fit(X_train, y_train)
#rwd.fit(X_train, y_train)
rwd_fix.fit(X_train, y_train)
#rwd_fix_simple.fit(X_train, y_train)
rwd_fix_reg.fit(X_train, y_train)
#rwd_fix_epoch.fit(X_train, y_train)



top_X_test = X_test[y_test>=5]
for i in range(30):
    x_test = top_X_test[np.random.choice(range(top_X_test.shape[0]))]
    # create _X whose 0-indice is x_test and others is noise_items.
    noise_items = np.random.choice(np.unique(X[:,1]), replace=False, size=50)
    n_noise = len(noise_items)
    _X = np.array([x_test]*(n_noise+1))
    _X[1:, 1] = np.array(noise_items)       
    # get predict
    socre_dict = {}
    #socre_dict['rfnn'] = rfnn.predict(_X)
    #socre_dict['rwd'] = rwd.predict(_X)
    #socre_dict['rwd_fix_simple'] = rwd_fix_simple.predict(_X)
    socre_dict['svd'] = svd.predict(_X)
    socre_dict['rfnn_fix'] = rfnn_fix.predict(_X)
    socre_dict['rwd_fix'] = rwd_fix.predict(_X)
    socre_dict['rwd_fix_reg'] = rwd_fix_reg.predict(_X)
    socre_dict['rwd_fix_epoch'] = rwd_fix_epoch.predict(_X)

    _df = pd.DataFrame(socre_dict)

    #_df['svd-rfnn'] = socre_dict['svd'] - socre_dict['rfnn']
    #_df['svd-rwd'] = socre_dict['svd'] - socre_dict['rwd']
    #_df['svd-rwd_fix_simple'] = socre_dict['svd'] - socre_dict['rwd_fix_simple']
    _df['svd-rwd_fix'] = socre_dict['svd'] - socre_dict['rwd_fix']
    _df['svd-rwd_fix_reg'] = socre_dict['svd'] - socre_dict['rwd_fix_reg']

    #print(_df)
    #print(_df.describe())
    print(_df[['svd','rwd_fix','rwd_fix_reg','rwd_fix_epoch']].plot())


rfnn.model.variables
'''








    


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
