#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 16:39:04 2019

@author: takumi_uchida
"""
import re
import numpy as np
import pandas as pd
from src import util
from collections import defaultdict, Counter

import multiprocessing as mp


def cv(model, X, y, k_hold=5, 
       need_hit=True, hit_threshold=5, hit_max_sample=1000,
       detail=False, seed=123):
    """
    Cross-Validation on a recommender system model.
    """

    # set up
    np.random.seed(seed)
    
    # random split indice set.
    train_test_indices = [random_split(X) for _ in range(k_hold)]
    
    # k_hold
    if need_hit:
        item_indice = model.X_columns['item']
        size = min(999, np.unique(X[:,item_indice]).size)
        noise_items = np.random.choice(np.unique(X[:,item_indice]), replace=False, size=size)

    result_dict = defaultdict(list)
    for k, (train, test) in enumerate(train_test_indices):
        print('k_hold = {}/{}'.format(k+1, k_hold))
        # model reset
        model.reset()
        
        # split X, y to train and test 
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test] 
        
        # fit and predict
        model.fit(X_train, y_train)
        predict = model.predict(X_test)
        
        # set result to result_df        
        result_df = pd.DataFrame(X_test, columns=sorted(model.X_columns, key=lambda x:model.X_columns[x]))        
        labeled_user, labeled_item = label_id(X_train, X_test, X_columns=model.X_columns)
        result_df['labeled_user'], result_df['labeled_item'] = labeled_user, labeled_item
        result_df['y_test'] = y_test
        result_df['predict'] = predict
        result_df['k_hold'] = k        
        
        result_df['error'] = predict - y_test
        result_df['abs_error'] = np.abs(predict - y_test)
        
                
        if need_hit:
            
            hit_indice = (y_test>=hit_threshold)
            result_df['rank'] = np.nan            
            result_df['rank'][hit_indice][:hit_max_sample] = np.array(
                    [get_rank(model, x_test, noise_items) for x_test in X_test[hit_indice][:hit_max_sample]]
                    )
            '''
            # run as multiprocess
            pool = mp.Pool(mp.cpu_count())            
            mp_args = [(model, x_test, noise_items) for x_test in X_test[hit_indice]]
            mp_result = pool.map(get_rank_for_multiprocessing, mp_args)
            result_df['rank'][hit_indice] = np.array(mp_result)
            '''
            for top_n in [5,10,20,30,40,50,100]:
                column = 'hit_top_{}'.format(top_n)
                result_df[column] = np.nan
                result_df.loc[hit_indice, column] = (result_df.loc[hit_indice, 'rank']<=top_n).astype(int)
                result_dict[column+'_precision'].append(result_df[column].mean())
                result_dict[column+'_recall'].append(result_df[column].mean() / top_n)
            
        # set result_dict summeries of result_df.
        result_dict['MAE'].append(np.mean(result_df['abs_error'])) 
        result_dict['metrics_by_labeled_user'].append(label_groupby(result_df, by=['labeled_user']))
        result_dict['metrics_by_labeled_item'].append(label_groupby(result_df, by=['labeled_item']))
        result_dict['metrics_by_labeled_user_item'].append(label_groupby(result_df, by=['labeled_user','labeled_item']))
                    
        if detail:
            result_dict['detail'].append(result_df)
    
    return_dict = dict(
            each_k_hold=result_dict,
            total_mean=total_mean_result_dict(result_dict),
            )

    return return_dict
                
        
def random_split(X, test_rate=0.3):
    """
    return random splitted indices(train, validation, test)
    """
    n_sample = X.shape[0]
    test_size = int(n_sample * test_rate)
    train_size = n_sample - test_size
    all_indices = list(range(n_sample))
    np.random.shuffle(all_indices)   
    all_indices = np.array(all_indices)
    return all_indices[:train_size], all_indices[train_size:]

def get_rank(model, x_test, noise_items):
    # create _X whose 0-indice is x_test and others is noise_items.
    n_noise = len(noise_items)
    _X = np.array([x_test]*(n_noise+1))
    _X[1:, model.X_columns['item']] = np.array(noise_items)       
    # get predict
    scores = model.predict(_X)
    rank = util.get_rank(scores, target_indice=0)        
    return rank   

def get_rank_for_multiprocessing(tuple_arg):
    model, x_test, noise_items = tuple_arg
    return get_rank(model, x_test, noise_items)

        
def label_id(X_train, X_test, X_columns={'user':0, 'item':1},
             bins=[0,10,20,30,40,50,np.inf], new_name='000_(0.0, 0.0)'):
    train_users, train_items = X_train[:, X_columns['user']], X_train[:, X_columns['item']]
    test_users, test_items = X_test[:, X_columns['user']], X_test[:, X_columns['item']]
    
    def label_test(train_, test_, ):
        cnt_train_ = pd.DataFrame(Counter(train_).items(), columns=['id', 'count'])
        cnt_train_['label'] = util.labeled_cut(cnt_train_['count'], bins=bins)
        _dict = {i:row['label'] for i,row in cnt_train_.iterrows()}
        _label_test = np.array([_dict.get(id, new_name) for id in test_])
        return _label_test

    return label_test(train_users, test_users), label_test(train_items, test_items)
        
def label_groupby(result_df, 
                  by=['labeled_user', 'labeled_item']):    
    _df = result_df.copy(); _df['n_sample'] = 1
    metrics_cols = [col for col in _df.columns if re.match(r'(abs_error|hit_top_)', col)]
    agg_dict = {col:np.mean for col in metrics_cols}
    agg_dict.update({'n_sample':sum})
    return _df.groupby(by=by).agg(agg_dict)

def total_mean_result_dict(result_dict):
    keys = [key for key in result_dict if re.match(r'(MAE|hit_top_)', key)]
    _dict0 = {key:np.mean(result_dict[key]) for key in keys}
    
    keys = [key for key in result_dict if re.match(r'metrics_by_', key)]
    def _groupby_mean(key):
        _df = pd.concat(result_dict[key])
        return _df.groupby(by=_df.index).mean()         
    _dict1 = {key:_groupby_mean(key) for key in keys}
    
    _dict0.update(_dict1)
    return _dict0
    
    


if __name__ == 'how to use':
    # load data
    import pandas as pd
    csv_fp = 'data/ml-latest-small/ratings.csv'
    data = pd.read_csv(csv_fp)
    column_names = ['userId', 'movieId', 'timestamp']
    label_name = 'rating'    
    X, y = data[column_names].values, data[label_name].values
    #X, y = X[:10000], y[:10000] 

    # --- example 1 ---
    # build recommender model and wrap to adjust I/O     
    from src.surprise_algo_wrapper import surprise_algo_wrapper    
    from surprise import SVD
    svd = surprise_algo_wrapper(SVD())
    
    # run cv
    cv_result = cv(svd, X, y, k_hold=3)
    
    # MAEs
    print(cv_result['each_k_hold']['MAE'])    
    print(cv_result.keys())
    
    # --- example 2 ---
    from src.keras_model_wrapper import keras_model_wrapper 
    from src.DNN_recommender import R_Wide_and_Deep # Re-FNN
    max_user, max_item = X[:,0].max()+1, X[:,1].max()+1
    model_init_dict = dict(max_user=max_user, max_item=max_item, embedding_size=8, dnn_hidden_units=(64), l2_reg=0.010)
    
    model = keras_model_wrapper(R_Wide_and_Deep, model_init_dict)
    
    # run cv
    k_hold = 3
    cv_result = cv(model, X, y, k_hold=k_hold, need_hit=True, seed=999)

    # MAEs
    print(cv_result['each_k_hold']['MAE'])    
    print(cv_result['each_k_hold']['metrics_by_labeled_item'][0])  
    cv_result['each_k_hold']['metrics_by_labeled_item'][0].iloc[:, -4:]

    
    
    
