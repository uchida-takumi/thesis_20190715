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


def cv(model, X, y, k_hold=5, need_hit=True, hit_threshold=5, 
       detail=False, seed=None):
    """
    Cross-Validation on a recommender system model.
    """

    # set up
    np.random.seed(seed)
    
    # random split indice set.
    train_test_indices = [random_split(X, test_rate=0.3) for _ in range(k_hold)]
    
    # k_hold
    if need_hit:
        item_indice = model.X_columns['item']
        noise_items = np.random.choice(np.unique(X[:,item_indice]), replace=False, size=999)

    result_dict = defaultdict(list)
    for k, (train, test) in enumerate(train_test_indices):
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
            _X_test = X_test[hit_indice]
            result_df['rank'] = np.nan
            result_df['rank'][hit_indice] = np.array(
                    [get_rank(model, x_test, noise_items) for x_test in _X_test]
                    )
            for top_n in [5,10,20,30,40,50,100]:
                column = 'hit_top_{}'.format(top_n)
                result_df[column] = np.nan
                result_df.loc[hit_indice, column] = (result_df.loc[hit_indice, 'rank']<=top_n).astype(int)
                result_dict[column+'_precision'].append(result_df[column].mean())
                result_dict[column+'_recall'].append(result_df[column].mean() / top_n)
            result_dict

        # set result_dict summeries of result_df.
        result_dict['MAE'].append(np.mean(result_df['abs_error'])) 
        result_dict['metrics_by_labeled_user'].append(label_groupby(result_df, by=['labeled_user']))
        result_dict['metrics_by_labeled_item'].append(label_groupby(result_df, by=['labeled_item']))
        result_dict['metrics_by_labeled_user_item'].append(label_groupby(result_df, by=['labeled_user','labeled_item']))
                    
        if detail:
            result_dict['detail'].append(result_df)
        
        return result_dict
                
        
def random_split(X, test_rate=0.3):
    """
    return random splitted indices(train_indices, test_indices)
    """
    n_sample = X.shape[0]
    train_size = n_sample - int(n_sample * test_rate)    
    all_indices = list(range(n_sample))
    np.random.shuffle(all_indices)    
    return np.array(all_indices[:train_size]), np.array(all_indices[train_size:])

def get_rank(model, x_test, noise_items):
    # create _X whose 0-indice is x_test and others is noise_items.
    n_noise = len(noise_items)
    _X = np.array([x_test]*(n_noise+1))
    _X[1:, model.X_columns['item']] = np.array(noise_items)       
    # get predict
    scores = model.predict(_X)
    rank = util.get_rank(scores, target_indice=0)        
    return rank   
        
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
    agg_dict = {col:[np.mean, np.median, np.std] for col in metrics_cols}
    agg_dict.update({'n_sample':sum})
    return _df.groupby(by=by).agg(agg_dict)
     
    