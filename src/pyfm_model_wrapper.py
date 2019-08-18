#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adjust pyfm Interface as same with others.
"""

from copy import deepcopy
import numpy as np

from sklearn.feature_extraction import DictVectorizer


class pyfm_model_wrapper:
    def __init__(self, model, X_columns={'user':0, 'item':1, 'timestamp':2}):
        self.model = model
        self.init_model = deepcopy(model)
        self.X_columns = X_columns

    def fit(self, X, y, try_cnt=20):
        cnt = 1
        while cnt <= try_cnt:
            try:
                _dict = self._transform_X_to_dicts(X)
                self.dictvectorizer = DictVectorizer()
                _X = self.dictvectorizer.fit_transform(_dict)
                self.model.fit(_X, y)
                return self
            except:
                print("Error Occure, retry {}".format(cnt))
                self.reset()
                cnt += 1
        raise "Over limit of retry at self.model.fit(_X, y)"


    def predict(self, X):
        _dict = self._transform_X_to_dicts(X)
        _X = self.dictvectorizer.transform(_dict)
        return self.model.predict(_X)
    
    def reset(self):
        self.model = deepcopy(self.init_model)
        return self
    
    def _transform_X_to_dicts(self, X):
        u, i = self.X_columns['user'], self.X_columns['item']
        dictorizer = lambda row: {'user':str(row[u]), 'item':str(row[i])}
        _dict = [dictorizer(row) for row in X]
        return _dict




if __name__ == 'how to use':
    # INPUTs
    import pandas as pd
    csv_fp = 'data/ml-latest-small/ratings.csv'
    data = pd.read_csv(csv_fp)
    column_names = ['userId', 'movieId', 'timestamp']
    label_name = 'rating'
    
    X, y = data[column_names].values, data[label_name].values
    
    pos = 80000
    X_train, X_test = X[:pos], X[pos:]
    y_train, y_test = y[:pos], y[pos:]

    # build model
    from pyfm import pylibfm
    model = pylibfm.FM(
            num_factors=8, num_iter=10,
            validation_size=0.0, task='regression',
            reg_0=0.0, reg_w=0.01, reg_v=0.05,
            )
    self = pyfm_model_wrapper(model)
        
    self.fit(X_train, y_train)
    y_pred = self.predict(X_test)
    
    np.mean(np.abs(y_pred - y_test))
    
    dir(self.model)

    self.model.w0
    self.model.w
    self.model.v
    
    self.model.reg_0    
    self.model.reg_w    
    self.model.reg_v    

    print(np.abs(self.model.v).mean())
    
    # new_prediction
    newX = np.array([
            ['new123', 'new123', 123],
            ['new123', 1, 123],
            [1, 'new123', 123],
            [1, 1, 123],
            ])
    self.predict(newX)
    
    # regulalization test
    model = pylibfm.FM(
            num_factors=8, num_iter=10,
            validation_size=0.0, task='regression',
            reg_0=0.0, reg_w=0.01, reg_v=0.99,
            )
    self = pyfm_model_wrapper(model)        
    self.fit(X_train, y_train)
    
    self.model.w0
    self.model.w
    self.model.v
    
    self.predict(X)
    
    print(np.abs(self.model.v).mean())