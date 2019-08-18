#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adjust fastFM Interface as same with others.
"""

from copy import deepcopy
from scipy.sparse import csc_matrix
import numpy as np


class fastFM_model_wrapper:
    def __init__(self, model, X_columns={'user':0, 'item':1, 'timestamp':2},
                 y_norm=False):
        self.model = model
        self.init_model = deepcopy(model)
        self.X_columns = X_columns
        self.y_norm = y_norm

    def fit(self, X, y,):
        if self.y_norm:
            self.y_train_mean = y.mean()
            _y = y - self.y_train_mean
        else:
            _y = y
        self._fit_X_sparse_encoder(X)
        sparse_X = self._convert_X_to_sparse(X)
        self.model.fit(sparse_X, _y)
        return self

    def predict(self, X):
        sparse_X = self._convert_X_to_sparse(X)
        predicted = self.model.predict(sparse_X)
        if self.y_norm:
            predicted += self.y_train_mean
        return predicted
    
    def reset(self):
        self.model = deepcopy(self.init_model)
        return self
    
    def _convert_X_to_sparse(self, X):
        users = X[:, self.X_columns['user']]
        items = X[:, self.X_columns['item']]
        user_indices = [self.user_id_encoder.get(user, None) for user in users]
        item_indices = [self.item_id_encoder.get(item, None) for item in items]
        
        samples, feature_indices = [],[]
        for i,(user,item) in enumerate(zip(user_indices, item_indices)):
            if user is not None:
                samples.append(i)
                feature_indices.append(user)
            if item is not None:
                samples.append(i)
                feature_indices.append(item)
        
        sparse_X = csc_matrix(([1]*len(samples), (samples,feature_indices)))

        # resize sparse_X
        n_sample = X.shape[0]
        sparse_X.resize((n_sample, self.x_feature_dim))
        
        return sparse_X
    
    
    def _fit_X_sparse_encoder(self, X):
        u_users = np.unique(X[:, self.X_columns['user']])
        u_items = np.unique(X[:, self.X_columns['item']])
        self.user_id_encoder = {user:i for i,user in enumerate(u_users)}
        max_user_indice = max(self.user_id_encoder.values())
        self.item_id_encoder = {item:max_user_indice+i+1 for i,item in enumerate(u_items)}
                
        self.x_feature_dim = max(self.item_id_encoder.values()) + 1       




if __name__ == 'how to use':
    # INPUTs
    import pandas as pd
    csv_fp = 'data/ml-latest-small/ratings.csv'
    data = pd.read_csv(csv_fp)
    column_names = ['userId', 'movieId', 'timestamp']
    label_name = 'rating'
    
    X, y = data[column_names].values, data[label_name].values
    
    pos = 5000
    X_train, X_test = X[:pos], X[pos:]
    y_train, y_test = y[:pos], y[pos:]

    # build model
    from fastFM import sgd, als
    model = sgd.FMRegression(n_iter=1000000, init_stdev=0.1, rank=0, l2_reg_w=0.01, l2_reg_V=0.05)
    self = fastFM_model_wrapper(model)
        
    self.fit(X_train, y_train)
    y_pred = self.predict(X_test)
    
    np.mean(np.abs(y_pred - y_test))
    
    self.model.w0_
    self.model.w_[:5]
    self.model.V_
    self.model.V_[:,0]
    self.model.V_[:,5]
    self.model.V_[:,100]
    self.user_id_encoder
    self.item_id_encoder
    

    # 
    newX = np.array([
            [9999999,9999999,123],
            [1,9999999,123],
            [9999999,1,123],
            ])    
    self.predict(newX)


    # ---------     
    from surprise import SVD
    from src.surprise_algo_wrapper import surprise_algo_wrapper

    algo = SVD(n_factors=0)
    algo_w = surprise_algo_wrapper(algo)

    algo_w.fit(X_train, y_train)
    y_pred_svd = algo_w.predict(X_test)
    np.mean(np.abs(algo_w.predict(X_test) - y_test))
    
    dir(algo_w.algo)
    algo_w.predict(np.array([[99999999,999999999,123]]))
    algo_w.algo.bi
    algo_w.algo.bu
    algo_w.algo.pu
    algo_w.algo.qi

    # ---------
    for a,b in zip(y_pred, y_pred_svd):
        print(a,b,a-b)


    
    # ---------
    X = np.array([
            [3,1,123],
            [3,2,123],
            [3,4,123],
            [5,1,123],
            [5,2,123],
            ])
    y = np.array([1,2,3,4,5])

    from fastFM import als
    model = als.FMRegression(n_iter=1000, rank=0)
    self = fastFM_model_wrapper(model)
    self.fit(X, y)
    cX = self._convert_X_to_sparse(X)
    cX.toarray()
    new_X = np.array([
            [9,9,123],
            [9,0,123],
            [3,9,123],
            ])
    cX = self._convert_X_to_sparse(new_X)
    cX.toarray()
    self.predict(new_X)
    self.model.w0_
    self.model.w_
    self.model.V_
    
    # ------------
    X = np.array([
            [3,1,123],
            [3,2,123],
            [3,4,123],
            [5,1,123],
            [5,2,123],
            [5,4,123],
            [6,1,123],
            [6,2,123],
            [6,4,123],
            [7,1,123],
            [7,2,123],
            [7,4,123],
            [8,1,123],
            [8,2,123],
            [8,4,123],
            [9,1,123],
            [9,2,123],
            [9,4,123],
            ])
    y = 1*X[:,0] + 1*X[:,1]

    from fastFM import als,sgd
    model = als.FMRegression(n_iter=1, rank=0, l2_reg=0.001)
    fm = fastFM_model_wrapper(model, y_norm=True)
    fm.fit(X, y)

    from surprise import SVD
    from src.surprise_algo_wrapper import surprise_algo_wrapper
    model = SVD(n_epochs=1000, n_factors=0, reg_all=0.001)
    svd = surprise_algo_wrapper(model)
    svd.fit(X, y)
    

    fm.predict(X)
    svd.predict(X)
    
    fm.model.w0_
    fm.model.w_
    fm.model.V_
    
    svd.algo.trainset.global_mean
    svd.algo.bu
    svd.algo.bi
    svd.algo.pu
    svd.algo.qi
    