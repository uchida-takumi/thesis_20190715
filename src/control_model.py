#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 10:14:23 2019

@author: takumi_uchida
"""
from collections import Counter
import numpy as np

class random_model:
    def __init__(self, *pos_args, X_columns={'user':0, 'item':1, 'timestamp':2}, **key_args):
        self.X_columns = X_columns
                
    def fit(self, X, y):
        self.max_rating, self.min_rating = y.max(), y.min()
        return self
                        
    def predict(self, X):
        n_sample = X.shape[0]
        return self.min_rating + np.random.rand(n_sample) * (self.max_rating - self.min_rating)

    def reset(self):
        pass

class popular_model:
    def __init__(self, *pos_args, X_columns={'user':0, 'item':1, 'timestamp':2}, **key_args):
        self.X_columns = X_columns
    
    def fit(self, X, y):
        items = X[:, self.X_columns['item']]
        cnt = Counter(items)
        inverse_ranking = sorted(cnt, key=lambda x:cnt[x])
        n_item = len(inverse_ranking)

        max_, min_ = y.max(), y.min()
        self.middle_score = (max_ + min_) / 2
        self.ranking_score = {item : min_+(max_-min_)*((i+1)/n_item) for i,item in enumerate(inverse_ranking)}
        
    def predict(self, X):
        items = X[:, self.X_columns['item']]
        return np.array([self.ranking_score.get(i, self.middle_score) for i in items])
    
    def reset(self):
        pass
        

if __name__ == 'how to use it':
    import numpy as np
    X = np.array([
            [1,2],
            [1,3],
            [2,1],
            [2,2],
            ])     
    y = np.array([
            5,
            4,
            1,
            3,
            ])    
    
    random = random_model()
    random.fit(X, y)
    random.predict(X)
    
    pop = popular_model()
    pop.fit(X, y)
    pop.predict(X)


            
