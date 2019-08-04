#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 10:14:23 2019

@author: takumi_uchida
"""
import numpy as np

class random_model:
    def __init__(self, *pos_args, **key_args):
        pass
                
    def fit(self, X, y):
        self.max_rating, self.min_rating = y.max(), y.min()
        return self
                        
    def predict(self, X):
        n_sample = X.shape[0]
        return self.min_rating + np.random.rand(n_sample) * (self.max_rating - self.min_rating)

    def reset(self):
        pass

class pop_model:
    def __init__(self)
        pass
