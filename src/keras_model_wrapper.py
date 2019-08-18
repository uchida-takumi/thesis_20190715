#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class keras_model_wrapper:
    def __init__(self, model_module, model_init_dict={}, 
                 epochs=5, batch_size=128, 
                 X_columns={'user':0, 'item':1, 'timestamp':2}):        
        self.model = model_module(**model_init_dict)
        self.model_module = model_module
        self.model_init_dict = model_init_dict
        self.epochs = epochs
        self.batch_size = batch_size
        self.X_columns = X_columns
                
        
    def fit(self, X, y):
        X_dict = self._get_X_dict(X)
        y_dict = self._get_y_dict(y)
        self.model.fit(X_dict, y_dict, epochs=self.epochs, batch_size=self.batch_size)
        return self
                        
    def predict(self, X):
        X_dict = self._get_X_dict(X)
        predict = self.model.predict(X_dict)
        return predict.squeeze()

    def reset(self):
        self.model = self.model_module(**self.model_init_dict)
        
    def _get_X_dict(self, X):
        X_dict = {}
        for input_name in self.model.input_names:
            X_dict[input_name] = X[:, self.X_columns[input_name]]
        return X_dict

    def _get_y_dict(self, y):
        y_dict = {}
        for output_name in self.model.output_names:
            y_dict[output_name] = y # y must be 1-d array in this I/O.
        return y_dict



if __name__ == 'how to use':
    from src.DNN_recommender import RFNN
    # INPUTs
    import pandas as pd
    csv_fp = 'data/ml-latest-small/ratings.csv'
    data = pd.read_csv(csv_fp)
    column_names = ['userId', 'movieId', 'timestamp']
    label_name = 'rating'
    
    X, y = data[column_names].values, data[label_name].values
    
    max_user, max_item = X[:,0].max()+1, X[:,1].max()+1
    model = keras_model_wrapper(RFNN, dict(max_user=max_user, max_item=max_item))

    model.fit(X, y)
    model.predict(X)
    model.reset()
    