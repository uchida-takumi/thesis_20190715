#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
グラフの作図などを定義
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def keras_model_wrapper_fit_history_chart(
        fitted_keras_model_wrapper, 
        title='model loss',
        save_png_path=None):
    fit_history = fitted_keras_model_wrapper.get_fit_history()    
    fig, ax = plt.subplots(ncols=1, figsize=(5,2))
    for label, scores in fit_history.items():
        if label=='loss':
            ax.plot(fit_history[label], label='train loss')
        if label=='val_loss':
            ax.plot(fit_history[label], label='test loss')
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend(loc='upper right')
    # save as png
    if save_png_path is not None:
        fig.savefig(save_png_path)
    
    return fig, ax
    


if __name__ == 'how to use':

    from src.keras_model_wrapper import keras_model_wrapper
    from src.DNN_recommender import FNN

    # data load
    import pandas as pd
    csv_fp = 'data/ml-latest-small/ratings.csv'
    data = pd.read_csv(csv_fp)
    column_names = ['userId', 'movieId', 'timestamp']
    label_name = 'rating'
    
    X, y = data[column_names].values, data[label_name].values
    
    max_user, max_item = X[:,0].max()+1, X[:,1].max()+1
    fitted_keras_model_wrapper = keras_model_wrapper(FNN, dict(max_user=max_user, max_item=max_item))

    # --- get fit history ---
    X_train, y_train = X[:10000], y[:10000]
    X_test, y_test = X[10000:], y[10000:]
    fitted_keras_model_wrapper.fit(X_train, y_train, X_test, y_test)

    keras_model_wrapper_fit_history_chart(
            fitted_keras_model_wrapper,
            save_png_path='/tmp/aaa.png'
            )


