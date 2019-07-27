#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras import regularizers, initializers, optimizers, losses
from tensorflow.keras.models import Model

almost0init = initializers.RandomUniform(minval=-1e-5, maxval=1e-5)

def RFNN(max_user, max_item, 
         embedding_size=8, dnn_hidden_units=(128, 128), l2_reg=1e-5):    
    # --- INPUT --- #
    input_user = Input(shape=(1,), name='user')
    input_item = Input(shape=(1,), name='item')

    # --- Embedding --- #
    user_embedding = id_embedding(
            input_user, max_user, embedding_size, l2_reg, sufix_name='user')
    item_embedding = id_embedding(
            input_item, max_item, embedding_size, l2_reg, sufix_name='item')    
    x = Concatenate(name='user_item_concatenate'
                    )([user_embedding, item_embedding])
    
    # --- HIDDEN --- #
    for i,d in enumerate(dnn_hidden_units):
        x = Dense(
                d, activation='relu', use_bias=True, 
                kernel_regularizer=regularizers.l2(l2_reg),
                name='hidden_{}'.format(i)
                )(x)

    # --- OUTPUT --- #
    output = Dense(
                    1, activation=None, use_bias=True,
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name='output'
                   )(x)
    
    # --- build as model --- #
    model = Model(inputs=[input_user, input_item], outputs=[output])
    
    ## compile 
    model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.mean_absolute_error
            )
    
    return model
    

    '''
    user_embedding = Embedding(
            input_dim=max_user,
            output_dim=embedding_size,
            embeddings_initializer=almost0init,
            embeddings_regularizer=regularizers.l2(l2_reg),
            name='embedding_user'            
            )(user)
    user_embedding = Flatten(name='flatten_user')(user_embedding)
    '''
    





#####################
# sub module
def id_embedding(input_, max_, embedding_size, l2_reg, sufix_name):
    id_embedding = Embedding(
            input_dim=max_,
            output_dim=embedding_size,
            embeddings_initializer=almost0init,
            embeddings_regularizer=regularizers.l2(l2_reg),
            name='embedding_{}'.format(sufix_name)            
            )(input_)
    id_embedding = Flatten(name='flatten_{}'.format(sufix_name))(id_embedding)
    return id_embedding

    
#####################
if __name__ == 'how to use':
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
    
    max_user, max_item = X[:,0].max()+1, X[:,1].max()+1
    
    model = RFNN(max_user, max_item)    
    model.fit(
            x={'user':X[:,0], 'item':X[:,1]}, 
            y={'output':y},
            batch_size=1, epochs=20)
    
    # predict on in-samples
    model.predict(x={'user':X[:,0], 'item':X[:,1]})
    
    # predcit on new ids
    model.predict(x={'user':np.array([0,0,1]), 'item':np.array([0,1,0])})
        
    
    
    
    
    
        
    