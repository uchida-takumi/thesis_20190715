#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Lambda
from tensorflow.keras import regularizers, initializers, optimizers, losses
from tensorflow.keras.models import Model

almost0init = initializers.RandomUniform(minval=-1e-5, maxval=1e-5)

def FNN(max_user, max_item, fix_global_bias=None,
        embedding_size=8, dnn_hidden_units=(128, 128), l2_reg=1e-5):  
    '''
    ARGUMENT
    -------------
    max_user [int]:
        max id number of user.
    max_item [int]:
        max id number of item.
    max_new_user [int]:
        max id number of addtional user when transfer learning.
    max_new_item [int]:
        max id number of addtional item when transfer learning.        
    fix_global_bias [int or None]:
        fix global_bias as inputted int. if None, trained global_bias as weight.
    embedding_size [int]:
        latent factor number.
    dnn_hidden_units [array]:
        hidden layer size. (ex.) dnn_hidden_units=(62, 128, 62)
    l2_reg [float]:
        L2 reguralization.        
    '''
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
    x = multiple_hidden(x, dnn_hidden_units, l2_reg, prefix_name='hidden')

    # --- OUTPUT --- #
    output = output_dense(x, fix_global_bias, l2_reg, name='output')
    
    # --- build as MODEL --- #
    model = Model(inputs=[input_user, input_item], outputs=[output])
    
    ## compile 
    model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.mean_squared_error
            )    
    return model
    

def Wide_and_Deep(max_user, max_item, 
                  add_max_user=None, add_max_item=None, 
                  fix_global_bias=None, embedding_size=8, 
                  dnn_hidden_units=(128, 128), l2_reg=1e-5):
    '''
    ARGUMENT
    -------------
    max_user [int]:
        max id number of user.
    max_item [int]:
        max id number of item.
    fix_global_bias [int or None]:
        fix global_bias as inputted int. if None, trained global_bias as weight.
    embedding_size [int]:
        latent factor number.
    dnn_hidden_units [array]:
        hidden layer size. (ex.) dnn_hidden_units=(62, 128, 62)
    l2_reg [float]:
        L2 reguralization.        
    '''
    
    # --- INPUT --- #
    input_user = Input(shape=(1,), name='user')
    input_item = Input(shape=(1,), name='item')
    
    # --- WIDE part --- #
    ## binary embedding
    if add_max_user is None:
        user_binary_embedding = id_binary_embedding(
                input_user, max_user, l2_reg, sufix_name='wide_user')
    else:
        user_binary_embedding = sep_id_binary_embedding(
                input_user, max_user, add_max_user, l2_reg, sufix_name='wide_user')
    if add_max_item is None:
        item_binary_embedding = id_binary_embedding(
                input_item, max_item, l2_reg, sufix_name='wide_item')    
    else:
        item_binary_embedding = sep_id_binary_embedding(
                input_item, max_item, add_max_item, l2_reg, sufix_name='wide_item')
        
    wide_x = Concatenate(name='wide_user_item_concatenate'
                         )([user_binary_embedding, item_binary_embedding])

    
    # --- DEEP part --- #
    ## embbedding
    if add_max_user is None:
        user_embedding = id_embedding(
                input_user, max_user, embedding_size, l2_reg, 
                sufix_name='deep_user')
    else:
        user_embedding = sep_id_embedding(
                input_user, max_user, add_max_user, embedding_size, l2_reg, 
                sufix_name='deep_user')    
        
    if add_max_item is None:
        item_embedding = id_embedding(
                input_item, max_item, embedding_size, l2_reg, 
                sufix_name='deep_item')    
    else:
        item_embedding = sep_id_embedding(
                input_item, max_item, add_max_item, embedding_size, l2_reg, 
                sufix_name='deep_item')    
        
    deep_x = Concatenate(name='deep_user_item_concatenate'
                    )([user_embedding, item_embedding])    
    deep_x = multiple_hidden(
            deep_x, dnn_hidden_units, l2_reg, prefix_name='deep_hidden')
    
    # --- OUTPUT --- #
    x = Concatenate(name='wide_deep_concatenate'
                    )([wide_x, deep_x])
    output = output_dense(x, fix_global_bias, l2_reg, name='output')

    # --- build as MODEL --- #
    model = Model(inputs=[input_user, input_item], outputs=[output])
    
    ## compile 
    model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.mean_squared_error
            )    
    
    return model
    
def fine_tunning_compile_Wide_and_Deep(model):
    train_stop_layers = [
            'embedding_pre_id_deep_user',
            'embedding_pre_id_deep_item',
            'deep_hidden_',
            ]    
    # change not-trainable to finetuning
    for layer in model.layers:
        if layer.trainable and len(layer.variables):
            for pattern in train_stop_layers:
                if re.match(pattern, layer.name):
                    layer.trainable = False
    # need to re-compile to make availble trainable change 
    model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.mean_squared_error
            )    
    return model


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

def sep_id_embedding(input_, pre_max_, add_max_, embedding_size, l2_reg, sufix_name):
    '''
    import numpy as np   
    import tensorflow as tf
    pre_max_, add_max_ = 3, 2
    input_ = np.random.choice(range(pre_max_+add_max_), size=20)
    input_ = tf.convert_to_tensor(input_)
    sufix_name = 'sep_id_embedding'
    '''
    dtype = input_.dtype
    pre_masking = tf.cast(input_<pre_max_,  dtype=dtype, name='pre_masking_'+sufix_name)
    add_masking = tf.cast(input_>=pre_max_, dtype=dtype, name='add_masking_'+sufix_name)
    pre_input_ = pre_masking * (input_ + 1)
    add_input_ = add_masking * (input_ - pre_max_ + 1)
    pre_id_embedding = id_embedding(
            pre_input_, pre_max_+1, embedding_size, l2_reg, 
            sufix_name='pre_id_'+sufix_name)
    add_id_embedding = id_embedding(
            add_input_, add_max_+1, embedding_size, l2_reg, 
            sufix_name='add_id_'+sufix_name)    
    pre_add_id_embedding = tf.add(
            pre_id_embedding * tf.reshape(tf.cast(pre_masking, dtype=tf.float32), shape=(-1,1)),
            add_id_embedding * tf.reshape(tf.cast(add_masking, dtype=tf.float32), shape=(-1,1)),
            name='pre_add_id_'+sufix_name
            )
    return pre_add_id_embedding

def id_binary_embedding(input_, max_, l2_reg, sufix_name):
    return id_embedding(input_, max_, 1, l2_reg, sufix_name)

def sep_id_binary_embedding(input_, pre_max_, add_max_, l2_reg, sufix_name):
    return sep_id_embedding(input_, pre_max_, add_max_, 1, l2_reg, sufix_name)

    
def multiple_hidden(x, dnn_hidden_units, l2_reg, prefix_name):
    if isinstance(dnn_hidden_units, int):
        dnn_hidden_units = [dnn_hidden_units]
    for i,d in enumerate(dnn_hidden_units):
        x = Dense(
                d, activation='relu', use_bias=True, 
                bias_initializer='zeros',
                kernel_regularizer=regularizers.l2(l2_reg),
                name=prefix_name+'_{}'.format(i)
                )(x)
    return x

def output_dense(x, fix_global_bias, l2_reg, name):
    if fix_global_bias is None:
        use_bias=True
        global_bias = 0
    else:
        use_bias=False
        global_bias = fix_global_bias

    x = Dense( 1, activation=None, use_bias=use_bias,
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name=name+'0'
                   )(x)    
    x = Lambda(lambda o:o + global_bias, name=name)(x) 
    return x    


    
#####################
if __name__ == 'how to use':    
    # load dataset
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
    
    # --- FNN --- 
    model = FNN(max_user, max_item)    
    model.fit(
            x={'user':X[:,0], 'item':X[:,1]}, 
            y={'output':y},
            batch_size=1, epochs=20)
    
    # predict on in-samples
    model.predict(x={'user':X[:,0], 'item':X[:,1]})
    
    # predcit on new ids
    model.predict(x={'user':np.array([0,0,1]), 'item':np.array([0,1,0])})

    # --- FNN with fix_global_bias ---
    model = FNN(max_user, max_item, fix_global_bias=10)    
    model.fit(
            x={'user':X[:,0], 'item':X[:,1]}, 
            y={'output':y},
            batch_size=1, epochs=20)
    model.predict(x={'user':X[:,0], 'item':X[:,1]})
    
        
    # --- Wide_and_Deep --- 
    model = Wide_and_Deep(max_user, max_item)    
    model.fit(
            x={'user':X[:,0], 'item':X[:,1]}, 
            y={'output':y},
            batch_size=1, epochs=20)
    
    # predict on in-samples
    model.predict(x={'user':X[:,0], 'item':X[:,1]})
    
    # predcit on new ids
    model.predict(x={'user':np.array([0,0,1]), 'item':np.array([0,1,0])})
    
    # --- Wide_and_Deep with fix_global_bias ---
    model = Wide_and_Deep(max_user, max_item, fix_global_bias=10)    
    model.fit(
            x={'user':X[:,0], 'item':X[:,1]}, 
            y={'output':y},
            batch_size=1, epochs=20)
    model.predict(x={'user':X[:,0], 'item':X[:,1]})
    
    
    # --- fine-tuning on Wide_and_Deep ---
    model = Wide_and_Deep(max_user, max_item, add_max_user=1, add_max_item=2)    
    model.fit(
            x={'user':X[:,0], 'item':X[:,1]}, 
            y={'output':y},
            batch_size=1, epochs=20)
    
    # save weights as 'before' 
    from copy import deepcopy
    before_after = {'before':{}, 'after':{}}
    for layer in model.layers:
        if layer.name in ['embedding_pre_id_deep_item', 'deep_hidden_0', 'embedding_add_id_deep_item']:
            before_after['before'][layer.name] = deepcopy(layer.weights)
    # stop some layer train.
    model = fine_tunning_compile_Wide_and_Deep(model)

    # set new id data to fit
    add_X = np.array([
                [1,4], #new-item
                [1,5], #new-item
                [3,1], #new-user
                [3,5], #new-user, new-item
            ])     
    add_y = np.array([
                2,
                2,
                5,
                4,
            ])
    model.fit(
            x={'user':add_X[:,0], 'item':add_X[:,1]}, 
            y={'output':add_y},
            batch_size=1, epochs=20)
    
    # save weight asd 'after'
    for layer in model.layers:
        if layer.name in ['embedding_pre_id_deep_item', 'deep_hidden_0', 'embedding_add_id_deep_item']:
            before_after['after'][layer.name] = deepcopy(layer.weights)
            
    # compair(assert)
    layer_name =  'embedding_pre_id_deep_item'
    before_ = before_after['before'][layer_name]
    after_  = before_after['after'][layer_name]
    assert (before_[0].numpy() == after_[0].numpy()).all()
    
    layer_name =  'deep_hidden_0'
    before_ = before_after['before'][layer_name]
    after_  = before_after['after'][layer_name]
    assert (before_[0].numpy() == after_[0].numpy()).all()
        
    layer_name =  'embedding_add_id_deep_item'
    before_ = before_after['before'][layer_name]
    after_  = before_after['after'][layer_name]
    assert (before_[0].numpy() == after_[0].numpy()).all()
    