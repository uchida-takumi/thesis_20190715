#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

class id_transformer:
    def __init__(self):
        """
        transform ids to the index which start from 0.
        """
        pass
    def fit(self, ids):
        """
        ARGUMETs:
            ids [array-like object]: 
                array of id of user or item.        
        """
        ids_ = sorted(list(set(ids)))
        self.id_convert_dict = {i:index for index,i in enumerate(ids_)}
    
    def transform(self, ids, unknown=None):
        """
        ARGUMETs:
            ids [array-like object]: 
                array of id of user or item.                
        """
        return [self.id_convert_dict.get(i, unknown) for i in ids]

    def fit_transform(self, ids):
        self.fit(ids)
        return self.transform(ids)
        
    def inverse_transform(self, indexes, unknown=None):
        """
        ARGUMETs:
            indexes [array-like object]: 
                array of index which are transformed                
        """
        return [get_key_from_val(self.id_convert_dict, ind) for ind in indexes]
    
    def fit_update(self, ids):
        """
        ARGUMETs:
            ids [array-like object]: 
                array of id of user or item.        
        """
        ids_ = sorted(list(set(ids)))
        ids_ = [id_ for id_ in ids_ if id_ not in self.id_convert_dict.keys()]
        now_max_id = max(self.id_convert_dict.values())
        new_id_convert_dict = {i:now_max_id+1+index for index,i in enumerate(ids_)}
        self.id_convert_dict.update(new_id_convert_dict)


def get_key_from_val(dict_, val, unknown=None):
    """
    dict_ = {'aa':123}
    val = 123
    get_key_from_val(dict_, val)
    > 'aa'    
    """
    list_vals = list(dict_.values())
    if val in list_vals:
        return list(dict_.keys())[list_vals.index(val)]    
    else:
        return unknown


def labeled_qcut_with_nan(x, q, reverse=False):
    '''
    nanを含むnp.arrayのxを受け取り、nanをカウントに含めた上でpandas.qcutを実行する。
    
    EXAMPLE
    ----------------
    x = np.array(list(range(7)) + [np.nan, np.nan, np.nan])
    q = [0.0, 0.33, 0.66, 1.0]
    print( labeled_qcut_with_nan(x, q) )
     > """
        ['001_(-0.001, 2.94]',
         '001_(-0.001, 2.94]',
         '001_(-0.001, 2.94]',
         '002_(2.94, 6.0]',
         '002_(2.94, 6.0]',
         '002_(2.94, 6.0]',
         '002_(2.94, 6.0]',
         '999_NotAvailable',
         '999_NotAvailable',
         '999_NotAvailable']
       """
    '''
    global labeled_qcut
    _x = np.array(x)
    nan_index = np.isnan(_x)
    _x[nan_index] = np.min(_x[~nan_index])
    qcuted = labeled_qcut(_x, q, reverse=reverse, duplicates='drop')
    qcuted[nan_index] = '999_NotAvailable'
    return qcuted

def labeled_qcut(*pos_args, reverse=False, **key_args):
    '''
    EXAMPLE
    ----------------
    x = range(10)
    q = [0.0, 0.33, 0.66, 1.0]
    print( labeled_qcut(x, q) )
     > """
        ['001_(-0.001, 2.97]',
         '001_(-0.001, 2.97]',
         '001_(-0.001, 2.97]',
         '002_(2.97, 5.94]',
         '002_(2.97, 5.94]',
         '002_(2.97, 5.94]',
         '003_(5.94, 9.0]',
         '003_(5.94, 9.0]',
         '003_(5.94, 9.0]',
         '003_(5.94, 9.0]']
       """
    '''
    qcuted = pd.qcut(*pos_args, **key_args)
    if hasattr(qcuted, 'categories'):
        categories = qcuted.categories
    else:
        categories = qcuted.dtype.categories
    
    if reverse:
        n_categories = len(categories)
        labels = {val:'{}_{}'.format('%03d'%(n_categories-i), val) for i,val in enumerate(categories)}        
    else:
        labels = {val:'{}_{}'.format('%03d'%(i+1), val) for i,val in enumerate(categories)}

    result = [labels.get(qcut) for qcut in qcuted]
    return np.array(result)

def labeled_cut(*pos_args, reverse=False, **key_args):
    '''
    EXAMPLE
    ----------------
    x = range(10)
    bins = [-0.1,3,8,10]
    print( labeled_cut(x, bins) )
     > """
        ['001_(-0.1, 3.0]',
         '001_(-0.1, 3.0]',
         '001_(-0.1, 3.0]',
         '001_(-0.1, 3.0]',
         '002_(3.0, 8.0]',
         '002_(3.0, 8.0]',
         '002_(3.0, 8.0]',
         '002_(3.0, 8.0]',
         '002_(3.0, 8.0]',
         '003_(8.0, 10.0]']
       """
    '''
    cuted = pd.cut(*pos_args, **key_args)
    
    if hasattr(cuted, 'categories'):
        categories = cuted.categories
    else:
        categories = cuted.dtype.categories
    
    if reverse:
        n_categories = len(categories)
        labels = {val:'{}_{}'.format('%03d'%(n_categories-i), val) for i,val in enumerate(categories)}        
    else:
        labels = {val:'{}_{}'.format('%03d'%(i+1), val) for i,val in enumerate(categories)}

    result = [labels.get(cut) for cut in cuted]
    return np.array(result)


