#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 17:39:37 2019

@author: takumi_uchida
"""

import time
from datetime import datetime
from copy import deepcopy
import numpy as np

class timestamp_encoder:
    def __init__(self, 
                 min_year=1990, max_year=2020, 
                 n_padding={'month':1,'day':2,'hour':2}):
        '''
        min_year, max_year = 1990, 2020
        n_padding={'month':1,'day':2,'hour':2}
        
        ts_encoder = timestamp_encoder(min_year, max_year, n_padding)
        datetime_array = [
                datetime(2015,3,12,12,59,59),
                datetime(2018,12,1,3,59,59),
                datetime(2019,5,29,23,59,59),
                ]
        # run 
        ts_encoder.transform_normalize(datetime_array)
        ts_encoder.transform_binary(datetime_array)
        ts_encoder.transform_binary_padding(datetime_array)
        '''
        self.min_year = min_year
        self.max_year = max_year
        self.n_padding = n_padding
        self.binary_dict_padding = _get_binary_dict(n_padding)
        self.binary_dict = _get_binary_dict({'month':0,'day':0,'hour':0})
                
    def reset(self, **kargs):
        self.__init__(**kargs)
        
    def transform_normalize(self, datetime_array):
        kargs = dict(min_year=self.min_year, max_year=self.max_year)
        return [_normalize(dt, **kargs) for dt in datetime_array]

    def transform_binary(self, datetime_array):
        kargs = dict(min_year=self.min_year, max_year=self.max_year)
        return [_binararize(dt, self.binary_dict, **kargs) for dt in datetime_array]
        
    def transform_binary_padding(self, datetime_array):
        kargs = dict(min_year=self.min_year, max_year=self.max_year)
        return [_binararize(dt, self.binary_dict_padding, **kargs) for dt in datetime_array]


def _get_binary_dict(n_padding={'month':0,'day':0,'hour':0}):
    """
    EXAMPLE
    --------------
    n_padding={'month':2,'day':3,'hour':4}
    result = _get_binary_dict(n_padding)

    print(result.keys())
     > dict_keys(['month', 'day', 'hour'])
    print(result['month'])
     > { 1: [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
         2: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
         3: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
         4: [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         5: [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         6: [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         7: [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
         8: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
         9: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
         10: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
         11: [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         12: [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]}
    """
    BINARY_DICT = {
        'month' : {i:[1 if j==i else 0 for j in range(1,13)] for i in range(1,13)},
        'day'   : {i:[1 if j==i else 0 for j in range(1,32)] for i in range(1,32)},
        'hour'  : {i:[1 if j==i else 0 for j in range(0,24)] for i in range(0,24)},
         }
    if n_padding=={'month':0,'day':0,'hour':0}:
        return BINARY_DICT
    else:
        _BINARY_DICT = deepcopy(BINARY_DICT)
        for key,vecs in BINARY_DICT.items():
            for k,vec in vecs.items():
                ind = vec.index(1)
                for i in range(ind-n_padding[key], ind+n_padding[key]+1):
                    i %= len(_BINARY_DICT[key][k])
                    _BINARY_DICT[key][k][i] = 1
    return _BINARY_DICT            

def from_unixtime_to_datetime(unixtime):
    '''
    unixtime = time.time()
    '''
    return datetime.utcfromtimestamp(unixtime)


def _normalize(dt, min_year=2010, max_year=2020):
    '''
    dt = datetime(2019,4,2,11,23,50)
    min_year=2010
    max_year=2020
    '''
    _year  = (dt.year - min_year) / (max_year - min_year)
    _month = (dt.month - 1) / (12 - 1)
    _day  = (dt.day - 1) / (31 - 1)
    _hour = (dt.hour - 1) / (24 - 1)
    return [_year, _month, _day, _hour]


def _binararize(dt, binary_dict, min_year=2010, max_year=2020):
    '''
    dt = datetime(2019,4,2,11,23,50)
    min_year=2010
    max_year=2020
    binary_dict = _get_binary_dict(n_padding={'month':0,'day':0,'hour':0})
    '''    
    _year  = (dt.year - min_year) / (max_year - min_year)
    _month = binary_dict['month'][dt.month]
    _day   = binary_dict['day'][dt.month]
    _hour  = binary_dict['hour'][dt.month]
    return [_year] + _month + _day + _hour


if __name__ == 'how to use this':
    min_year, max_year = 1990, 2020
    n_padding={'month':1,'day':2,'hour':2}
    
    ts_encoder = timestamp_encoder(min_year, max_year, n_padding)
    datetime_array = [
            datetime(2015,3,12,12,59,59),
            datetime(2018,12,1,3,59,59),
            datetime(2019,5,29,23,59,59),
            ]
    # run 
    ts_encoder.transform_normalize(datetime_array)
    ts_encoder.transform_binary(datetime_array)
    ts_encoder.transform_binary_padding(datetime_array)
    
