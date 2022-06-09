# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 14:33:30 2021

@author: joao_
"""

import wfdb

record = wfdb.rdrecord('sample-data/119', sampto=60000)
ann = wfdb.rdann('sample-data/119', 'atr', sampto=60000)

wfdb.plot_wfdb(record=record, annotation=ann, plot_sym=True,
                   time_units='seconds', title='MIT-BIH Record 100',
                   figsize=(10,4))