#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 08:54:30 2021

@author: joao
"""

import pandas as pd
import wfdb

data_source = pd.DataFrame(columns=['ECG', 'P', 'Q', 'R', 'S', 'T', 'RR', 'QT', 'QRS', 'HR', 'Class'])
columns = list(data_source)

traces = ['100','101','103','105','109','111','118', '106', '214','124']

N = 0; V = 0;

for file in traces:
    worksheet = pd.read_excel(file+'.xlsx')
    
    ecgtmp,fields = wfdb.rdsamp('sample-data/'+file,channels=[0]);
    ecg=[];
    for i in range(fields.get('sig_len')):
        ecg.append(ecgtmp[i][0])
        
    data = worksheet.values
    
    N = 0;
    for i in range(data.shape[0]):
        if data[i][5] != 'V':
            P = ecg[data[i][0]]
            Q = ecg[data[i][1]]
            R = ecg[data[i][2]]
            S = ecg[data[i][3]]
            T = ecg[data[i][4]]
            QRS = (data[i][3]-data[i][1])/360
            QT = (data[i][4]-data[i][1])/360
            if i+1 < data.shape[0] and (data[i+1][5] == 'N' or  data[i+1][5] == 'V' or data[i+1][5] == '/' or data[i+1][5] == 'L' or data[i+1][5] == 'R' or data[i+1][5] == 'A'):
                RR = (data[i+1][2]-data[i][2])/360
                HR = 60/RR
                Class = 'N'
                if data[i][5] != 'N':
                    Class = 'A'
                zipped =  zip(columns, [file, P, Q, R, S, T, RR, QT, QRS, HR, Class])
                data_source = data_source.append(dict(zipped), ignore_index=True)

v_traces = ['119', '208'] # Gerar traces PVC
for file in v_traces:
    ecgtmp,fields = wfdb.rdsamp('sample-data/'+file,channels=[0]);
    ecg=[];
    for i in range(fields.get('sig_len')):
        ecg.append(ecgtmp[i][0])
        
    ann = wfdb.rdann('PQ-Ann/'+ file,'atr');
    a = ann.symbol
    smp = ann.sample
    for i in range(len(a)):
        if a[i] == 'V' and i+3<len(a) and a[i+1] == 's':
            P = ecg[smp[i-2]]
            Q = ecg[smp[i-1]]
            R = ecg[smp[i]]
            S = ecg[smp[i+1]]
            T = ecg[smp[i+2]]
            QRS = (smp[i+1] - smp[i-1])/360
            QT = (smp[i+2] - smp[i-1])/360
            RR = (smp[i+3] - smp[i])/360
            HR = 60/RR
            Class = 'A'
            zipped =  zip(columns, [file, P, Q, R, S, T, RR, QT, QRS, HR, Class])
            data_source = data_source.append(dict(zipped), ignore_index=True)
            
            
tq_traces = ['cu01', 'cu02'] # Gerar traces VF e VT
for file in tq_traces:
    ecgtmp,fields = wfdb.rdsamp('tachyarrhythmia-database/'+file,channels=[0]);
    ecg=[];
    for i in range(fields.get('sig_len')):
        ecg.append(ecgtmp[i][0])
        
    ann = wfdb.rdann('tachyarrhythmia-database/'+ file,'atr');
    a = ann.symbol
    smp = ann.sample
    for i in range(len(a)):
        if a[i] == 'R' and i+3<len(a) and a[i+1] == 'S':
            P = -1
            Q = ecg[smp[i-1]]
            R = ecg[smp[i]]
            S = ecg[smp[i+1]]
            T = -1
            QRS = (smp[i+1] - smp[i-1])/250
            QT = -1
            RR = (smp[i+3] - smp[i])/250
            HR = 60/RR
            Class = 'A'
            zipped =  zip(columns, [file, P, Q, R, S, T, RR, QT, QRS, HR, Class])
            data_source = data_source.append(dict(zipped), ignore_index=True)
        

data_source.to_csv('dataset.csv', index=False)

        