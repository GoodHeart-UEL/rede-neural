#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:23:00 2021

@author: sakuray
"""


#from scipy.signal import butter, lfilter, group_delay, filtfilt
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import pandas as pd

def ReadECGTrace(f):
    ecgtmp,fields = wfdb.rdsamp('sample-data/'+f,channels=[0]);
    ann = wfdb.rdann('sample-data/'+ f,'atr');
    sig_len = fields.get('sig_len')#20000;#
    ecg=[];
    for i in range(sig_len):
        ecg.append(ecgtmp[i][0])
    
    return (ecg,fields,ann)


def annotation(ann,ann_len,ecg, t):
    t_ann=[]
    val_ann=[]
    
    for i in range(ann_len):
        if ann[i]>=0:
            t_ann.insert(i,t[ann[i]])
            val_ann.insert(i, ecg[ann[i]])
    return t_ann,val_ann




# def AjustaPontoR(ecg,N,r,size_r):
#   annotation = r
#   for i in range(size_r):
#       ini = annotation[i] - 5
#       fim = annotation[i] + 5
#       if ini<1:
#           ini = 1
#       if fim>N:
#           fim=N
#       maior = ecg[ini]
#       ind = ini
#       #print ("i=",i,"R=",annotation[i]," ini=",ini," fim=",fim," maior=",maior," R:",annotation[i])
#       for j in range(ini,fim):
#           #print("ecg[",j,"]=",ecg[j])
#           if ecg[j]>maior:
#               ind=j
#               maior=ecg[j]
#       #print("==>selecionado:",ecg[ind])
#       annotation[i]=ind
#   return annotation

# def FindQ(ecg,R):
#   inc = -1
#   i = R + inc
#   while ecg[i]<ecg[i-inc]:
#       i=i+inc
#   return i

# def FindMin(ecg,R,window):
#   i = R
#   idmenor = R
#   if ecg[i]==ecg[i+1]:
#       i+=1
#   if i+window< len(ecg)-1:
#       fim = i+window
#   else:
#       fim = len(ecg)-1
#   for j in range(i,fim):
#       if ecg[j]<ecg[idmenor]:
#           idmenor = j

#   return idmenor


traces= ['124']
#traces= ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
#traces = ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','121','122','123','124','201','202','203','205','207','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','234']
CodConhecidos = ['N','/','A', 'L', 'R']
for tr in traces:
    ecg,fields,ann= ReadECGTrace(tr)

    sig_len = fields.get('sig_len')
    fs = fields.get('fs');
    T = 1/fs
    t=np.linspace(0,(sig_len*T),sig_len);
    print (">>> Processando trace: ",tr," ann_len:",ann.ann_len);
    
    # workbook object is created
    df = pd.read_excel(tr+'.xlsx')
 
    P = df['P']
    Q = df['Q']
    R = df['R']
    S = df['S']
    T = df['T']
    Code = df['Code']
    
    # Plot dos pontos
    plt.figure(1);
    plt.clf()
    plt.plot(ecg[0:32000])
    for i, txt in enumerate(P):
         #plt.plot(txt,ecg[txt],'*')
        if i>100:
            break
        plt.annotate('p'+str(i), # this is the text
             (txt, ecg[txt]), # these are the coordinates to position the label
              textcoords="offset points", # how to position the text
              xytext=(0,10), # distance from text to points (x,y)
              ha='center') # horizontal alignment can be left, right or center

    
    
    for i, txt in enumerate(Q):
         #plt.plot(txt,ecg[txt],'*')
        if i>100:
            break
        plt.annotate('q'+str(i), # this is the text
             (txt, ecg[txt]), # these are the coordinates to position the label
              textcoords="offset points", # how to position the text
              xytext=(0,10), # distance from text to points (x,y)
              ha='center') # horizontal alignment can be left, right or center
        
    for i, txt in enumerate(R):
         #plt.plot(txt,ecg[txt],'*')
        if i>100:
            break
        plt.annotate('r'+str(i), # this is the text
             (txt, ecg[txt]), # these are the coordinates to position the label
              textcoords="offset points", # how to position the text
              xytext=(0,10), # distance from text to points (x,y)
              ha='center') # horizontal alignment can be left, right or center
    
    for i, txt in enumerate(S):
         #plt.plot(txt,ecg[txt],'*')
        if i>100:
            break
        plt.annotate('s'+str(i), # this is the text
             (txt, ecg[txt]), # these are the coordinates to position the label
              textcoords="offset points", # how to position the text
              xytext=(0,10), # distance from text to points (x,y)
              ha='center') # horizontal alignment can be left, right or center
    
    for i, txt in enumerate(T):
         #plt.plot(txt,ecg[txt],'*')
        if i>100:
            break
        plt.annotate('t'+str(i), # this is the text
             (txt, ecg[txt]), # these are the coordinates to position the label
              textcoords="offset points", # how to position the text
              xytext=(0,10), # distance from text to points (x,y)
              ha='center') # horizontal alignment can be left, right or center




    

    # plot em relação ao tempo
    plt.figure(2);
    plt.clf()
    plt.plot(t,ecg, label='Noisy signal')

    # coordenadas em relação ao tempo
    x_P,y_P=annotation(P, len(P), ecg, t)
    x_Q,y_Q=annotation(Q, len(Q), ecg, t)
    x_S,y_S=annotation(S, len(S), ecg, t)
    x_R,y_R=annotation(R, len(R), ecg, t)
    x_T,y_T=annotation(T, len(T), ecg, t)
    plt.plot(x_P,y_P,'b*')
    plt.plot(x_R,y_R,'o')
    plt.plot(x_Q,y_Q,'g*')
    plt.plot(x_S,y_S,'r*')
    plt.plot(x_T,y_T,'y*')
    for i, txt in enumerate(Code):
        
        plt.annotate(txt, # this is the text
                     (t[R[i]],ecg[R[i]]), # these are the coordinates to position the label
                      textcoords="offset points", # how to position the text
                      xytext=(0,10), # distance from text to points (x,y)
                      ha='center') # horizontal alignment can be left, right or center
        



