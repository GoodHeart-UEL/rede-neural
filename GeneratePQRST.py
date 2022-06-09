#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:23:00 2021

@author: sakuray
"""


#from scipy.signal import butter, lfilter, group_delay, filtfilt
import numpy as np
import wfdb
import xlsxwriter

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
        t_ann.insert(i,t[ann[i]])
        val_ann.insert(i, ecg[ann[i]])
    return t_ann,val_ann




def AjustaPontoR(ecg,N,r,size_r):
  annotation = r
  for i in range(size_r):
      ini = annotation[i] - 5
      fim = annotation[i] + 5
      if ini<1:
          ini = 1
      if fim>N:
          fim=N
      maior = ecg[ini]
      ind = ini
      #print ("i=",i,"R=",annotation[i]," ini=",ini," fim=",fim," maior=",maior," R:",annotation[i])
      for j in range(ini,fim):
          #print("ecg[",j,"]=",ecg[j])
          if ecg[j]>maior:
              ind=j
              maior=ecg[j]
      #print("==>selecionado:",ecg[ind])
      annotation[i]=ind
  return annotation

def FindQ(ecg,R):
  inc = -1
  i = R + inc
  while ecg[i]<ecg[i-inc]:
      i=i+inc
  return i

def FindMin(ecg,R,window):
  i = R
  idmenor = R
  if ecg[i]==ecg[i+1]:
      i+=1
  if i+window< len(ecg)-1:
      fim = i+window
  else:
      fim = len(ecg)-1
  for j in range(i,fim):
      if ecg[j]<ecg[idmenor]:
          idmenor = j

  return idmenor

def FindS(ecg,R):
  inc = 1
  i = R
  if ecg[i]==ecg[i+1]:
      i+=1
  while ecg[i]>ecg[i+inc] and i+inc<len(ecg)-1:
      i=i+inc
  return i



#traces = ['109']
#traces= ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200','201','202','203','205','207','208','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','233','234']
#traces = ['100','101','102','103','104','105','106','107','108','109','111','112','113','114','115','116','117','118','121','122','123','124','201','202','203','205','207','209','210','212','213','214','215','217','219','220','221','222','223','228','230','231','232','234']
#traces = ['109','111','214', '118', '124', '212'] # train
traces = ['100','101','103','105','109','111','118','124','106','214']
CodConhecidos = ['N','/','A', 'L', 'R']
for tr in traces:
    ecg,fields,ann= ReadECGTrace(tr)

    sig_len = fields.get('sig_len')
    fs = fields.get('fs');
    T = 1/fs
    t=np.linspace(0,(sig_len*T),sig_len);
    print (">>> Processando trace: ",tr," ann_len:",ann.ann_len);

    sample =  ann.sample[3:ann.ann_len-1]
    symbol =  ann.symbol[3:ann.ann_len-1]
    sample_len = len(sample)
    R=AjustaPontoR (ecg,sig_len,sample,sample_len)

    Q=[];
    S=[];
    for i in range(len(R)):
        Q.insert (i,FindQ(ecg,R[i])+1)
        S.insert(i, FindMin(ecg,R[i],45))
        
    new_ann = wfdb.rdann('PQ-Ann/'+ tr,'atr');
    
    new_symbols =  new_ann.symbol

    P=[]
    T=[]
    
    for i, txt in enumerate(new_symbols[0:-1]):
        if txt == 'p':
            P.insert(i, new_ann.sample[i])
        else:
            T.insert(i, new_ann.sample[i])

       
    workbook = xlsxwriter.Workbook(tr+'.xlsx')
    worksheet = workbook.add_worksheet('planilha')
    worksheet.write(0,0,'P')
    worksheet.write(0,1,'Q')
    worksheet.write(0,2,'R')
    worksheet.write(0,3,'S')
    worksheet.write(0,4,'T')
    worksheet.write(0,5,'Code')
    row=1
    for i, txt in enumerate(symbol[0:-1]):
        # print(i,txt,x_Q[i],x_R[i],x_S[i])
        if i >= len(P) or i >= len(Q) or i >= len(R) or i >= len(S) or i >= len(T):
            break
        if txt in CodConhecidos:
            worksheet.write(row,0,P[i])
            worksheet.write(row,1,Q[i])
            worksheet.write(row,2,R[i])
            worksheet.write(row,3,S[i])
            worksheet.write(row,4,T[i])
            worksheet.write(row,5,txt)
        else:  # pontos que nao consigo definir pontos Q e S
            worksheet.write(row,0,-1)
            worksheet.write(row,1,-1)
            worksheet.write(row,2,R[i])
            worksheet.write(row,3,-1)
            worksheet.write(row,4,-1)
            worksheet.write(row,5,txt)
        row+=1
                
    workbook.close()
