#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 09:36:36 2021

@author: joao
"""

import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score


arquivo = pd.read_csv('dataset.csv')
arquivo = shuffle(arquivo)


#selected_dataset = arquivo[['ECG', 'P', 'Q', 'R', 'S', 'T', 'RR', 'QT', 'QRS', 'Class']]

#selected_dataset = arquivo[['ECG', 'P', 'R', 'S', 'T', 'QRS', 'Class']] # Teste-F Anova

selected_dataset = arquivo[['ECG', 'Q', 'R', 'S', 'T', 'QRS', 'Class']] # RelieF

previsores = selected_dataset.iloc[:, :6].values
classe = selected_dataset.iloc[:, 6].values

classe = classe != 'N'


def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 4, activation = 'relu', input_dim = 5))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    classificador.compile(optimizer = 'Adam', loss = 'binary_crossentropy',
                          metrics = ['binary_accuracy'])
    return classificador


# Cross validation
# classificador = KerasClassifier(build_fn = criarRede,
#                                   epochs = 190,
#                                   batch_size = 10)
# resultados = cross_val_score(estimator = classificador,
#                               X = previsores, y = classe,
#                               cv = 7, scoring = 'accuracy')
# media = resultados.mean()
# desvio = resultados.std()

#resultados = {
#    '1': [],
#    '2': [],
#    '3': [],
#    '4': [],
#    '5': [],
#    '6': [],
#    '7': [],
#    '8': [],
#    '9': [],
#}

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy

skf = StratifiedKFold(n_splits=7, random_state=None)
# X is the feature set and y is the target

reg = []
val = []
prev = []

result_array = []


for train_index, test_index in skf.split(previsores, classe): 
    #print("Train:", train_index)
    #print("Validation:", test_index) 
    X_train, X_test = numpy.array(previsores[train_index][:, 1:], dtype=numpy.float64), previsores[test_index]
    y_train, y_test = classe[train_index], classe[test_index]
    
    classificador = criarRede()
    history = classificador.fit(X_train, y_train, batch_size = 10, epochs = 190)

    
    results = classificador.predict(numpy.array(X_test[:, 1:], dtype=numpy.float64))
    
    previsoes = (results > 0.5)
    
    result_array.append(accuracy_score(y_test, previsoes))
    
    
    #for i in X_test:
    #    reg.append(i[0])
        
    #for i in y_test:
    #    val.append(i)
        
    #for i in results:
    #    prev.append(i[0])
    
    # n = int(1)
    # while(n<10):
        
    #     previsoes = (results > (n/10))
    
    #     acuracia = accuracy_score(y_test, previsoes)
    
    #     resultados[str(n)].append(acuracia)
    #     n+=1

#print(resultados)


# classificador = criarRede()
# history = classificador.fit(previsores, classe,
#                  batch_size = 10, epochs = 150)


# test_dataset = pd.read_csv('test-dataset.csv')
# selected_test_dataset = test_dataset[['P', 'Q', 'R', 'S', 'T', 'QRS', 'Class']]

# previsores_teste = selected_test_dataset.iloc[:, :6].values
# classe_teste = selected_test_dataset.iloc[:, 6].values
# classe_teste = classe_teste != 'N'

# previsoes = classificador.predict(previsores_teste)
# previsoes2 = (previsoes > 0.5)


# from sklearn.metrics import confusion_matrix, accuracy_score
# precisao = accuracy_score(classe_teste, previsoes2)
# matriz = confusion_matrix(classe_teste, previsoes2)
# f1 = f1_score(classe_teste, previsoes2)

#test = pd.DataFrame(columns=['ECG', 'P', 'Q', 'R', 'S', 'T', 'RR', 'QT', 'QRS', 'HR', 'Class'])
#
#for index, row in arquivo.iterrows():
#    if row['ECG'] == 106:
#        test = test.append(row)
#        arquivo = arquivo.drop(labels=index, axis=0)
#    elif row['ECG'] == 118:
#        test = test.append(row)
#        arquivo = arquivo.drop(labels=index, axis=0)


#from sklearn.model_selection import train_test_split
#previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15)

#previsores_treinamento = previsores
#classe_treinamento = classe
#previsores_teste = test.iloc[:, 1:9].values
#classe_teste = test.iloc[:, 10].values
#classe_teste = classe_teste != 'N'


#callback = tf.keras.callbacks.EarlyStopping(monitor='binary_accuracy', mode='max', patience=60, min_delta=0.0001, restore_best_weights=True)
#classificador = criarRede()
#history = classificador.fit(previsores_treinamento, classe_treinamento,
#                  batch_size = 10, epochs = 200)
#
#previsoes = classificador.predict(previsores_teste)
#previsoes2 = (previsoes > 0.5)
#from sklearn.metrics import confusion_matrix, accuracy_score
#precisao = accuracy_score(classe_teste, previsoes2)
#matriz = confusion_matrix(classe_teste, previsoes2)
#f1 = f1_score(classe_teste, previsoes2)

#Exportar pesos e Bias
#file = open("neural_network_c/h_weights.txt", "w+")
#hidden_weights = classificador.layers[0].get_weights()[0]
#for column in hidden_weights.T:
#    for elem in column:
#        file.write(str(elem)+"\n")
#file.close()
#
#file = open("neural_network_c/h_bias.txt", "w+")
#hidden_biases = classificador.layers[0].get_weights()[1]
#for elem in hidden_biases:
#    file.write(str(elem)+"\n")
#file.close()
#
#file = open("neural_network_c/o_weights.txt", "w+")
#output_weights = classificador.layers[1].get_weights()[0]
#for column in output_weights.T:
#    for elem in column:
#        file.write(str(elem)+"\n")
#file.close()
#
#file = open("neural_network_c/o_bias.txt", "w+")
#output_biases = classificador.layers[1].get_weights()[1]
#for elem in output_biases:
#    file.write(str(elem)+"\n")
#file.close()


#resultado = classificador.evaluate(previsores_teste, classe_teste)

# print("Accuracy: " + str(accuracy(matriz)))
# print("Sensitivity: " + str(sensitivity(matriz)))
# print("Specificity: " + str(specificity(matriz)))



