#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 13:27:02 2021

@author: joao
"""

import pandas as pd
import seaborn as sns


data = pd.read_csv('dataset.csv')

print(data.describe())

sns.pairplot(data.drop("ECG", axis=1), hue="Class", diag_kind="kde")