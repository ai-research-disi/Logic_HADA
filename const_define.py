# -*- coding: utf-8 -*-
"""
Created on Thu Mar 2 16:37:45 2023

@author: EleMisi
"""

import os

# Paths constants
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'CP2021_datasets')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')

# Data constants
ALG_PARAM = {'ANTICIPATE': 'nScenarios',
             'CONTINGENCY': 'nTraces'}
FEATURES = {'ANTICIPATE':
                {MODEL_DIR + '/no_input-memory_DecisionTree_MaxDepth10': 'nScenarios'}}
TARGET = {'ANTICIPATE':
              {MODEL_DIR + '/no_input-memory_DecisionTree_MaxDepth10': 'memAvg(MB)'}}
ML_TARGETS = ['memAvg(MB)', 'time(sec)', 'sol(keuro)']
INSTANCE_FEATURES = ['PV_mean', 'PV_std', 'Load_mean', 'Load_std']
