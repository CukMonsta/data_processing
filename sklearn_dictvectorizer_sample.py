# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Categorical Features
from sklearn.feature_extraction import DictVectorizer

dict_encoder = DictVectorizer()

# Load Data
cities = [
        {'city': 'Madrid'},
        {'city': 'New York'},
        {'city': 'Londres'},
        {'city': 'New York'},
        {'city': 'Londres'}
        ]

# Si tenemos varios diccionarios con las mismas keys podemos categorizarlos con esta función
transform_data = dict_encoder.fit_transform(cities)

print(transform_data)
print(transform_data.toarray())
