

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer,normalize

# Generating dataset
X = np.random.uniform(low=5.5, high=113.3, size=(100,2))
y = np.random.uniform(low=0, high=1, size=(100,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling

#Standardization 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

#MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range =(0,1))
X_train_minmax = min_max_scaler.fit_transform(X_train)
Y_min_max_scaler = MinMaxScaler(feature_range =(0,1))
y_train_minmax = min_max_scaler.fit_transform(y_train)

#QuantileTransformer
quantile_transformer =QuantileTransformer(random_state=0)
X_train_quantile_transformer = quantile_transformer.fit_transform(X_train)
y_quantile_transformer =QuantileTransformer(random_state=0)
y_train_quantile_transformer = quantile_transformer.fit_transform(y_train)

#Normalization
X_normalized = normalize(X, norm='l1')
#X_normalized = normalize(X, norm='l2')
#X_normalized = normalize(y, norm='l1')

