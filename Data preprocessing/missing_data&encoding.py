
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


#________________________________________________________________________________________
# Taking care of missing data where 'nan' is replaced by mean of the data for that column
#----------------------------------------------------------------------------------------

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') # alternatively : 'median','most_frequent','constant'
imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])



#________________________________________________________________________________________
# Encoding categorical data
#----------------------------------------------------------------------------------------


# Encoding the Independent Variable
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

#encoding neumerial labels for categories.
#you can directly apply onehotencoder with out label encoding
onehotencoder = OneHotEncoder(categories ="auto")
X = onehotencoder.fit_transform(X).toarray()



# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#to revert the onehot encoding
X_ = onehotencoder.inverse_transform(X)




#this file is using sklean for preprocessing ,however pandas also provides preprocessing but with more abstract way,
#so sklearn is easy and straight forward.

