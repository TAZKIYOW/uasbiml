#library
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

#dataset 

X= np.array([128,256,256,128,512,256,256,128,128,128,256,256,128,128,128,256]).reshape((-1, 1))
Y= np.array([14999000,19799100,24999000,14999000,26999000,15999000,15999000,2499000,2499000,2799000,5999000,5999000,2799000,2499000,2999000,5399000])

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


#Import library pandas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


#Fitting model with trainig data
regressor.fit(X.values, Y)


import warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict(128))
