from numpy.lib import polynomial
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np

import random

X=[1,2,3,4,5,6,7,8,9,10]
Y=[1,5,20,60,90,120,180,220,340,410]


X=np.asarray(X)

Y=np.asarray(Y)

X=X[:,np.newaxis]
Y=Y[:,np.newaxis]



nb_degree=2

polynomial_features=PolynomialFeatures(degree=nb_degree)

X_TRANSF=polynomial_features.fit_transform(X)

model= LinearRegression()
model.fit(X_TRANSF,Y)

Y_NEW = model.predict(X_TRANSF)
rmse=np.sqrt(mean_squared_error(Y,Y_NEW))

r2=r2_score(Y,Y_NEW)

print('RMSE',rmse)
print('R2',r2)

x_new_main=0.0
x__new_max=50.0


X_NEW=np.linspace(x_new_main,x__new_max,50)

X_NEW=X_NEW[:,np.newaxis]
X_NEW_TRANSF =polynomial_features.fit_transform(X_NEW)

Y_NEW=model.predict(X_NEW_TRANSF)

plt.plot(X_NEW,Y_NEW,color='coral',linewidth=3)
plt.grid()
plt.xlim(x_new_main,x__new_max)
plt.ylim(0,1000)

title='Degre={}; RMSE ={}; R2 ={}'.format(nb_degree,round(rmse,2),round(r2,2))
plt.title("POLYNOMIAL LINEAR REGRESSION USING"+title)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
