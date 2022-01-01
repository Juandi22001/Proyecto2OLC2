import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
df = pd.read_csv('pred.csv')

X = df['NO'].values # Feature
y = df['A'].values # Target

print(X.shape)
print(y.shape)

y = y.reshape(-1, 1)
X = X.reshape(-1, 1)
print(X)
print(y.shape)
print(X.shape)
reg.fit(X, y)
prediction_space = np.linspace(min(X), max(X)).reshape(-1, 1)
plt.scatter(X, y, color='black')
plt.plot(prediction_space, reg.predict(prediction_space))
plt.show()
