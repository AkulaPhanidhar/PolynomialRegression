import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

Dataset = pd.read_csv('SampleData.csv')
X = Dataset.iloc[:, 1:2].values
Y = Dataset.iloc[:, 2].values

Poly_Reg = PolynomialFeatures(degree = 4)
X_Poly = Poly_Reg.fit_transform(X)
Poly_Reg.fit(X_Poly, Y)
lin_reg = LinearRegression()
lin_reg.fit(X_Poly, Y)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg.predict(Poly_Reg.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
