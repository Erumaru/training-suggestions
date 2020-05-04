from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from numpy import loadtxt, reshape, split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

# define dataset
x = loadtxt('formatted_data_x.txt')
x = x.reshape(-1, 1)
y = loadtxt('formatted_data_y.txt')
train_x, test_x = split(x, 2)
train_y, test_y = split(y, 2)
# define the model
model = LinearRegression()
# fit the model
model.fit(train_x, train_y)
pred_y = model.predict(test_x)
# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(test_y, pred_y))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(test_y, pred_y))

scores = cross_val_score(model, x, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Plot outputs
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter3D(test_x[:,0], test_x[:,1], test_y, c=test_y, cmap='Greens')
plt.scatter(test_x, test_y)
plt.plot(test_x, pred_y, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()