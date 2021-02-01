import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)


#Train data
X = np.random.normal(size=400)
y = np.sin(X)
X = X[:, np.newaxis]

# Test Data
X_test = np.random.normal(size=200)
y_test = np.sin(X_test)
X_test = X_test[:, np.newaxis]

# plt.scatter(X,y)
# plt.title('Regular Data Generating Process')
# plt.show()


#Adding Outliers in Y direction
y_errors = y.copy()
y_errors[::3] = 10 # every 3rd value is an outlier

X_errors = X.copy()
X_errors[::3] = 10

X_train = X
y_errors_train = y_errors

# plt.scatter(X_train,y_errors)
# plt.title('Data Generating Process With Outliers on Y')
# plt.show()
#
# plt.scatter(X_errors,y)
# plt.title('Data Generating Process With Outliers on X')
# plt.show()

# Adding Polynomial Features
X_train_t = PolynomialFeatures(degree=4).fit_transform(X)
X_test_t = PolynomialFeatures(degree=4).fit_transform(X_test)
X_errors_t = PolynomialFeatures(degree=4).fit_transform(X_errors)


class RegressionFramework():
    def __init__(self, loss_function, name, Xtr, ytr, Xte=None, yte=None):
        self.loss_function = loss_function  # loss function to optimize
        self.name = name
        self.Xtr = Xtr  # Design Matrix Training Chunk
        self.Xte = Xte  # Design Matrix Testing Chunk
        self.ytr = ytr  # Labels Training Chunk
        self.yte = yte  # Labels Testing Chunk
        self.param_opt = None
        self.prediction = None
        self.mse = None
        self.mae = None

    def _score_prediction(self):
        """
        Produces Performance Measures based on a given prediction.
        """
        self.mse = mean_squared_error(self.yte, self.prediction)
        self.mae = mean_absolute_error(self.yte, self.prediction)
        print(f'{self.name} Scores:')
        print(f'Prediction MSE: {round(self.mse, 3)}')
        print(f'Prediction MAE: {round(self.mae, 3)}')
    #
    # def _plot_reg(self, title, x_pred, y_pred, x_org, y_org, ylim=(-2, 11), xlim=(-3, 3)):
    #     """
    #     Plots the underlyign regression.
    #     """
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(x_org, y_org, '.', color='green', alpha=1)
    #     plt.plot(x_pred, y_pred, '.', color='red', alpha=1)
    #     plt.title(title)
    #     plt.ylim(ylim)
    #     plt.xlim(xlim)
    #     plt.show()

    def plot_loss(self):
        MSE = [self.loss_function(np.array([1]), np.array([i]), np.array([0])) for i in range(-5000, 5000)]
        plt.plot(list(range(-5000, 5000)), MSE)
        plt.title('Least Squares Cost Function y = 0')
        plt.xlabel('Prediction')
        plt.ylabel('MSE')
        plt.show()

    def fit(self):
        """
        Minimization of loss function to find regression parameters
        """
        theta = np.array([5.0] * len(X_train_t[0]))
        self.param_opt = minimize(cost_least_squares, theta, args=(X_train_t, y_errors_train), method='SLSQP').x

    def predict(self):
        """
        Prediction
        """
        self.prediction = self.Xte.dot(self.param_opt)
        self._score_prediction()


"""
Individually defining the loss functions.
"""


def cost_least_squares(params, X, y):
    return np.sum((y - X.dot(params)) ** 2) / float(np.size(y))


def cost_least_deviation(params, X, y):
    return np.sum(np.abs(y - X.dot(params)))


def cost_huber(params, X, y, delta):
    pred = X.dot(params)
    loss = np.where(np.abs(y - pred) < delta,
                    0.5 * ((y - pred) ** 2),
                    delta * np.abs(y - pred) - 0.5 * (delta ** 2))
    return np.sum(loss)


# For Debugging
if __name__ == '__main__':
    least_squares = RegressionFramework(cost_least_squares, name='Least Squares' ,Xtr=X_train_t,
                                        ytr=y_errors_train, Xte=X_test_t, yte=y_test)
    # least_squares.plot_loss()
    least_squares.fit()
    least_squares.predict()

    least_squares = RegressionFramework(cost_least_deviation, name='Least Deviation', Xtr=X_train_t,
                                        ytr=y_errors_train, Xte=X_test_t, yte=y_test)
    # least_squares.plot_loss()
    least_squares.fit()
    least_squares.predict()

    least_squares = RegressionFramework(cost_huber, name='Huber Loss', Xtr=X_train_t,
                                        ytr=y_errors_train, Xte=X_test_t, yte=y_test)
    # least_squares.plot_loss()
    least_squares.fit()
    least_squares.predict()
    print('hello world')
