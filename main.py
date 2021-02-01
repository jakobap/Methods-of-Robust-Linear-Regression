import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


def dgp(s=400):
    """
    Data Generating Process a Gaussian RV transformed to sin
    We define an extreme set of outliers (every 3rd value = 10)
    """
    X = np.random.normal(size=s)
    y = np.sin(X)
    X = X[:, np.newaxis]

    # Test Data
    X_test = np.random.normal(size=int(s/2))
    y_test = np.sin(X_test)
    X_test = X_test[:, np.newaxis]

    plt.scatter(X,y)
    plt.title('Regular Data Generating Process')
    plt.show()

    # Adding Outliers
    y_errors = y.copy()
    y_errors[::3] = 10  # every 3rd value is an outlier

    X_train = X
    y_errors_train = y_errors

    plt.scatter(X_train,y_errors)
    plt.title('Data Generating Process With Outliers on Y')
    plt.show()

    # Adding Polynomial Features
    X_train_poly = PolynomialFeatures(degree=4).fit_transform(X)
    X_test_poly = PolynomialFeatures(degree=4).fit_transform(X_test)
    return X_train_poly, X_test_poly, y_errors_train, X_test, y_test, X


class RegressionFramework:
    def __init__(self, Xtr, ytr, Xte=None, yte=None):
        self.Xtr = Xtr  # Design Matrix Training Chunk
        self.Xte = Xte  # Design Matrix Testing Chunk
        self.ytr = ytr  # Labels Training Chunk
        self.yte = yte  # Labels Testing Chunk
        self.param_opt = None  # Optimized Parameters
        self.prediction = None  # Predicted labels
        self.mse = None  # MSE of Prediction
        self.mae = None  # MAE of Prediction

    def _score_prediction(self):
        """
        Produces Performance Measures based on a given prediction.
        """
        self.mse = mean_squared_error(self.yte, self.prediction)
        self.mae = mean_absolute_error(self.yte, self.prediction)
        print(f'{self.name} Scores:')
        print(f'Prediction MSE: {round(self.mse, 3)}')
        print(f'Prediction MAE: {round(self.mae, 3)}')

    def plot_reg(self, title, x_test, x_org, y_org, ylim=(-2, 11), xlim=(-3, 3)):
        """
        Plots the computed regression
        """
        plt.figure(figsize=(10, 5))
        plt.plot(x_org, y_org, '.', color='green', alpha=1)
        plt.plot(x_test, self.prediction, '.', color='red', alpha=1)
        plt.title(title)
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.show()

    def plot_loss(self, title):
        """
        Plots the chosen loss function
        """
        MSE = [self.loss_function(params=np.array([1]), X=np.array([i]), y=np.array([0])) for i in range(-5000, 5000)]
        plt.plot(list(range(-5000, 5000)), MSE)
        plt.title(title)
        plt.xlabel('Prediction')
        plt.ylabel('MSE')
        plt.show()

    def fit(self):
        """
        Minimization of loss function to find regression parameters
        """
        theta = np.array([5.0] * len(self.Xtr[0]))
        self.param_opt = minimize(self.loss_function, theta, args=self.args, method='SLSQP').x

    def predict(self):
        """
        Prediction based on computed parameters
        """
        self.prediction = self.Xte.dot(self.param_opt)
        self._score_prediction()


class LeastSquares(RegressionFramework):
    def __init__(self, Xtr, ytr, Xte, yte):
        super().__init__(Xtr, ytr, Xte, yte)
        self.loss_function = self._cost_least_squares  # Loss Function
        self.name = 'Least Squares'  # Model Name
        self.args = (self.Xtr, self.ytr)  # Loss Parameters for Optimizer

    def _cost_least_squares(self, params, X, y):
        return np.sum((y - X.dot(params)) ** 2) / float(np.size(y))


class LeastDeviation(RegressionFramework):
    def __init__(self, Xtr, ytr, Xte, yte):
        super().__init__(Xtr, ytr, Xte, yte)
        self.loss_function = self._cost_least_deviation  # Loss Function
        self.name = 'Least Deviation'  # Model Name
        self.args = (self.Xtr, self.ytr)  # Parameters for Loss Function

    def _cost_least_deviation(self, params, X, y):
        return np.sum(np.abs(y - X.dot(params)))


class HuberRegression(RegressionFramework):
    def __init__(self, Xtr, ytr, Xte, yte, delta):
        super().__init__(Xtr, ytr, Xte, yte)
        self.loss_function = self._cost_huber  # Loss Function
        self.name = 'Huber Loss'  # Model name
        self.args = (self.Xtr, self.ytr, delta)  # Loss Arguments for optimizer
        self.delta = delta  # Delta, unique to Huber Loss Function

    def _cost_huber(self, params, X, y, delta):
        pred = X.dot(params)
        loss = np.where(np.abs(y - pred) < delta,
                        0.5 * ((y - pred) ** 2),
                        delta * np.abs(y - pred) - 0.5 * (delta ** 2))
        return np.sum(loss)

    def plot_loss(self, title):
        """
        Plots the chosen loss function
        """
        for d in [0, .5, 1, 5, 10]:
            MSE = [self.loss_function(np.array([1]), np.array([i]), np.array([0]), d) for i in range(-5000, 5000)]
            plt.plot(list(range(-5000, 5000)), MSE, label=f'Delta: {d}')

        plt.title(title)
        plt.xlabel('Prediction')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

    def plot_reg(self, title, x_test, x_org, y_org, ylim=(-2, 11), xlim=(-3, 3), delta_benchmark=True):
        """
        Plotting the Huber Regression with the option to vizualize multiple delta values
        """
        plt.figure(figsize=(10, 5))
        if delta_benchmark:
            for d in [0, .01, .1, 1, 5, 10, 20]:
                self.delta = d
                self.args = (self.Xtr, self.ytr, d)
                self.fit()
                self.predict()
                plt.plot(x_test, self.prediction, '.', alpha=1, label=f'Delta: {d}')
        else:
            plt.plot(x_test, self.prediction, '.', color='red', alpha=1)

        plt.plot(x_org, y_org, '.', color='green', alpha=1)
        plt.title(title)
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.legend()
        plt.show()


# For Debugging
if __name__ == '__main__':
    np.random.seed(69)
    X_train_poly, X_test_poly, y_errors_train, X_test, y_test, X = dgp()

    least_squares = LeastSquares(Xtr=X_train_poly, ytr=y_errors_train, Xte=X_test_poly, yte=y_test)
    least_squares.plot_loss(title='Least Squares Cost Function y = 0')
    least_squares.fit()
    least_squares.predict()
    least_squares.plot_reg(title='Least Squares', x_test=X_test, x_org=X, y_org=y_errors_train)

    least_squares = LeastDeviation(Xtr=X_train_poly, ytr=y_errors_train, Xte=X_test_poly, yte=y_test)
    least_squares.plot_loss(title='Least Deviation Cost Function y = 0')
    least_squares.fit()
    least_squares.predict()
    least_squares.plot_reg(title='Least Deviation', x_test=X_test, x_org=X, y_org=y_errors_train)

    least_squares = HuberRegression(Xtr=X_train_poly, ytr=y_errors_train, Xte=X_test_poly, yte=y_test, delta=0.01)
    least_squares.plot_loss(title='Huber Loss Cost Function y = 0')
    least_squares.fit()
    least_squares.predict()
    least_squares.plot_reg(title='Huber Loss', x_test=X_test, x_org=X, y_org=y_errors_train)

    print('hello world')
