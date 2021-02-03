import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, linprog

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


def dgp(s=400):
    """
    Data Generating Process a Student-t RV transformed to sin
    We define an extreme set of outliers (every 3rd value = 10)
    """
    # X = np.random.normal(size=s)
    X = np.random.standard_t(df=4, size=s)
    y = np.sin(X)
    X = X[:, np.newaxis]

    # Test Data
    # X_test = np.random.normal(size=int(s/2))
    X_test = np.random.standard_t(df=4, size=int(s/2))
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
    plt.title('Data Generating Process With Extremes and Outliers')
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
        self.mse = round(mean_squared_error(self.yte, self.prediction), 2)
        self.mae = round(mean_absolute_error(self.yte, self.prediction), 2)
        print(f'{self.name} Scores:')
        print(f'Prediction MSE: {self.mse}')
        print(f'Prediction MAE: {self.mae,}')

    def plot_reg(self, title, x_test, x_org, y_org, ylim=(-2, 11), xlim=(-3, 3)):
        """
        Plots the computed regression
        """
        plt.figure(figsize=(10, 5))
        plt.plot(x_org, y_org, '.', color='green', alpha=1)
        plt.plot(x_test, self.prediction, '+', color='red', alpha=1, label=f'MSE: {self.mse}')
        plt.title(title)
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.legend()
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
            HUBER = [self.loss_function(np.array([1]), np.array([i]), np.array([0]), d) for i in np.linspace(-10, 10, 200)]
            plt.plot(np.linspace(-10, 10, 200), HUBER, label=f'Delta: {d}')
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
                plt.plot(x_test, self.prediction, '+', alpha=1, label=f'Delta: {d} MSE: {self.mse}')
        else:
            plt.plot(x_test, self.prediction, '+', color='red', alpha=1)

        plt.plot(x_org, y_org, '.', color='green', alpha=1)
        plt.title(title)
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.legend()
        plt.show()


class QuantileRegression(RegressionFramework):
    def __init__(self, Xtr, ytr, Xte, yte, tau=.5):
        super().__init__(Xtr, ytr, Xte, yte)
        self.name = 'Quantile Regression'  # Model name
        self.tau = tau  # Delta, unique to Huber Loss Function

    def _equality_constr(self, X, y, n):
        """
        Slack variables to separate positive and negative residuals and input them into the
        program as equality constraint.
        """
        ecr1 = X
        ecr2 = X * -1
        ecr3 = np.identity(n)
        ecr4 = np.identity(n) * -1
        ecr = np.concatenate((ecr1, ecr2, ecr3, ecr4), axis=1)
        return ecr, y

    def _inequality_constr(self, X, y, n):
        icl = np.identity(n) * -1
        icr = np.zeros(n).reshape(-1, 1)
        return icl, icr

    def _objective(self, p, n, tau=.5, ):
        return np.concatenate((np.repeat(0, 2 * p), tau * np.repeat(1, n), (1 - tau) * np.repeat(1, n)))

    def fit(self):
        p, n = self.Xtr.shape[1], self.ytr.shape[0]

        # equality constraints
        eq = self._equality_constr(self.Xtr, self.ytr, n)
        A_eq, B_eq = eq[0], eq[1]

        # inequality constraints
        ub = self._inequality_constr(self.Xtr, self.ytr, A_eq.shape[1])
        A_ub, B_ub = ub[0], ub[1]

        sol = linprog(self._objective(p, n, self.tau), A_ub, B_ub, A_eq, B_eq, method='interior-point')
        self.param_opt = sol.x[0:p] - sol.x[p:2 * p]

    def plot_reg(self, title, x_test, x_org, y_org, ylim=(-2, 11), xlim=(-3, 3), tau_benchmark=True):
        """
        Plotting the Huber Regression with the option to vizualize multiple delta values
        """
        plt.figure(figsize=(10, 5))
        if tau_benchmark:
            for tau in [.1, .5, .8]:
                self.tau = tau
                self.fit()
                self.predict()
                plt.plot(x_test, self.prediction, '+', alpha=1, label=f'Tau: {tau} MSE: {self.mse}')
        else:
            plt.plot(x_test, self.prediction, '+', color='red', alpha=1)

        plt.plot(x_org, y_org, '.', color='green', alpha=1)
        plt.title(title)
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.legend()
        plt.show()


# For Debugging
if __name__ == '__main__':
    np.random.seed(42)
    X_train_poly, X_test_poly, y_errors_train, X_test, y_test, X = dgp()

    least_squares = LeastSquares(Xtr=X_train_poly, ytr=y_errors_train, Xte=X_test_poly, yte=y_test)
    least_squares.plot_loss(title='Least Squares Cost Function y = 0')
    least_squares.fit()
    least_squares.predict()
    least_squares.plot_reg(title='Least Squares Regression', x_test=X_test, x_org=X,
                           y_org=y_errors_train, xlim=(-7, 7))

    least_squares = LeastDeviation(Xtr=X_train_poly, ytr=y_errors_train, Xte=X_test_poly, yte=y_test)
    least_squares.plot_loss(title='Least Deviation Cost Function y = 0')
    least_squares.fit()
    least_squares.predict()
    least_squares.plot_reg(title='Least Deviation Regression', x_test=X_test, x_org=X, y_org=y_errors_train,
                           xlim=(-7, 7))

    least_squares = HuberRegression(Xtr=X_train_poly, ytr=y_errors_train, Xte=X_test_poly, yte=y_test, delta=0.01)
    least_squares.plot_loss(title='Huber Loss Cost Function y = 0')
    least_squares.fit()
    least_squares.predict()
    least_squares.plot_reg(title='Huber Loss Regression', x_test=X_test, x_org=X,
                           y_org=y_errors_train, xlim=(-7, 7))

    quantile_reg = QuantileRegression(Xtr=X_train_poly, ytr=y_errors_train, Xte=X_test_poly, yte=y_test, tau=0.5)
    quantile_reg.fit()
    quantile_reg.predict()
    quantile_reg.plot_reg(title='Quantile Regression', x_test=X_test, x_org=X, y_org=y_errors_train, xlim=(-7, 7))

    print('hello world')
