import numpy as np
import matplotlib.pyplot as plt


def plotData(x, y):
    """ Plots the data points x and y into a new figure;
        plotData(x, y) plots the data points and gives the figure axes labels of population and profit.
    """
    plt.figure()
    plt.plot(X, y, 'rx')
    plt.xlabel('Profit in $10,000s')
    plt.ylabel('Population of City in 10,000s')


# load comma separated txt file
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0][:, np.newaxis]
y = data[:, 1][:, np.newaxis]
plotData(X, y)


def computeCost(X, y, theta):
    """ computeCost(X, y, theta) computes the cost of using theta as the
   parameter for linear regression to fit the data points in X and y
    """

    m = len(y)
    J = 0.5 / m * np.sum((np.dot(X, theta)-y)**2)

    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    """ gradientDescent(X, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    """
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    print(J_history.shape)

    for iter in range(num_iters):

        # update theta
        theta = theta - alpha / m * np.dot(X.T, (np.dot(X, theta)-y))
        J_history[iter, 0] = computeCost(X, y, theta)

    return theta, J_history


m = len(y)
# X with bias items
X_wb = np.append(np.ones((m, 1)), X, axis=1)
theta0 = np.zeros((2, 1))

# gradient descent settings
iterations = 1500
alpha = 0.01

# test cost function
J = computeCost(X_wb, y, theta0)
print('Expected 32.07 here.')
print('Actual result is {:.2f}.'.format(J))

J = computeCost(X_wb, y, np.array([[-1], [2]]))
print('Expected 54.24 here.')
print('Actual result is {:.2f}.'.format(J))


# run gradient descent
theta, _ = gradientDescent(X_wb, y, theta0, alpha, iterations)

# test theta
print('Expected \n-3.6303 \n1.1664\n')
print('Actual result is \n{:.4f} \n{:.4f}'.format(theta[0, 0], theta[1, 0]))

# Plot the result
plotData(X, y)
plt.plot(X, np.dot(X_wb, theta))
plt.legend(['Training data', 'Linear regression'])

# Create theta grid
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i, j] = computeCost(X_wb, y, t)

J_valsT = J_vals.T

# # Contour plot
# plt.contour(theta0_vals, theta1_vals, J_valsT, np.logspace(-2, 3, 20))
# plt.plot(theta[0], theta[1], 'rx')

# Surface plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Xt, Yt = np.meshgrid(theta0_vals, theta1_vals)
surf = ax.plot_surface(Xt, Yt, J_valsT)
ax.view_init(30, -125)

plt.show()
