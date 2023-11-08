#%% Random data generator
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt



#%% Provide your implementation here
def gdSolver(X, y, EPOCH, alpha=0.2, tol=1e-4):
    # Inputs of the function: 
        # X: n x p matrix with n samples and p variables (excluding intercept)
        # y: n x 1 vector with n samples and one response
        # EPOCH: number of gradient descent iterations
        # alpha: learning rate (i.e., step size), default value 0.2
        # tol: tolerance for early stop, default value 0.0001

    # n number of samples and p number of predictors
    n, p = X.shape
    # initialize the weight matrix to zeros (p X 1)
    beta = np.zeros(p).T
    # intercept initialized to 0 - scalar value
    intercept = 0

    # EPOCH is the number of iterations to perform gradient descent
    for each_epoch in range(EPOCH):
        # make predictions
        y_pred = X @ beta + intercept

        # calculate gradients
        gradient_beta = (-2/n) * X.T @ (y - y_pred)
        gradient_intercept = (-2/n) * np.sum(y - y_pred)

        # update the model coefficients
        beta -= alpha * gradient_beta
        intercept -= alpha * gradient_intercept

        # stopping criteria
        if np.linalg.norm(gradient_intercept) + abs(gradient_intercept) < tol:
            # converging to a minimum of the cost function
            break

    # Outputs of the function:
        # beta: p x 1 vector with p coefficients
        # b: scalar (intercept)
    
    # TODO: implement below

    # return beta, b
    return beta, intercept
    #pass

def exactSolver(X, y):
    # Inputs of the function: 
        # X: n x p matrix with n samples and p variables (excluding intercept)
        # y: n x 1 vector with n samples and one response
    # Outputs of the function:
        # beta: p x 1 vector with p coefficients
        # b: scalar (intercept)

    # TODO: implement below
    # n number of samples and p number of predictors
    n, p = X.shape
    # adding intercept to the feature matrix
    feature_matrix = np.hstack((np.ones((n, 1)), X))
    # calculate the coefficients
    beta = np.linalg.inv(feature_matrix.T @ feature_matrix) @ feature_matrix.T @ y
    # extract intercept from first element of the array
    b = beta[0]
    # extract coefficients from second element of the array
    beta = beta[1:]
    return beta, b
    # return beta, b
    #pass

#%% Validation
n = 100
p = 10
alpha = 0.01
n_samples, n_features = n, p
rng = np.random.RandomState(0)
noise = rng.normal(loc=0.0, scale=10, size=n_samples)
X, y = make_regression(n_samples, n_features, random_state=rng)
y = y + noise

beta_gd, b_gd = gdSolver(X, y, 10000, alpha=0.2, tol=1e-4)
y_gd = X @ beta_gd + b_gd
beta_exact, b_exact = exactSolver(X, y)
y_exact = X @ beta_exact + b_exact

plt.scatter(y_gd,y_exact)
plt.show()

# %% Print beta and c

print("Beta from gradient descent is {}".format(beta_gd))
print("Beta from exact solver is {}".format(beta_exact))
print("C from gradient descent is {}".format(b_gd))
print("C from exact solver is {}".format(b_exact))
