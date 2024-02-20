import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw2_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    n = x_train.shape[0]
    alpha = torch.zeros(n, requires_grad=True)

    for _ in range(num_iters):
        # Ensure computation requires gradient
        K = kernel(x_train, x_train)  # Kernel matrix
        y_outer = y_train.unsqueeze(1) * y_train.unsqueeze(0)  # shape [n, n]
        alpha_outer = alpha.unsqueeze(1) * alpha.unsqueeze(0)
        # First term: 0.5 * sum of element-wise product of alpha, y_outer, and K
        term1 = 0.5 * torch.sum(alpha_outer * y_outer * K)

        # Second term: sum of alpha
        term2 = torch.sum(alpha)

        # Objective function value
        objective = term1 - term2

        # Compute gradients
        objective.backward()

        # Update alpha using gradient descent
        with torch.no_grad():
            alpha -= lr * alpha.grad
            test = alpha.grad
            # Project alpha back into the feasible region C
            alpha.clamp_(min=0, max=c if c is not None else float('inf'))

        # Clear gradients for the next iteration
        alpha.grad.zero_()

    return alpha.detach()

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    with torch.no_grad():
        # Compute the kernel matrix between training and test examples
        K_test = kernel(x_train, x_test)

        # Compute the decision values for the test examples
        decision_values = torch.mv(K_test, alpha * y_train)

        # Make predictions based on the sign of the decision values
        predictions = torch.sign(decision_values)

    return predictions


# Get the XOR data
x_train, y_train = hw2_utils.xor_data()

# Set the hyperparameters
lr = 0.1
num_iters = 10000
c = None  # None for hard-margin SVM

# Choose the kernel function
kernel = hw2_utils.poly(degree=1)  # Polynomial kernel of degree 2

# Train the SVM
alpha = svm_solver(x_train, y_train, lr, num_iters, kernel, c)

# Make predictions on a new test point
x_test = torch.tensor([[1, -1]])  # Example test point
predictions = svm_predictor(alpha, x_train, y_train, x_test, kernel)

# Print the predictions
print(predictions)

