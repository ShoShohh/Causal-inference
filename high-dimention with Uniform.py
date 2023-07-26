# reference : https://amzn.asia/d/cJiluxp

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D

# Set seed for reproducibility
np.random.seed(1)

# Edit the sample size
size= 5000

# Structural Causal Model 1
def structural_causal_model1(error1, error2, error3, f_1):
    x1 = 0.5 * f_1 + error1
    x2 = 0.8 * x1 + 0.5 * f_1 + error2
    x3 = 0.8 * x1 + 0.5 * f_1 + error3
    return(x1, x2, x3)

# Structural Causal Model 2
def structural_causal_model2(error1, error2, error3, f_1):
    x2 = 0.5 * f_1 + error2
    x1 = 0.8 * x2 + 0.5 * f_1 + error1
    x3 = 0.8 * x2 + 0.5 * f_1 + error3
    return(x1, x2, x3)

# Structural Causal Model 3
def structural_causal_model3(error1, error2, error3,f_1):
    x1 = 0.9 * f_1 + error1
    x2 = 0.9 * f_1 + error2
    x3 = 0.9 * f_1 + error3
    return(x1, x2, x3)

# Function generates random numbers following a Uniform distribution
def generate_uniform_numbers(size, low, high):
    return np.random.uniform(low, high, size)

# Generate error variables and hidden variables
error1 = generate_uniform_numbers(size, 0, 1)
error2 = generate_uniform_numbers(size, 0, 1)
error3 = generate_uniform_numbers(size, 0, 1)
f_1 = generate_uniform_numbers(size, 0, 1)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Add labels and title to the plot
ax.scatter(structural_causal_model1(error1, error2, error3, f_1)[0],
           structural_causal_model1(error1, error2, error3, f_1)[1],
           structural_causal_model1(error1, error2, error3, f_1)[2], c='blue', marker='o', s=100)
ax.scatter(structural_causal_model2(error1, error2, error3, f_1)[0],
           structural_causal_model2(error1, error2, error3, f_1)[1],
           structural_causal_model2(error1, error2, error3, f_1)[2], c='green', marker='+', s=100)
ax.scatter(structural_causal_model3(error1, error2, error3, f_1)[0],
           structural_causal_model3(error1, error2, error3, f_1)[1],
           structural_causal_model3(error1, error2, error3, f_1)[2], c='red', marker='*', s=100)
ax.set(xlabel='x1', ylabel='x2', zlabel='x3')  # Set the labels for all axes in one line

# Set the view angle (elevation, azimuth)
ax.view_init(elev=20, azim=30)

plt.title('Scatter Plot with Error Variables following a Gaussian distribution')

# Display the plot
plt.show()