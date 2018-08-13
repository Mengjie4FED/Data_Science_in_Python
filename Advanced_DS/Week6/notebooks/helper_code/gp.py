# Standard imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Perform GP Regression on observed data
# Return predicted y values and errors
def fit_gp(x_data, y_data, x_sample):
    
    # Create Kernel
    kernel = ConstantKernel(12.5, (0.01, 100.0)) * RBF(1.5, (0.01, 100.0))

    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(x_data.reshape(x_data.shape[0], 1), y_data)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    return gp.predict(x_sample.reshape(x_sample.shape[0], 1),
                      return_std=True)

# Plot a single Gaussian Process fit
# Given: region of interest in x and function of interest
# If no observations, we simply plot the function
#
# 1. Generate observations over region of interest
# 2. Include error on observations
# 3. Generate sample points over region of interest
# 4. Perform GP rgression on observed data, predict on sample points
# 5. Plot observations, function, and predicted data and confidence interval
#
def make_gp_plot(axs, x_low, x_high, num_pts, myfunc):
    
    # Define random sed for reproducability
    rng=np.random.RandomState(seed=23)

    # Create sample points
    x_sample = np.linspace(x_low, x_high, 1000)

    # Plot sample points as the function of interest
    axs.plot(x_sample, myfunc(x_sample), 
             c=sns.xkcd_rgb["dusty purple"], linestyle=':',
             label='Function')
    
    if num_pts > 0:

        # Define error terms
        a = 1.25
        b = 1.75
        
        # Create observation data
        x_data = np.linspace(x_low, x_high, num_pts)
        y_data = myfunc(x_data)
        
        # Create error on dependent variable
        y_error = a + b * rng.rand(y_data.shape[0])
        y_data += rng.normal(0, y_error)
        
        # Plot observations
        axs.scatter(x_data, y_data, 
                    c=sns.xkcd_rgb["pale red"],
                    s=45, label='Observations')
        
        # Plot error bars on observed data
        axs.errorbar(x_data, y_data, xerr=None, yerr=y_error, fmt=None,
                     ecolor=sns.xkcd_rgb["pale red"])
        
        # Predict for sample points, request mean and std
        y_mean, y_std = fit_gp(x_data, y_data, x_sample)
        
        # Plot GP fit points
        axs.plot(x_sample, y_mean, 
                 c=sns.xkcd_rgb["denim blue"], linestyle='--',
                 label='GP Prediction')
        
        # Plot contour interval
        axs.fill_between(x_sample, 
                         (y_mean - 2. * y_std), (y_mean + 2. * y_std),
                         color=sns.xkcd_rgb["windows blue"],
                         alpha=.1, label='95% confidence interval')
        
        # Set title based on number of observations
        axs.set_title(f'GP Regression ({num_pts} Observations)')
        
    else: # No observations so we are simply plotting the function
        
        # Set title as functional view only
        axs.set_title('Function of Interest')
    
    # Decorate plot
    axs.set_xlabel('$X$')
    axs.set_ylabel('$Y$')
    axs.legend(loc='lower left')

    # Cleanup
    sns.despine(ax=axs, offset = 2, trim=True)